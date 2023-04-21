# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging

logger = logging.getLogger(__name__)

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * torch.tensor(x)))

def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

def warmup_linear_down(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    return abs(1-x)

def none(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    return 1.0

SCHEDULES = {
    'warmup_cosine':   warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear':   warmup_linear,
    'warmup_linear_down': warmup_linear_down,
    'none': none,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup='warmup_linear', warmup_proportion=-1, milestones=None, milestones_proportion=0.1,
                 t_total=-1, t_warmup=-1, b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if warmup not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(warmup))
        if not 0.0 <= warmup_proportion < 1.0 and not warmup_proportion == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup_proportion))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, warmup=warmup, warmup_proportion=warmup_proportion, milestones=milestones, milestones_proportion=milestones_proportion,
                        t_total=t_total, t_warmup=t_warmup, b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    return [0]

                lr_scheduled = group['lr']
                if group['milestones'] and state['epoch'] != -1 and group['warmup'] != "none":
                    lr_scheduled = group['lr'] * (group['milestones_proportion'] ** len([v for v in group['milestones'] if v <= state['epoch']]))

                if (group['t_warmup'] != -1):
                    schedule_fct = SCHEDULES[group['warmup']]
                    step = state['step'] % (group['t_warmup']+1)
                    progress = step/(group['t_warmup']+1)
                    lr_scheduled = lr_scheduled * schedule_fct(progress, group['warmup_proportion'])

                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None, epoch=-1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)
                state['step'] += 1
                state['epoch'] = epoch

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # next_m.mul_(beta1).add_(1 - beta1, grad) --> pytorch 1.7
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad) --> pytorch 1.7
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                lr_scheduled = group['lr']
                if group['milestones'] and epoch != -1 and group['warmup'] != "none":
                    lr_scheduled = group['lr'] * (group['milestones_proportion'] ** len([v for v in group['milestones'] if v <= epoch]))

                if (group['t_warmup'] != -1):
                    schedule_fct = SCHEDULES[group['warmup']]
                    step = state['step'] % (group['t_warmup']+1)
                    progress = step/(group['t_warmup']+1)
                    lr_scheduled = lr_scheduled * schedule_fct(progress, group['warmup_proportion'])


                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)




        return loss