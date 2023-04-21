from torch.nn.utils import clip_grad_norm_
import torch
from torch.nn import functional as F
from typing import Optional, Any
from torch import Tensor
from torch import nn
import torch

class MELTRgrad:
    def __init__(self):
        pass
    
    def grad(self, loss_train, loss_val, params, aux_params):
        dwdA = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )

        dwdT = torch.autograd.grad(
            loss_train,
            params,
            create_graph=True,
            allow_unused=True
        )

        temp_t, temp_a = [], []
        for t, a in zip(dwdT, dwdA):
            if a is None:
                continue
            temp_t.append(t)
            temp_a.append(a)


        v4 = torch.autograd.grad(
            tuple(temp_t), 
            aux_params, 
            grad_outputs=tuple(temp_a),
            allow_unused=True
        )

        return v4


class MELTROptimizer:
    def __init__(self, meta_optimizer, max_grad_norm=10):
        self.meta_optimizer = meta_optimizer
        self.hypergrad = MELTRgrad()

        self.max_grad_norm = max_grad_norm

    def step(self, train_loss, val_loss, parameters, aux_params):
        self.zero_grad()

        hyper_grads = self.hypergrad.grad(
            loss_train=train_loss,
            loss_val=val_loss,
            params=parameters,
            aux_params=aux_params,
        )

        for p, g in zip(aux_params, hyper_grads):
            if g is not None:
                p.grad = -g

        if self.max_grad_norm is not None:
            clip_grad_norm_(aux_params, max_norm=self.max_grad_norm)
        
        self.meta_optimizer.step()
        
    def zero_grad(self):
        self.meta_optimizer.zero_grad()


class MELTR(nn.Module):
    def __init__(self, t_dim, f_dim, i_dim, h1_dim, h2_dim, o_dim):
        super(MELTR, self).__init__()
        self.task_embedding = nn.Embedding(t_dim, h2_dim) 
        self.loss_fc1 = nn.Linear(i_dim, h1_dim)
        self.activation1 = nn.ReLU()
        self.loss_fc2 = nn.Linear(h1_dim, h2_dim)

        self.encoder = nn.TransformerEncoderLayer(d_model=h2_dim, nhead=8, batch_first=True, dim_feedforward=f_dim)
        self.fc1 = nn.Linear(h2_dim, o_dim, bias=False)



    def forward(self, x):
        scale_embedding = self.loss_fc2(self.activation1(self.loss_fc1(x)))
        input = scale_embedding + self.task_embedding.weight
        output = self.encoder(input)
        output = self.fc1(output.mean(1))
        return output