from torch.nn.utils import clip_grad_norm_
from modules.modeling_MELTR import MELTRgrad

class MELTROptimizer:

    def __init__(self, meta_optimizer, max_grad_norm=10):
        self.meta_optimizer = meta_optimizer
        self.hypergrad = MELTRgrad()

        self.max_grad_norm = max_grad_norm

    def step(self, train_loss, val_loss, parameters, aux_params):
        self.zero_grad()

        hyper_gards = self.hypergrad.grad(
            loss_val=val_loss,
            loss_train=train_loss,
            aux_params=aux_params,
            params=parameters,
        )
        for p, g in zip(aux_params, hyper_gards):
            if g is not None:
                p.grad = -g

        if self.max_grad_norm is not None:
            clip_grad_norm_(aux_params, max_norm=self.max_grad_norm)

        self.meta_optimizer.step()
    def zero_grad(self):
        self.meta_optimizer.zero_grad()