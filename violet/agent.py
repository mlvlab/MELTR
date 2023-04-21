from meltr import MELTROptimizer
from lib import *

class Agent_Base:
    def __init__(self, args, model):
        super().__init__()
        
        self.args, self.model = args, model
        
        self.loss_func = T.nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.optzr = T.optim.AdamW(self.model.parameters(), lr=args['lr'], betas=(0.9, 0.98), weight_decay=args['decay'])
        self.scaler = T.cuda.amp.GradScaler()
        
class Agent_Base_MELTR:
    def __init__(self, args, model, aux_model=None):
        super().__init__()
        
        self.args, self.model, self.aux_model = args, model, aux_model
        
        self.loss_func = T.nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.optzr = T.optim.AdamW(self.model.parameters(), lr=args['lr'], betas=(0.9, 0.98), weight_decay=args['decay'])
        self.scaler = T.cuda.amp.GradScaler()
        
        if aux_model is not None:
            self.aux_optzr = T.optim.AdamW(self.aux_model.parameters(), lr=args['meltr_lr'], betas=(0.9, 0.98), weight_decay=args['meltr_decay'])
            self.meta_optim = MELTROptimizer(meta_optimizer=self.aux_optzr, max_grad_norm=args['max_grad_norm'])