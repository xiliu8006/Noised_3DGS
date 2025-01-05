import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import List, Optional



class LieGroupSGD(Optimizer):
    def __init__(self, params, lr=None, gamma=0.9, dampening=0, scheme='heavy_ball'):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        assert scheme in ['momentumless', 'heavy_ball', 'NAG_SC'], 'scheme not correct'
        

        defaults = dict(lr=lr, gamma=gamma, dampening=dampening, scheme=scheme)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            h = group['lr']
            gamma = group['gamma']
            dampening = group['dampening']
            scheme = group['scheme']
            for g in group['params']:
                if g.grad is None:
                    continue
                param_state = self.state[g]
                trivialized_grad=g.T@g.grad-g.grad.T@g
                if 'xi' not in param_state:
                    # param_state['xi']=-gamma/(1-gamma*h)*trivialized_grad
                    param_state['xi']=torch.zeros_like(trivialized_grad)
                xi = param_state['xi']
                if 't' not in param_state:
                    param_state['t']=0
                
                if scheme=='heavy_ball':
                    xi_new=(1-gamma*h)*xi-h*trivialized_grad
                    xi.copy_(xi_new)
                elif scheme=='NAG_SC':
                    if 'trivialized_grad_last' not in param_state:
                        param_state['trivialized_grad_last']=trivialized_grad
                    trivialized_grad_last=param_state['trivialized_grad_last']
                    
                    xi_new=(1-gamma*h)*xi-(1-gamma*h)*h*(trivialized_grad-trivialized_grad_last)-h*trivialized_grad
                    xi.copy_(xi_new)
                    param_state['trivialized_grad_last'].copy_(trivialized_grad)
                elif scheme=='momentumless':
                    xi.copy_(-trivialized_grad)
                else:
                    raise NotImplementedError()
                g.copy_(g@torch.matrix_exp(h*xi))
                param_state['t']+=h
        return loss