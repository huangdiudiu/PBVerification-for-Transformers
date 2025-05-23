# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch

class Controller:
    def __init__(self, args, eps,index=None):
        self.args = args
        self.layers = []
        self.p = args.p
        self.q = 1. / (1. - 1. / args.p) if args.p != 1 else float("inf") # dual norm
        if args.p>=5:
            self.p = float("inf")
            self.q=1
        self.eps = eps
        self.perturbed_words = args.perturbed_words
        self.index=index
        self.opt=False
        self.opt_vars=[]
        self.lb=None
        self.ub=None

    def clear_opt_vars(self):
        self.opt_vars=[]

    def append(self, layer):
        self.layers.append(layer)

    def pop(self):
        self.layers.pop()
    def remove_layer(self, layer):
        self.layers.remove(layer)

    def compute(self, length, dim,device=None,flag=False):
        if device is None:
            device = self.args.device

        self.lb = torch.zeros(length, dim).to(device)
        self.ub = self.lb.detach().clone()
        self.final_lw = self.final_uw = None
        self.final_lb = self.final_ub = None
        for layer in self.layers[::-1]:
            if layer.lw is not None:
                layer.backward(device,flag=flag)

    def get_all_opt_vars(self,device=None):
        if device is None:
            device = self.args.device
        self.clear_opt_vars()
        for layer in self.layers[::-1]:
            if layer.need_pass:
                layer.backward_get_vars(device)

    def concretize_l(self, lw=None):
        return -self.eps * torch.norm(lw, p=self.q, dim=-1)

    def concretize_u(self, uw=None):      
        return self.eps * torch.norm(uw, p=self.q, dim=-1)

    def concretize(self, lw, uw):
        if self.perturbed_words == 2:
            assert(len(lw.shape) == 3)
            half = lw.shape[-1] // 2
            return \
                self.concretize_l(lw[:, :, :half]) + self.concretize_l(lw[:, :, half:]),\
                self.concretize_u(uw[:, :, :half]) + self.concretize_u(uw[:, :, half:])
        elif self.perturbed_words == 1:
            return self.concretize_l(lw), self.concretize_u(uw)
        else:
            raise NotImplementedError

    def concretize1(self, lw, uw):
        l=torch.sum(-self.eps * torch.norm(lw[:,:,self.index,:], p=self.q, dim=-1), dim=-1)
        u = torch.sum(self.eps * torch.norm(uw[:, :, self.index, :], p=self.q, dim=-1), dim=-1)
        return l, u

