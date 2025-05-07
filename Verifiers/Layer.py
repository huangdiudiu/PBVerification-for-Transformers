# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import torch.nn as nn
import math, time
from Verifiers.Controller import Controller
from Verifiers.Bounds import Bounds
epsilon = 1e-12

class Layer:
    def __init__(self, args, controller, length, dim, bounds=None):
        self.args = args
        self.controller = controller
        self.length = length
        self.dim = dim
        self.use_forward = args.method == "baf"
        self.parents = []
        self.l = self.u = None
        # for back propagation
        self.lw = self.uw = None
        self.need_pass=False
        # bounds of the layer 
        self.final_lw = self.final_uw = None 
        self.final_lb = self.final_ub = None
        self.empty_cache = args.empty_cache
        self.controller.append(self)

        # bounds obtained from the forward framework
        if bounds is not None:
            self.back = False
            self.bounds = bounds

            self.l, self.u = bounds.concretize()
            self.final_lw, self.final_lb = bounds.lw.transpose(-1, -2), bounds.lb
            self.final_uw, self.final_ub = bounds.uw.transpose(-1, -2), bounds.ub

            # incompatible format (batch)
            self.l = self.l[0]
            self.u = self.u[0]
            self.final_lw = self.final_lw[0]
            self.final_lb = self.final_lb[0]
            self.final_uw = self.final_uw[0]
            self.final_ub = self.final_ub[0]
        else:
            self.back = True
            
    def print(self, message):
        print(message)
        print("shape (%d, %d)" % (self.length, self.dim))
        print("mean abs %.5f %.5f" % (torch.mean(torch.abs(self.l)), torch.mean(torch.abs(self.u))))
        print("diff %.5f %.5f %.5f" % (torch.min(self.u - self.l), torch.max(self.u - self.l), torch.mean(self.u - self.l)))
        print("min", torch.min(self.l))
        print("max", torch.max(self.u))
        print()

    def add_edge(self, edge):
        self.parents.append(edge)

    def pop_layer(self):
        self.controller.pop()


    def next(self, edge, length=None, dim=None,device=None):
        if length is None:
            length = self.length
        if dim is None:
            dim = self.dim

        layer = Layer(self.args, self.controller, length, dim)
        layer.add_edge(edge)
        layer.compute(device)
        return layer

    def compute(self,device=None,flag=False):
        if device is None:
            device = self.args.device
        if self.use_forward:
            if torch.cuda.is_available():
                self.lw = torch.eye(self.dim).cuda()\
                    .reshape(1, self.dim, self.dim).repeat(self.length, 1, 1)
            else:
                self.lw = torch.eye(self.dim)\
                    .reshape(1, self.dim, self.dim).repeat(self.length, 1, 1)
        else:
            self.lw = torch.eye(self.length * self.dim).reshape(self.length, self.dim, self.length,
                                                                                     self.dim).to(device)
        self.uw = self.lw.detach().clone()
        self.controller.compute(self.length, self.dim,device=device,flag=flag)
        self.l, self.u = self.controller.lb, self.controller.ub
        self.final_lw, self.final_uw = self.controller.final_lw, self.controller.final_uw
        self.final_lb, self.final_ub = self.controller.final_lb, self.controller.final_ub

    def get_all_opt_vars(self,device=None): #store in self.controller.opt_vars
        self.need_pass=True
        if device is None:
            device = self.args.device
        self.controller.get_all_opt_vars(device=device)

    def optimize(self,device):

        if device is None:
            device = self.args.device
        self.controller.opt = True
        self.get_all_opt_vars(device=device)
        #print("opt vars number:{}".format(len(self.controller.opt_vars)))

        # for var in self.controller.opt_vars:
        #     var.data.zero_()


        # edge = self.parents[-1]
        # X0_1_opt,X0_2_opt=edge.X0_1_opt,edge.X0_2_opt
        # device = X0_1_opt.device
        # l_a=l_a.reshape(X0_1.shape)
        # u_a=u_a.reshape(X0_1.shape)
        # l_b=l_b.reshape(X0_2.shape)
        # u_b=u_b.reshape(X0_2.shape)
        optimizer = torch.optim.Adam(self.controller.opt_vars, lr=0.1)
        #optimizer = torch.optim.Adam([X0_1_opt,X0_2_opt], lr=0.1)

        best_obj = torch.tensor(float("inf"), dtype=self.l[0][0].dtype).to(device)
        best_l, best_u = self.l.detach().clone(), self.u.detach().clone()
        best_opt = [v.detach().clone() for v in self.controller.opt_vars]
        #best_diff=(self.u-self.l).sum()
        #diff_sum=best_diff
        #best_l,best_u=self.l, self.u
        # best_final_lw,best_final_uw=self.final_lw, self.final_uw
        # best_final_lb,best_final_ub=self.final_lb, self.final_ub
        # best_opt1,best_opt2=X0_1_opt.clone(), X0_2_opt.clone()

        it_num=20

        for it in range(it_num):

            # X0_1.data.clamp_(min=l_a, max=u_a)
            # X0_2.data.clamp_(min=l_b, max=u_b)

            optimizer.zero_grad()
            self.compute(device)

            _obj=(self.u-self.l).sum()

            if _obj < best_obj:
                best_obj = _obj.detach().clone()
                best_l, best_u = self.l.detach().clone(), self.u.detach().clone()
                best_opt=[v.detach().clone() for v in self.controller.opt_vars]
                best_final_lw, best_final_uw=self.final_lw.detach().clone(), self.final_uw.detach().clone()
                best_final_lb, best_final_ub=self.final_lb.detach().clone(), self.final_ub.detach().clone()
                #best_opt1, best_opt2=X0_1_opt.clone(), X0_2_opt.clone()

            _obj.backward()
            optimizer.step()
            #print(edge.X0_1_opt.sum().item(), edge.X0_2_opt.sum().item())

        #del _obj
        if self.empty_cache:
            torch.cuda.empty_cache()

        with torch.no_grad():
            device = self.args.device
            self.comprehensive_l, self.comprehensive_u=best_l.to(device), best_u.to(device)
            # self.final_lw, self.final_uw = best_final_lw.to(device), best_final_uw.to(device)
            # self.final_lb, self.final_ub=best_final_lb.to(device), best_final_ub.to(device)
            # edge.X0_1_opt, edge.X0_2_opt=best_opt1.to(device), best_opt2.to(device)
            for a, b in zip(self.controller.opt_vars, best_opt):
                #a.zero_()
                a.copy_(b)
            # for var in self.controller.opt_vars:
            #     var.data.zero_()
            self.compute(device)
            # edge.rebuild(edge.X0_1_opt,edge.X0_2_opt,device=device)
            # self.controller.opt = False
            # self.u = torch.min(self.comprehensive_u,self.optimized_u)
            # self.l = torch.max(self.comprehensive_l, self.optimized_l)
            self.u = self.comprehensive_u
            self.l = self.comprehensive_l
        self.controller.opt = False

    def optimize_ul(self,isLB=True,device=None):

        self.controller.opt = True
        if device is None:
            device = self.args.device

        self.get_all_opt_vars(device=device)
        #self.compute(device)

        backup_opt_vars=[v.detach().clone() for v in self.controller.opt_vars]

        for var in self.controller.opt_vars:
            var.data.zero_()

        optimizer = torch.optim.Adam(self.controller.opt_vars, lr=0.1)

        if isLB:
            self.optimized_l=self.l.detach().clone()
        else:
            self.optimized_u=self.u.detach().clone()

        best_obj = torch.tensor(float("inf"), dtype=self.optimized_l[0][0].dtype).to(device)
        it_num = 20
        for it in range(it_num):
            optimizer.zero_grad()
            self.compute(device)
            if isLB:
                _obj = -self.l.sum()
            else:
                _obj = self.u.sum()
            #print(_obj.item())
            _obj.backward()
            optimizer.step()
            #print(edge.X0_1_opt.sum().item(), edge.X0_2_opt.sum().item())
            if torch.allclose(_obj, best_obj, atol=1e-8):
                break

            if _obj < best_obj:
                best_obj = _obj.detach().clone()
                if isLB:
                    self.optimized_l = self.l.detach().clone()
                else:
                    self.optimized_u = self.u.detach().clone()

        if self.empty_cache:
            torch.cuda.empty_cache()

        with torch.no_grad():
            device = self.args.device
            if isLB:
                self.optimized_l = self.optimized_l.to(device)
            else:
                self.optimized_u = self.optimized_u.to(device)
            for a, b in zip(self.controller.opt_vars, backup_opt_vars):
                a.copy_(b)
        self.controller.opt = False



    def last_layer_optimize(self,label,device):

        self.controller.opt = True
        if device is None:
            device = self.args.device

        self.get_all_opt_vars(device=device)
        #print("last opt vars number:{}".format(len(self.controller.opt_vars)))
        # for var in self.controller.opt_vars:
        #     var.data.zero_()


        optimizer = torch.optim.Adam(self.controller.opt_vars, lr=0.1)


        #bset_opt_vars=self.controller.opt_vars
        if label == 0:
            _obj = -self.l[0][0]
        else:
            _obj = self.u[0][0]

        best_obj=torch.tensor(float("inf"), dtype=self.l[0][0].dtype).to(device)
        best_l, best_u = self.l.clone(), self.u.clone()
        bset_opt_vars = [t.clone() for t in self.controller.opt_vars]
        best_final_lw, best_final_uw = self.final_lw.clone(), self.final_uw.clone()
        best_final_lb, best_final_ub = self.final_lb.clone(), self.final_ub.clone()

        best_l, best_u = self.l, self.u
        it_num = 30
        for it in range(it_num):
            ######
            # prev_vars = [var.detach().clone() for var in self.controller.opt_vars]
            optimizer.zero_grad()
            self.compute(device)
            if label == 0:
                _obj = -self.l[0][0]
            else:
                _obj = self.u[0][0]

            #print(f"{self.l[0][0].item():.7f}, {self.u[0][0].item():.7f}")
            # for i, (prev, curr) in enumerate(zip(prev_vars, self.controller.opt_vars)):
            #     if torch.equal(prev, curr):
            #         print(f"Variable {i} was NOT updated.")
            #     else:
            #         print(f"Variable {i} was updated.")

            if _obj<=best_obj:
                best_obj=_obj.clone()
                best_l, best_u=self.l.clone(), self.u.clone()
                bset_opt_vars=[t.clone() for t in self.controller.opt_vars]
                best_final_lw, best_final_uw=self.final_lw.clone(), self.final_uw.clone()
                best_final_lb, best_final_ub=self.final_lb.clone(), self.final_ub.clone()
            if best_obj<0:
                break

            _obj.backward()
            optimizer.step()


        #del _obj
        if self.empty_cache:
            torch.cuda.empty_cache()

        with torch.no_grad():
            device = self.args.device
            self.l, self.u=best_l.to(device), best_u.to(device)
            self.final_lw, self.final_uw = best_final_lw.to(device), best_final_uw.to(device)
            self.final_lb, self.final_ub=best_final_lb.to(device), best_final_ub.to(device)
            #edge.X0_1_opt, edge.X0_2_opt=best_opt1.to(device), best_opt2.to(device)
            #edge.rebuild(edge.X0_1_opt,edge.X0_2_opt,device=device)
        self.controller.opt = False




    def backward_buffer(self, lw, uw):
        if self.lw is None:
            self.lw, self.uw = lw, uw
        else:
            self.lw =self.lw+lw
            self.uw =self.uw+ uw
    def need_pass_buffer(self):
        self.need_pass = True

    def backward(self,device=None,flag=False):
        if device is None:
            device = self.args.device
        self.lw=self.lw.to(device)
        self.uw=self.uw.to(device)

        if self.back:
            for edge in self.parents:
                edge.backward(self.lw, self.uw,flag=flag)
        else:
            bounds_l = self.bounds.matmul(self.lw)\
                .add(self.controller.lb.unsqueeze(0))
            bounds_u = self.bounds.matmul(self.uw)\
                .add(self.controller.ub.unsqueeze(0))
            bounds = Bounds(
                bounds_l.args, bounds_l.p, bounds_l.eps,
                lw = bounds_l.lw, lb = bounds_l.lb,
                uw = bounds_u.uw, ub = bounds_u.ub   
            )
            self.controller.final_lw = bounds.lw[0].transpose(1, 2)
            self.controller.final_uw = bounds.uw[0].transpose(1, 2)
            self.controller.final_lb = bounds.lb[0]
            self.controller.final_ub = bounds.ub[0]
            self.controller.lb, self.controller.ub = bounds.concretize()
            self.controller.lb = self.controller.lb[0]
            self.controller.ub = self.controller.ub[0]

        del(self.lw)
        del(self.uw)
        if self.empty_cache:
            torch.cuda.empty_cache()
        self.lw, self.uw = None, None

    def backward_get_vars(self,device=None):
        for edge in self.parents:
            edge.backward_get_vars(device)
        self.need_pass=False