# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import time

import torch

from Verifiers import Verifier
from Verifiers.Layer import Layer
from Verifiers.Controller import Controller
from Verifiers.custom_SDP_solver import SDP_custom
import pdb
from multiprocessing import Pool, cpu_count
from Verifiers.utils import *
epsilon = 1e-12

# l_a = a.l.reshape(a.length, num_attention_heads, self.attention_head_size)
# u_a = a.u.reshape(a.length, num_attention_heads, self.attention_head_size)
# l_b = b.l.reshape(b.length, num_attention_heads, self.attention_head_size)
# u_b = b.u.reshape(b.length, num_attention_heads, self.attention_head_size)
# self.alpha_l = torch.zeros(a.length, num_attention_heads, b.length, self.attention_head_size)
# self.alpha_u = torch.zeros(a.length, num_attention_heads, b.length, self.attention_head_size)
# self.beta_l = torch.zeros(a.length, num_attention_heads, b.length, self.attention_head_size)
# self.beta_u = torch.zeros(a.length, num_attention_heads, b.length, self.attention_head_size)
# self.gamma_l = torch.zeros(a.length, num_attention_heads, b.length)
# self.gamma_u = torch.zeros(a.length, num_attention_heads, b.length)
#elf.alpha_l = self.alpha_l.reshape(a.length, num_attention_heads,b.length, self.attention_head_size)

#W_Q=W_Q.reshape(W_Q.shape[0],self.num_attention_heads,self.attention_head_size)
def test_solve(d_in,W,eps4_query):
    D_query_var = cvx.Variable(d_in, nonneg=True)
    W_param = cvx.Parameter((d_in, d_in))
    cons_PSD_d = cvx.diag(D_query_var) + W_param + W_param.T >> 0
    eps2_query = cvx.Parameter(d_in, nonneg=True)
    obj_d = cvx.Minimize(eps2_query @ D_query_var)
    prob_d = cvx.Problem(obj_d, [cons_PSD_d])

    eps2_query.value = eps4_query ** 2
    UNL_list = []
    # Repeat for lower and upper bounds
    for bnd in [0, 1]:
        Q_bound = {}
        time_start = time.time()
        W_param.value = -W if bnd == 1 else W
        obj_val = prob_d.solve(solver=cvx.MOSEK)
        time_end = time.time()
        Q_bound["D_query"] = D_query_var.value.copy()
        Q_bound["D_key"] = Q_bound["D_query"]
        Q_bound["Time"] = time_end - time_start
        print("For one diagnal case:{:.2f}".format(Q_bound["Time"]))
        Q_bound["Obj_val"] = obj_val
        UNL_list.append(Q_bound)
    return UNL_list


class SDP4QK:
    def __init__(self, d_in, solverNname):
        self.d_in = d_in
        self.solverNname=solverNname

    def diagSolve(self,W,eps4_query):
        D_query_var = cvx.Variable(self.d_in, nonneg=True)
        W_param = cvx.Parameter((self.d_in, self.d_in))
        cons_PSD_d = cvx.diag(D_query_var) + W_param + W_param.T >> 0
        eps2_query = cvx.Parameter(self.d_in, nonneg=True)
        obj_d = cvx.Minimize(eps2_query @ D_query_var)
        prob_d = cvx.Problem(obj_d, [cons_PSD_d])
        # for diagonal situation
        eps2_query.value = eps4_query ** 2
        UNL_list = []
        # Repeat for lower and upper bounds
        for bnd in [0, 1]:
            Q_bound={}
            time_start = time.time()
            W_param.value = W if bnd == 0 else -W
            obj_val = prob_d.solve(solver=self.solverNname,ignore_dpp=True)
            time_end = time.time()
            Q_bound["D_query"]= D_query_var.value
            Q_bound["D_key"] = Q_bound["D_query"]
            Q_bound["Time"] = time_end-time_start
            #print("For one diagnal case:{:.2f}".format(Q_bound["Time"]))
            Q_bound["Obj_val"] = obj_val
            UNL_list.append(Q_bound)
        return UNL_list

    def off_diagSolve(self,W,eps4_query,eps4_key):
        # for off-diagonal situation
        D_key_var = cvx.Variable(self.d_in, nonneg=True)
        D_query_var = cvx.Variable(self.d_in, nonneg=True)
        W_param = cvx.Parameter((self.d_in, self.d_in))
        cons_PSD_od = cvx.bmat([[cvx.diag(D_query_var), W_param], [W_param.T, cvx.diag(D_key_var)]]) >> 0
        eps2_key = cvx.Parameter(self.d_in, nonneg=True)
        eps2_query = cvx.Parameter(self.d_in, nonneg=True)
        obj_od = cvx.Minimize(eps2_query @ D_query_var + eps2_key @ D_key_var)
        prob_od = cvx.Problem(obj_od, [cons_PSD_od])
        # for off-diagonal situation
        eps2_query.value = eps4_query ** 2
        eps2_key.value = eps4_key ** 2
        UNL_list = []
        W_param.value = W
        time_start = time.time()
        obj_val = prob_od.solve(solver=self.solverNname,ignore_dpp=True)
        time_end = time.time()
        #print("For one off-dignal case:{:.2f}".format(time_end - time_start))
        # Repeat for lower and upper bounds
        # Solutions for lower and upper bounds are the same
        for bnd in [0, 1]:
            Q_bound = {}
            Q_bound["D_query"] = D_query_var.value
            Q_bound["D_key"] = D_key_var.value
            Q_bound["Time"] = time_end - time_start
            Q_bound["Obj_val"] = obj_val
            UNL_list.append(Q_bound)
        return UNL_list

    def sdp_sover(self,params):
        W=params['W']
        eps4_query=params['eps4_query']
        eps4_key=params['eps4_key']
        i,j=params['ij']
        h=params['h']
        #print("solving h:{},i:{},j:{}".format(h,i,j))
        if i==j:
            UNL_list=self.diagSolve(W, eps4_query)
        else:
            UNL_list=self.off_diagSolve(W, eps4_query,eps4_key)
        re={
            "i":i,
            "j":j,
            "h":h,
            "UNL_list":UNL_list
        }
        return re




class Edge:
    def __init__(self, args, controller):
        self.args = args
        self.controller = controller
        self.use_forward = args.method == "baf"
        self.empty_cache = args.empty_cache

    def backward(self, lw, uw,flag=False):
        raise NotImplementedError
    def backward_get_vars(self,device=None):
        raise NotImplementedError


class EdgeComplex(Edge):
    def __init__(self, args, controller):
        super(EdgeComplex, self).__init__(args, controller)

    def backward(self, lw, uw,flag=False):
        #print("EdgeComplex_backward")
        self.res.backward_buffer(lw, uw)
    def backward_get_vars(self,device=None):
        self.res.need_pass_buffer()

class EdgeDirect(Edge):
    def __init__(self, args, controller, par):
        super(EdgeDirect, self).__init__(args, controller)
        self.par = par

    def backward(self, lw, uw,flag=False):
        #print("EdgeDirect_backward")
        self.par.backward_buffer(lw, uw)
    def backward_get_vars(self,device=None):
        self.par.need_pass_buffer()

class EdgeInput(Edge):
    def __init__(self, args, controller, embeddings, index):
        super(EdgeInput, self).__init__(args, controller)
        self.embeddings = embeddings
        self.index = index
        self.perturbed_words = args.perturbed_words

    def backward(self, lw, uw,flag=False):
        device=lw.device
        if self.use_forward:
            if self.perturbed_words == 2:
                assert(type(self.index) == list)
                dim = lw.shape[2]
                if torch.cuda.is_available():
                    self.controller.final_lw = torch.zeros(lw.shape[0], lw.shape[1], dim * 2).cuda()
                    self.controller.final_uw = torch.zeros(lw.shape[0], lw.shape[1], dim * 2).cuda()
                else:
                    self.controller.final_lw = torch.zeros(lw.shape[0], lw.shape[1], dim * 2)
                    self.controller.final_uw = torch.zeros(lw.shape[0], lw.shape[1], dim * 2)
                self.controller.final_lw[self.index[0], :, :dim] = lw[self.index[0], :, :]
                self.controller.final_uw[self.index[0], :, :dim] = lw[self.index[0], :, :]
                self.controller.final_lw[self.index[1], :, dim:] = lw[self.index[1], :, :]
                self.controller.final_uw[self.index[1], :, dim:] = lw[self.index[1], :, :]
                _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)
            elif self.perturbed_words == 1:
                assert(type(self.index) == int)
                if torch.cuda.is_available():
                    self.controller.final_lw = torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2]).cuda()
                    self.controller.final_uw = torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2]).cuda()
                else:
                    self.controller.final_lw = torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2])
                    self.controller.final_uw = torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2])
                self.controller.final_lw[self.index, :, :] = lw[self.index, :, :]
                self.controller.final_uw[self.index, :, :] = uw[self.index, :, :]
                _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)
            else:
                raise NotImplementedError
        else:
            #print("EdgeInput_backward")
            '''
            assert(type(self.index) == int)
            self.controller.final_lw = lw[:, :, self.index, :]
            self.controller.final_uw = uw[:, :, self.index, :]
            '''
            self.controller.final_lw = lw
            self.controller.final_uw = uw
            embeddings=self.embeddings.detach().to(device)
            _lb = torch.sum(lw * embeddings.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
            _ub = torch.sum(uw * embeddings.unsqueeze(0).unsqueeze(0), dim=[-1, -2])

        self.controller.lb =self.controller.lb+ _lb
        self.controller.ub =self.controller.ub+ _ub

        self.controller.final_lb = self.controller.lb.reshape(_lb.shape).detach().clone()
        self.controller.final_ub = self.controller.ub.reshape(_lb.shape).detach().clone()
        '''
        l, u = self.controller.concretize(self.controller.final_lw, self.controller.final_uw)
        '''
        l, u = self.controller.concretize1(self.controller.final_lw, self.controller.final_uw)

        l = l.reshape(_lb.shape)
        u = u.reshape(_lb.shape)

        self.controller.lb =self.controller.lb+ l
        self.controller.ub =self.controller.ub+ u

        if self.empty_cache:
            torch.cuda.empty_cache()
    def backward_get_vars(self, device=None):
        pass
class EdgeSoftmax(EdgeComplex):
    def __init__(self, args, controller, par, num_attention_heads):
        super(EdgeSoftmax, self).__init__(args, controller)

        self.length = par.length
        self.num_attention_heads = num_attention_heads

        self.par = par       

        self.exp = self.par.next(EdgeExp(self.args, self.controller, self.par))

        if self.use_forward:
            raise NotImplementedError
        device=args.device
        ones = torch.ones(1, self.length, self.length).to(device)
        zeros = torch.zeros(num_attention_heads, self.length, self.length).to(device)

        w = torch.cat([
            ones, 
            torch.cat([zeros, ones], dim=0).repeat(num_attention_heads - 1, 1, 1)
        ], dim=0)\
        .reshape(num_attention_heads, num_attention_heads, self.length, self.length)\
        .permute(0, 2, 1, 3)\
        .reshape(num_attention_heads * self.length, num_attention_heads * self.length)

        self.sum = self.exp.next(EdgeDense(self.args, self.controller, self.exp, w=w, b=0.))

        self.res = self.exp.next(EdgeDivide(self.args, self.controller, self.exp, self.sum))


class EdgePooling(Edge):
    def __init__(self, args, controller, par):
        super(EdgePooling, self).__init__(args, controller)

        self.par = par
        self.length = par.length

    def backward(self, lw, uw,flag=False):
        #print("EdgePooling_backward")
        device=lw.device
        if self.use_forward:
            dim = 0
            zeros = torch.zeros(self.length - 1, lw.shape[1], lw.shape[2]).to(device)

        else:
            dim = 2
            zeros = torch.zeros(lw.shape[0], lw.shape[1], self.length - 1, lw.shape[3]).to(device)

        lw = torch.cat([lw, zeros], dim=dim)
        uw = torch.cat([uw, zeros], dim=dim)
        self.par.backward_buffer(lw, uw)

    def backward_get_vars(self, device=None):
        self.par.need_pass_buffer()
class EdgeDense(Edge):
    def __init__(self, args, controller, par, w=0., b=0., dense=None):
        super(EdgeDense, self).__init__(args, controller)
        self.par = par
        if dense is not None:
            w = dense.weight
            #b = dense.bias
            if dense.bias is not None:
                b = dense.bias
            else:
                b = 0.
        self.w = w
        if type(b) == float:
            self.b = torch.ones(w.shape[-1]).to(args.device) * b
        else:
            self.b = b

    def backward(self, lw, uw,flag=False):
        #print("EdgeDense_backward")
        device=lw.device
        w=self.w.to(device)
        _lw = torch.matmul(lw, w)
        _uw = torch.matmul(uw, w)

        if self.use_forward:
            self.controller.lb += torch.sum(lw * self.b, dim=-1)
            self.controller.ub += torch.sum(uw * self.b, dim=-1)
        else:
            b=self.b.to(device)
            self.controller.lb =self.controller.lb+ torch.sum(lw * b.reshape(1, 1, 1, -1), dim=[-1, -2])
            self.controller.ub =self.controller.ub+ torch.sum(uw * b.reshape(1, 1, 1, -1), dim=[-1, -2])

        return self.par.backward_buffer(_lw, _uw)
    def backward_get_vars(self, device=None):
        self.par.need_pass_buffer()
    
class EdgeActivation(Edge):
    def __init__(self, args, controller, par, par2=None):
        super(EdgeActivation, self).__init__(args, controller)
        self.par = par
        self.par2 = par2
        self.init_linear(args)


    def init_linear(self,args):
        self.mask_pos = torch.gt(self.par.l, 0).to(torch.float)
        self.mask_neg = torch.lt(self.par.u, 0).to(torch.float)
        self.mask_both = 1 - self.mask_pos - self.mask_neg 

        # element-wise for now
        shape = (self.par.length, self.par.dim)
        device=args.device
        self.lw = torch.zeros(shape).to(device)
        self.lb = torch.zeros(shape).to(device)
        self.uw = torch.zeros(shape).to(device)
        self.ub = torch.zeros(shape).to(device)


        if self.par2 is not None:
            shape = (self.par2.length, self.par2.dim)
            self.lw2 = torch.zeros(shape).to(device)
            self.lb2 = torch.zeros(shape).to(device)
            self.uw2 = torch.zeros(shape).to(device)
            self.ub2 = torch.zeros(shape).to(device)


    def add_linear(self, mask, type, k, x0, y0, second=False):
        if mask is None:
            mask = 1
        if type == "lower":
            if second:
                w_out, b_out = self.lw2, self.lb2
            else:
                w_out, b_out = self.lw, self.lb
        else:
            if second:
                w_out, b_out = self.uw2, self.ub2
            else:
                w_out, b_out = self.uw, self.ub  
        w_out +=  (mask * k)
        b_out += (mask * (-x0 * k + y0))

    def backward_par(self, lw, uw, self_lw, self_lb, self_uw, self_ub, par):
        device=lw.device
        self_lw=self_lw.to(device)
        self_lb=self_lb.to(device)
        self_uw=self_uw.to(device)
        self_ub=self_ub.to(device)
        mask_l = torch.gt(lw, 0.).to(torch.float)
        mask_u = torch.gt(uw, 0.).to(torch.float)
        if self.use_forward:
            _lw = mask_l * lw * self_lw.unsqueeze(1) +\
                (1 - mask_l) * lw * self_uw.unsqueeze(1)
            _lb = torch.sum(mask_l * lw * self_lb.unsqueeze(1) +\
                (1 - mask_l) * lw * self_ub.unsqueeze(1), dim=-1)
            _uw = mask_u * uw * self_uw.unsqueeze(1) +\
                (1 - mask_u) * uw * self_lw.unsqueeze(1)
            _ub = torch.sum(mask_u * uw * self_ub.unsqueeze(1) +\
                (1 - mask_u) * uw * self_lb.unsqueeze(1), dim=-1)
        else:

            _lb = torch.sum(mask_l * lw * self_lb.unsqueeze(0).unsqueeze(0) + \
                (1 - mask_l) * lw * self_ub.unsqueeze(0).unsqueeze(0), dim=[-1, -2])

            _ub = torch.sum(mask_u * uw * self_ub.unsqueeze(0).unsqueeze(0) + \
                (1 - mask_u) * uw * self_lb.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
            # def batched_sum(mask, w, self_low, self_high, batch_size=64):
            #     num_batches = (w.shape[0] + batch_size - 1) // batch_size  # 向上取整
            #     results = []
            #     for i in range(num_batches):
            #         # 取出当前 batch
            #         w_batch = w[i * batch_size:(i + 1) * batch_size]
            #         mask_batch = mask[i * batch_size:(i + 1) * batch_size]
            #
            #         batch_result = torch.sum(
            #             mask_batch * w_batch * self_low.unsqueeze(0).unsqueeze(0) +
            #             (1 - mask_batch) * w_batch * self_high.unsqueeze(0).unsqueeze(0),
            #             dim=[-1, -2]
            #         )
            #         results.append(batch_result)
            #         #print(i)
            #     return torch.cat(results, dim=0)
            #
            # _lb = batched_sum(mask_l, lw, self_lb, self_ub, batch_size=2)
            # _ub = batched_sum(mask_u, uw, self_ub, self_lb, batch_size=2)


            self.controller.lb = self.controller.lb + _lb
            self.controller.ub = self.controller.ub + _ub

            del _lb
            del _ub
            if self.empty_cache:
                torch.cuda.empty_cache()

            _lw = mask_l * lw * self_lw.unsqueeze(0).unsqueeze(0) +\
                (1 - mask_l) * lw * self_uw.unsqueeze(0).unsqueeze(0)

            _uw = mask_u * uw * self_uw.unsqueeze(0).unsqueeze(0) +\
                (1 - mask_u) * uw * self_lw.unsqueeze(0).unsqueeze(0)
            # def batched_operation(mask, w, self_low, self_high, batch_size=64):
            #     num_batches = (w.shape[0] + batch_size - 1) // batch_size  # 向上取整
            #
            #     results = []
            #     for i in range(num_batches):
            #         # 取出当前 batch
            #         w_batch = w[i * batch_size:(i + 1) * batch_size]
            #         mask_batch = mask[i * batch_size:(i + 1) * batch_size]
            #
            #         # 计算当前 batch
            #         batch_result = mask_batch * w_batch * self_low.unsqueeze(0).unsqueeze(0) + \
            #                        (1 - mask_batch) * w_batch * self_high.unsqueeze(0).unsqueeze(0)
            #
            #         results.append(batch_result)
            #
            #     return torch.cat(results, dim=0)
            #
            # # 使用 batch 计算
            # _lw = batched_operation(mask_l, lw, self_lw, self_uw, batch_size=2)
            # _uw = batched_operation(mask_u, uw, self_uw, self_lw, batch_size=2)

        par.backward_buffer(_lw, _uw)
        del _lw
        del _uw
        if self.empty_cache:
            torch.cuda.empty_cache()

    def backward(self, lw, uw,flag=False):
        #print("EdgeActivation_backward")
        self.backward_par(lw, uw, self.lw, self.lb, self.uw, self.ub, self.par)
        if self.par2 is not None:
            self.backward_par(lw, uw, self.lw2, self.lb2, self.uw2, self.ub2, self.par2)

    def backward_get_vars(self, device=None):
        self.par.need_pass_buffer()
        if self.par2 is not None:
            self.par2.need_pass_buffer()

# cannot be combined with the forward framework

class EdgeSoftmaxOpt(EdgeActivation):
    COMPARISON_EPS = 1e-6

    def return_tilde_diag(self, x, other):
        values = other.diagonal().to(self.device)
        res = (x - (values @ torch.ones_like(values).T)).to(self.device)
        res *= (torch.ones_like(res).to(self.device) - torch.eye(self.length, self.length).to(self.device))
        return res

    @staticmethod
    def batch_diag(tensor):
        if len(tensor.shape):
            return tensor.diag().unsqueeze(0)
        return torch.vmap(torch.diag, in_dims=0)(tensor)

    @staticmethod
    def concrete_l_er(x, l, u):
        # in case of u == l, one element has coefficient zero and the other one, so it sums to e^l.
        mask = (u != l)
        div_res_u = torch.ones_like(u)
        div_res_l = torch.zeros_like(l)

        div_res_u[mask] = torch.divide(u - x, u - l, rounding_mode=None)[mask]
        div_res_l[mask] = torch.divide(x - l, u - l, rounding_mode=None)[mask]

        chord_sum_exp_ub = ((div_res_u * torch.exp(l)) + (div_res_l * torch.exp(u)))
        return torch.reciprocal(chord_sum_exp_ub.sum(axis=-1, keepdim=True))

    @staticmethod
    def exp_difference_ratio(l_tilde, u_tilde):
        # put limit value in case of equality (using L'Hôpital))
        div_res = torch.exp(u_tilde)
        mask = (u_tilde != l_tilde)

        div_res[mask] = torch.divide(torch.exp(u_tilde) - torch.exp(l_tilde), u_tilde - l_tilde, rounding_mode=None)[mask]
        return div_res

    def concrete_l_lse(self, x_tilde, l_tilde, u_tilde):
        return torch.exp(x_tilde.squeeze(0).diag()).unsqueeze(-1) * EdgeSoftmaxOpt.concrete_l_er(x_tilde, l_tilde,
                                                                                                 u_tilde)

    def l_lse_partial_none_obj(self, x_tilde, l_tilde, u_tilde):
        return -torch.multiply(
            torch.exp(x_tilde.diagonal(dim1=1, dim2=2)).unsqueeze(-1) * torch.square(
                EdgeSoftmaxOpt.concrete_l_er(x_tilde, l_tilde, u_tilde)),
            EdgeSoftmaxOpt.exp_difference_ratio(l_tilde, u_tilde))

    def l_lse_b(self, c, l, u):
        u_tilde = self.return_tilde_diag(u, l)
        l_tilde = self.return_tilde_diag(l, u)
        c_tilde = self.return_tilde_diag(c, c)

        assert (torch.all(u_tilde >= c_tilde))
        assert (torch.all(c_tilde >= l_tilde))

        # f(c)
        l_lse = self.concrete_l_lse(c_tilde, l_tilde, u_tilde).squeeze(0).squeeze(-1)

        # sum of partial derivatives times c
        l_lse_b = self.l_lse_partial_none_obj(c_tilde, l_tilde, u_tilde)
        l_lse_b += EdgeSoftmaxOpt.batch_diag(l_lse).to(self.device)

        l_lse_b *= c.to(self.device)
        return torch.subtract(l_lse, l_lse_b.sum(axis=-1))

    def l_lse_w(self, c, l, u):
        u_tilde = self.return_tilde_diag(u, l)
        l_tilde = self.return_tilde_diag(l, u)
        c_tilde = self.return_tilde_diag(c, c)

        assert (torch.all(u_tilde >= c_tilde))
        assert (torch.all(c_tilde >= l_tilde))

        l_lse = self.concrete_l_lse(c_tilde, l_tilde, u_tilde).squeeze(0).squeeze(-1)

        # partial derivatives
        l_lse_w = self.l_lse_partial_none_obj(c_tilde, l_tilde, u_tilde)
        l_lse_w += EdgeSoftmaxOpt.batch_diag(l_lse).to(self.device)
        return l_lse_w

    @staticmethod
    def bound_p(bound_tilde):
        return torch.reciprocal((torch.exp(bound_tilde)).sum(axis=-1, keepdims=True))

    def concrete_u_lse(self, x_tilde, l_tilde, u_tilde):
        p_ub = EdgeSoftmaxOpt.bound_p(l_tilde).squeeze(-1).to(self.device)
        p_lb = EdgeSoftmaxOpt.bound_p(u_tilde).squeeze(-1).to(self.device)

        assert (torch.all(p_ub >= p_lb))

        log_ub = torch.log(p_ub)
        log_lb = torch.log(p_lb)

        numerator = (p_lb * log_ub) - (p_ub * log_lb) - torch.multiply(p_ub - p_lb,
                                                                       torch.tensor(torch.logsumexp(x_tilde, axis=-1)))
        denominator = log_ub - log_lb
        result = p_ub
        mask = (torch.abs(p_ub - p_lb) > EdgeSoftmaxOpt.COMPARISON_EPS)
        result[mask] = torch.divide(numerator, denominator, rounding_mode=None)[mask]
        return result


    def u_lse_partial_none_obj(self, x, l_tilde, u_tilde):
        p_ub = EdgeSoftmaxOpt.bound_p(l_tilde).to(self.device)
        p_lb = EdgeSoftmaxOpt.bound_p(u_tilde).to(self.device)
        assert (torch.all(p_ub >= p_lb))

        # if p_ub=p_lb, use 0 as derivative
        mask = (torch.abs(p_ub - p_lb) > EdgeSoftmaxOpt.COMPARISON_EPS)
        part1 = torch.zeros_like(p_ub)
        part1[mask] = torch.divide(p_lb - p_ub, torch.log(p_ub) - torch.log(p_lb))[mask]
        part2 = torch.softmax(torch.Tensor(x), dim=-1)

        return torch.multiply(part1, part2)

    def u_lse_partial_obj_add(self, l_tilde, u_tilde):
        p_ub = EdgeSoftmaxOpt.bound_p(l_tilde).to(self.device)
        p_lb = EdgeSoftmaxOpt.bound_p(u_tilde).to(self.device)
        assert (torch.all(p_ub >= p_lb))

        # if p_ub=p_lb, use 0 as derivative
        result = torch.zeros_like(p_ub)
        mask = (torch.abs(p_ub - p_lb) > EdgeSoftmaxOpt.COMPARISON_EPS)
        result[mask] = torch.divide(p_ub - p_lb, torch.log(p_ub) - torch.log(p_lb))[mask]

        return result.squeeze(-1)

    def u_lse_b(self, c, l, u):
        u_tilde = self.return_tilde_diag(u, l)
        l_tilde = self.return_tilde_diag(l, u)
        c_tilde = self.return_tilde_diag(c, c)

        assert (torch.all(u_tilde >= c_tilde))
        assert (torch.all(c_tilde >= l_tilde))

        # f(c)
        u_lse = self.concrete_u_lse(c_tilde, l_tilde, u_tilde)

        # sum of partial derivatives times c
        u_lse_b = self.u_lse_partial_none_obj(c, l_tilde, u_tilde)
        u_lse_b += EdgeSoftmaxOpt.batch_diag(self.u_lse_partial_obj_add(l_tilde, u_tilde)).to(self.device)
        u_lse_b *= c
        return torch.subtract(u_lse, u_lse_b.sum(axis=-1))

    def u_lse_w(self, c, l, u):
        u_tilde = self.return_tilde_diag(u, l)
        l_tilde = self.return_tilde_diag(l, u)
        c_tilde = self.return_tilde_diag(c, c)

        assert (torch.all(u_tilde >= c_tilde))
        assert (torch.all(c_tilde >= l_tilde))

        # partial derivatives
        u_lse_w = self.u_lse_partial_none_obj(c, l_tilde, u_tilde)
        u_lse_w += EdgeSoftmaxOpt.batch_diag(self.u_lse_partial_obj_add(l_tilde, u_tilde)).to(self.device)
        return u_lse_w

    def __init__(self, args, controller, par, num_attention_heads, device):
        super(EdgeSoftmaxOpt, self).__init__(args, controller, par)
        self.device = device
        self.length = par.length
        self.num_attention_heads = num_attention_heads

        self.par = par

        l, u = self.par.l, self.par.u
        if self.use_forward:
            raise NotImplementedError

        time_start = time.time()
        l = l.reshape(self.length, self.num_attention_heads, self.length)
        u = u.reshape(l.shape)
        m = (u + l) / 2.0

        lw = torch.zeros_like(l).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.length,
                                                                                  self.num_attention_heads, self.length)
        uw = torch.zeros_like(lw)
        lb = torch.zeros_like(l)
        ub = torch.zeros_like(u)

        for i in range(l.shape[0]):
            for j in range(l.shape[1]):
                l_col = l[i, j].unsqueeze(0).repeat(self.length, 1).unsqueeze(0).to(self.device)
                u_col = u[i, j].unsqueeze(0).repeat(self.length, 1).unsqueeze(0).to(self.device)
                m_col = m[i, j].unsqueeze(0).repeat(self.length, 1).unsqueeze(0).to(self.device)

                lw[i, j, :, i, j] = self.l_lse_w(m_col, l_col, u_col)  # shape = (1,20,20)
                uw[i, j, :, i, j] = self.u_lse_w(m_col, l_col, u_col)
                lb[i, j] = self.l_lse_b(m_col, l_col, u_col)  # shape = (1,20)
                ub[i, j] = self.u_lse_b(m_col, l_col, u_col)

        self.lw = lw.reshape(self.length, self.num_attention_heads * self.length, self.length,
                             self.num_attention_heads * self.length).to_sparse()
        self.uw = uw.reshape(self.length, self.num_attention_heads * self.length, self.length,
                             self.num_attention_heads * self.length).to_sparse()
        self.lb = lb.reshape(self.length, self.num_attention_heads * self.length)
        self.ub = ub.reshape(self.length, self.num_attention_heads * self.length)

        time_end = time.time()
        print("time for computing softmax bounds: {:.3f}".format(time_end - time_start))

    def backward(self, lw, uw,flag=False):
        time_start = time.time()

        mask = torch.gt(lw, 0.).to(torch.float)

        _lb = torch.sum((mask * lw) * self.lb.unsqueeze(0).unsqueeze(0) + \
                        ((1 - mask) * lw) * self.ub.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
        # self.lw [20,80,20,80] ->[1,1,20,80,20,80]   lw:[20,80,20,80] ->[20,80,20,80,1,1]

        lw = lw.unsqueeze(-1).unsqueeze(-1)
        mask = torch.gt(lw, 0.).to(torch.float)

        _lw = torch.sum((mask * lw) * self.lw + ((1 - mask) * lw) * self.uw, dim=[-3, -4]).to_dense()
        print(_lw.shape)

        mask = torch.gt(uw, 0.).to(torch.float)

        _ub = torch.sum((mask * uw) * self.ub.unsqueeze(0).unsqueeze(0) + \
                        ((1 - mask) * uw) * self.lb.unsqueeze(0).unsqueeze(0), dim=[-1, -2])

        uw = uw.unsqueeze(-1).unsqueeze(-1)
        mask = torch.gt(uw, 0.).to(torch.float)

        _uw = torch.sum((mask * uw) * self.uw + ((1 - mask) * uw) * self.lw, dim=[-3, -4]).to_dense()
        time_end = time.time()
        print("time for computing softmax backward procedure: {:.3f}".format(time_end - time_start))
        self.controller.lb += _lb
        self.controller.ub += _ub

        self.par.backward_buffer(_lw, _uw)

        del (mask)
        del (_lw)
        del (_uw)

        if self.empty_cache:
            torch.cuda.empty_cache()

def checkBounds(lb,ub,W_Q,W_K,alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u):
    pos=1
    sampled_num=100
    num_attention_heads=W_Q.shape[0]
    d_model=W_Q.shape[1]
    low = lb[pos]
    up = ub[pos]
    for h in range(num_attention_heads):
        W_q = W_Q[h, :, :]
        W_q = W_q.T
        W_k = W_K[h, :, :]
        W_k = W_k.T
        W = W_q @ W_k.T / math.sqrt(d_model)
        WW = W + W.T
        up_gap=0
        low_gap=0
        for n in range(sampled_num):
            random_x = torch.rand(low.shape)
            random_x = low + (up - low) * random_x
            UB=alpha_u[pos,h,pos]@random_x+beta_u[pos,h,pos]@random_x+gamma_u[pos,h,pos]
            LB=alpha_l[pos,h,pos]@random_x+beta_l[pos,h,pos]@random_x+gamma_l[pos,h,pos]
            Real_out=random_x@W@random_x
            up_gap+=(UB-Real_out)
            low_gap+=(Real_out-LB)
            print(UB-Real_out,end=" ")
        print("\n")

        print("head {}, Avg up_gap={:.3f},Avg low_gap={:.3f}".format(h,up_gap/sampled_num,low_gap/sampled_num))




class EdgeDotProduct4QK(Edge):
    def __init__(self, args, controller,par, Dense_query,Dense_key, num_attention_heads,std,QK_output=None):
        super(EdgeDotProduct4QK, self).__init__(args, controller)

        assert(args.method != "baf")

        self.par = par
        #self.par1=par1
        self.num_attention_heads = num_attention_heads
        Q_out_size=Dense_query.weight.shape[-1]
        K_out_size=Dense_key.weight.shape[-1]
        assert Q_out_size == K_out_size, "the attention_head_size of query and key should be the same"

        self.attention_head_size = Q_out_size // num_attention_heads

        W_Q,W_K=Dense_query.weight,Dense_key.weight
        W_Q=W_Q.reshape(self.num_attention_heads,self.attention_head_size,W_Q.shape[1])
        W_K=W_K.reshape(self.num_attention_heads,self.attention_head_size,W_K.shape[1])

        #=======


        time_start = time.time()
        #self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u =get_bounds_QK_bi(par.l, par.u, W_Q, W_K,QK_output=QK_output)
        self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = parall_get_bounds_QK_bi(par.l,
                                                                                                            par.u, W_Q,
                                                                                                            W_K,
                                                                                                            QK_output=QK_output)
        time_end = time.time()
        print("get_bounds_QK_bi time: {:.3f}".format(time_end - time_start))

        # checkBounds(par.l, par.u, W_Q, W_K, self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u)
        #exit(0)

    # self.beta_u = torch.zeros(a.length, num_attention_heads, b.length, self.attention_head_size)
    # self.gamma_l = torch.zeros(a.length, num_attention_heads, b.length)
    def backward(self, lw, uw,device,flag=False):

        #lw:(20,256,20,80)                  (20,256)|  (20,80)|   |(20,256)
         # [length, 1, h, length, r]
        #[20,4,20,256]
        #print("EdgeDotProduct4QK_backward")

        alpha_l = self.alpha_l.unsqueeze(0).unsqueeze(0)
        #[1,1,20,4,20,256]
        alpha_u = self.alpha_u.unsqueeze(0).unsqueeze(0)
        beta_l = self.beta_l.unsqueeze(0).unsqueeze(0)
        beta_u = self.beta_u.unsqueeze(0).unsqueeze(0)
        gamma_l = self.gamma_l.reshape(1, 1, self.par.length, -1)
        # [1,1,20,80]
        gamma_u = self.gamma_u.reshape(1, 1, self.par.length, -1)




        mask = torch.gt(lw, 0.).to(torch.float)
        _lb = torch.sum(lw * (
            mask * gamma_l + \
            (1 - mask) * gamma_u)
        , dim=[-1, -2])
        del(mask)

        mask = torch.gt(uw, 0.).to(torch.float)
        _ub = torch.sum(uw * (
            mask * gamma_u + \
            (1 - mask) * gamma_l)
        , dim=[-1, -2])
        del(mask)

        if self.empty_cache:
            torch.cuda.empty_cache()

        self.controller.lb += _lb
        self.controller.ub += _ub

        #=========
        # [length, h * length (o), h, length, 1]
        #alpha_L [1,1,20,4,20,256]   [20,80,20,4,20,1]
        #==================================================
        # _uw2 = torch.sum(mask * _uw2 * beta_u + (1 - mask) * _uw2 * beta_l, dim=-4)\
        #     .transpose(2, 3)

        #print("第二种")
        batch_size=80
        split_lw=lw.reshape(lw.shape[0]*lw.shape[1], lw.shape[2], self.num_attention_heads, self.par.length, 1)
        split_lw = torch.split(split_lw, batch_size, dim=0)
        temp=[]
        temp1=[]
        for batch in split_lw:
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * alpha_l.squeeze(0) + (1 - mask) * batch * alpha_u.squeeze(0), dim=[-2,-3]))
            bb=torch.sum(mask * batch * beta_l.squeeze(0) + (1 - mask) * batch * beta_u.squeeze(0), dim=-4).transpose(1,2)
            temp1.append(torch.sum(bb, dim=-2))
        _lw=torch.cat(temp, dim=0).reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)
        _lw2=torch.cat(temp1, dim=0).reshape(lw.shape[0], lw.shape[1], self.par.length, -1)

        _lw=_lw+_lw2
        del (_lw2)

        split_uw=uw.reshape(lw.shape[0]*lw.shape[1], lw.shape[2], self.num_attention_heads, self.par.length, 1)
        split_uw = torch.split(split_uw, batch_size, dim=0)
        temp=[]
        temp1=[]
        for batch in split_uw:
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * alpha_u.squeeze(0) + (1 - mask) * batch * alpha_l.squeeze(0), dim=[-2,-3]))
            bb = torch.sum(mask * batch * beta_u.squeeze(0) + (1 - mask) * batch * beta_l.squeeze(0), dim=-4).transpose(
                1, 2)
            temp1.append(torch.sum(bb, dim=-2))
        _uw=torch.cat(temp, dim=0).reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)
        _uw2=torch.cat(temp1, dim=0).reshape(lw.shape[0], lw.shape[1], self.par.length, -1)
        _uw=_uw+_uw2
        self.par.backward_buffer(_lw, _uw)
        #self.par1.backward_buffer(_lw2, _uw2)
        del(_uw2)


        # ==================================================

        # print("第一种")
        # _lw = lw\
        #     .reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.par.length, 1)
        # mask = torch.gt(_lw, 0.).to(torch.float)
        # _lw1 = torch.sum(mask * _lw * alpha_l + (1 - mask) * _lw * alpha_u, dim=[-2,-3])
        #
        # _lw2 = torch.sum(mask * _lw * beta_l + (1 - mask) * _lw * beta_u, dim=[-3, -4])
        #
        # _lw=_lw1+_lw2
        #
        #     #.reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)
        # # (20,80,20,4,20,1)（1,1,20,4,20,64）
        # # [length, h * length (o), h, length, 1]
        # _uw = uw\
        #     .reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.par.length, 1)
        # mask = torch.gt(_uw, 0.).to(torch.float)
        # _uw1 = torch.sum(mask * _uw * alpha_u + (1 - mask) * _uw * alpha_l, dim=[-2,-3])
        #     #.reshape(uw.shape[0], uw.shape[1], uw.shape[2], -1)
        # _uw2 = torch.sum(mask * _uw * beta_u + (1 - mask) * _uw * beta_l, dim=[-3,-4])
        # _uw=_uw1+_uw2
        #
        # self.par.backward_buffer(_lw, _uw)
        # #self.par1.backward_buffer(_lw2, _uw2)
        # del (mask)
        # del(_lw1)
        # del(_uw1)
        # del(_lw2)
        # del(_uw2)
        # if self.empty_cache:
        #     torch.cuda.empty_cache()

        #===========


        '''
        print("第三种")
        _lw = lw\
            .reshape(lw.shape[0]* lw.shape[1], lw.shape[2], self.num_attention_heads, self.par.length, 1)
        temp=[]
        temp1=[]
        for o in range(lw.shape[0]* lw.shape[1]):
            batch=_lw[o]
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * alpha_l.squeeze(0).squeeze(0) + (1 - mask) * batch * alpha_u.squeeze(0).squeeze(0), dim=[-2,-3]))
            temp1.append(torch.sum(mask * batch * beta_l.squeeze(0).squeeze(0) + (1 - mask) * batch * beta_u.squeeze(0).squeeze(0), dim=[-3, -4]))
        _lw1=torch.stack(temp, dim=0).reshape(lw.shape[0], lw.shape[1], lw.shape[2],-1)
        _lw2=torch.stack(temp1, dim=0).reshape(lw.shape[0], lw.shape[1], lw.shape[2],-1)
        _lw = _lw1 + _lw2

        _uw=uw.reshape(uw.shape[0]*uw.shape[1], uw.shape[2], self.num_attention_heads, self.par.length, 1)
        temp=[]
        temp1=[]
        for o in range(uw.shape[0]* uw.shape[1]):
            batch = _uw[o]
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * alpha_u.squeeze(0).squeeze(0) + (1 - mask) * batch * alpha_l.squeeze(0).squeeze(0), dim=[-2,-3]))
            temp1.append(torch.sum(mask * batch * beta_u.squeeze(0).squeeze(0) + (1 - mask) * batch * beta_l.squeeze(0).squeeze(0), dim=[-3,-4]))
        _uw1=torch.stack(temp, dim=0).reshape(uw.shape[0], uw.shape[1], uw.shape[2],-1)
        _uw2=torch.stack(temp1, dim=0).reshape(uw.shape[0], uw.shape[1], uw.shape[2],-1)
        _uw = _uw1+_uw2
        self.par.backward_buffer(_lw, _uw)
        del (mask)
        del(_lw2)
        del(_uw2)
        #==========
        '''
        # _lw2 = lw\
        #     .reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.par.length, 1)
        # mask = torch.gt(_lw2, 0.).to(torch.float)
        # (20,80,20,4,20,1)（1,1,20,4,20,256）
        #_lw2 = torch.sum(mask * _lw2 * beta_l + (1 - mask) * _lw2 * beta_u, dim=[-3,-4])
        # _lw2 = torch.sum(_lw2, dim=-3, keepdim=True)
        # _lw2=_lw2.transpose(2, 3)
        #     #.transpose(2, 3)
        # #_lw2 = torch.sum(_lw2, dim=-2,keepdim=True)
        # _lw2 = _lw2.reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)


        # _uw2 = uw\
        #     .reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.par.length, 1)
        # mask = torch.gt(_uw2, 0.).to(torch.float)
        # _uw2 = torch.sum(mask * _uw2 * beta_u + (1 - mask) * _uw2 * beta_l, dim=[-3,-4])
        # _uw2=torch.sum(_uw2,dim=-3,keepdim=True)
        # _uw2=_uw2.transpose(2, 3)
        #     #.transpose(2, 3)
        # _uw2 = _uw2.reshape(uw.shape[0], uw.shape[1], uw.shape[2], -1)
        # self.par.backward_buffer(_lw2, _uw2)
        # del (mask)
        # del(_lw2)
        # del(_uw2)
        #===========================================================


class EdgeDotProduct(Edge):
    def __init__(self, args, controller, a, b, num_attention_heads,QK_output=None,V_flag=False,version=None):
        super(EdgeDotProduct, self).__init__(args, controller)
        assert(args.method != "baf")

        self.a = a
        self.b = b
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = a.l.shape[-1] // num_attention_heads
        self.opt_able=False
        self.X0_1_opt=None
        self.X0_2_opt=None
        self.X1_1_opt=None
        self.X1_2_opt=None
        self.version=version

        assert self.version is not None, "version is None!!"

        if self.version== "origin":
            #print("origin")
            l_a = a.l.reshape(a.length, num_attention_heads, self.attention_head_size) \
                .repeat(1, 1, b.length).reshape(-1)
            u_a = a.u.reshape(a.length, num_attention_heads, self.attention_head_size) \
                .repeat(1, 1, b.length).reshape(-1)

            l_b = b.l.reshape(b.length, num_attention_heads, self.attention_head_size) \
                .transpose(0, 1).repeat(a.length, 1, 1).reshape(-1)
            u_b = b.u.reshape(b.length, num_attention_heads, self.attention_head_size) \
                .transpose(0, 1).repeat(a.length, 1, 1).reshape(-1)
            self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = \
                get_bounds_xy(l_a, u_a, l_b, u_b)
            # batch_size, length, h, h_size*length
            self.alpha_l = self.alpha_l \
                .reshape(a.length, num_attention_heads,
                         b.length, self.attention_head_size)
            self.alpha_u = self.alpha_u \
                .reshape(a.length, num_attention_heads,
                         b.length, self.attention_head_size)

            self.beta_l = self.beta_l \
                .reshape(a.length, num_attention_heads,
                         b.length, self.attention_head_size)  # .transpose(0, 2)
            self.beta_u = self.beta_u \
                .reshape(a.length, num_attention_heads,
                         b.length, self.attention_head_size)  # .transpose(0, 2)

            # batch_size, length, h, length*h_size
            self.gamma_l = self.gamma_l \
                .reshape(a.length, num_attention_heads, b.length, self.attention_head_size) \
                .sum(dim=-1)
            self.gamma_u = self.gamma_u \
                .reshape(a.length, num_attention_heads, b.length, self.attention_head_size) \
                .sum(dim=-1)

        else:
            #print("Quadratic")
            if self.version=="inner":
                #print("inner")
                l_a = self.a.l.reshape(self.a.length, self.num_attention_heads, self.attention_head_size)
                u_a = self.a.u.reshape(self.a.length, self.num_attention_heads, self.attention_head_size)

                l_b = self.b.l.reshape(self.b.length, self.num_attention_heads, self.attention_head_size)
                u_b = self.b.u.reshape(self.b.length, self.num_attention_heads, self.attention_head_size)
                X0_1 = (u_a + l_a)/2
                X0_2=(u_b+l_b)/2
                self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = \
                    get_bounds_xy_bi_adj(l_a, u_a, l_b, u_b, X0_1, X0_2,X0_1, X0_2)

            elif self.version=="bilinear":
                #print("bilinear")
                self.opt_able = True
                self.rebuild(self.args.device)
            elif self.version=="originPlus":
                #print("originPlus")
                self.opt_able = True
                self.rebuild_ori(self.args.device)


    def rebuild(self,device):
        l_a = self.a.l.reshape(self.a.length, self.num_attention_heads, self.attention_head_size).to(device)
        u_a = self.a.u.reshape(self.a.length, self.num_attention_heads, self.attention_head_size).to(device)

        l_b = self.b.l.reshape(self.b.length, self.num_attention_heads, self.attention_head_size).to(device)
        u_b = self.b.u.reshape(self.b.length, self.num_attention_heads, self.attention_head_size).to(device)

        if self.X0_1_opt is None and self.X0_2_opt is None:
            # m1=((l_a+u_a)/2)
            # m2=((l_b+u_b)/2)
            self.X0_1_opt = torch.zeros_like(l_a).to(device)
            self.X0_2_opt = torch.zeros_like(l_b).to(device)
            self.X1_1_opt = torch.zeros_like(l_a).to(device)
            self.X1_2_opt = torch.zeros_like(l_b).to(device)

        if self.X0_1_opt.device !=torch.device(device) or self.X0_2_opt.device !=torch.device(device):
            self.X0_1_opt=self.X0_1_opt.to(device)
            self.X0_2_opt=self.X0_2_opt.to(device)
        if self.X1_1_opt.device != torch.device(device) or self.X1_2_opt.device != torch.device(device):
            self.X1_1_opt=self.X1_1_opt.to(device)
            self.X1_2_opt=self.X1_2_opt.to(device)

        X0_1=l_a+(u_a-l_a)*torch.sigmoid(self.X0_1_opt)
        X0_2=l_b+(u_b-l_b)*torch.sigmoid(self.X0_2_opt)
        X1_1=l_a+(u_a-l_a)*torch.sigmoid(self.X1_1_opt)
        X1_2=l_b+(u_b-l_b)*torch.sigmoid(self.X1_2_opt)
        # self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = \
        #     get_bounds_xy_bi_adj(l_a, u_a, l_b, u_b,X0_1,X0_2)
        self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = \
            get_bounds_xy_bi_adj(l_a, u_a, l_b, u_b,X0_1,X0_2,X1_1,X1_2)

    def rebuild_ori(self,device):
        l_a = self.a.l.reshape(self.a.length, self.num_attention_heads, self.attention_head_size).to(device)
        u_a = self.a.u.reshape(self.a.length, self.num_attention_heads, self.attention_head_size).to(device)

        l_b = self.b.l.reshape(self.b.length, self.num_attention_heads, self.attention_head_size).to(device)
        u_b = self.b.u.reshape(self.b.length, self.num_attention_heads, self.attention_head_size).to(device)

        if self.X0_1_opt is None and self.X0_2_opt is None:
            self.X0_1_opt = torch.zeros_like(l_a).to(device)
        if self.X1_1_opt is None and self.X1_2_opt is None:
            self.X1_1_opt = torch.zeros_like(l_a).to(device)

        if self.X0_1_opt.device != torch.device(device):
            self.X0_1_opt = self.X0_1_opt.to(device)
        if self.X1_1_opt.device != torch.device(device):
            self.X1_1_opt = self.X1_1_opt.to(device)

        X0_1 = -1 + 2 * torch.sigmoid(self.X0_1_opt)
        X1_1 = -1 + 2 * torch.sigmoid(self.X1_1_opt)

        self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = \
            get_bounds_xy_adj(l_a, u_a, l_b, u_b,X0_1,X1_1)

    def backward(self, lw, uw,flag=False):
        # [length, 1, h, length, r]
        #print("EdgeDotProduct_backward")
        device = lw.device

        if self.controller.opt and self.opt_able:
            if self.version=="originPlus":
                self.rebuild_ori(device)
            else:
                self.rebuild(device)

        alpha_l = self.alpha_l.unsqueeze(0).unsqueeze(0).to(device)
        alpha_u = self.alpha_u.unsqueeze(0).unsqueeze(0).to(device)
        beta_l = self.beta_l.unsqueeze(0).unsqueeze(0).to(device)
        beta_u = self.beta_u.unsqueeze(0).unsqueeze(0).to(device)
        gamma_l = self.gamma_l.reshape(1, 1, self.a.length, -1).to(device)
        gamma_u = self.gamma_u.reshape(1, 1, self.a.length, -1).to(device)

        mask = torch.gt(lw, 0.).to(torch.float)
        _lb = torch.sum(lw * (
            mask * gamma_l + \
            (1 - mask) * gamma_u)
        , dim=[-1, -2])
        del(mask)

        mask = torch.gt(uw, 0.).to(torch.float)
        _ub = torch.sum(uw * (
            mask * gamma_u + \
            (1 - mask) * gamma_l)
        , dim=[-1, -2])      
        del(mask)
        del(gamma_l)
        del(gamma_u)

        if self.empty_cache:
            torch.cuda.empty_cache()              

        self.controller.lb =self.controller.lb+ _lb
        self.controller.ub =self.controller.ub+ _ub


        # [length, h * length (o), h, length, 1]
        #========================================================
        # _lw = lw\
        #     .reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        # mask = torch.gt(_lw, 0.).to(torch.float)
        # # (20,80,20,4,20,1)（1,1,20,4,20,64）
        # _lw = torch.sum(mask * _lw * alpha_l + (1 - mask) * _lw * alpha_u, dim=-2)\
        #     .reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)
        #
        # # [length, h * length (o), h, length, 1]
        # _uw = uw\
        #     .reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        # mask = torch.gt(_uw, 0.).to(torch.float)
        # _uw = torch.sum(mask * _uw * alpha_u + (1 - mask) * _uw * alpha_l, dim=-2)\
        #     .reshape(uw.shape[0], uw.shape[1], uw.shape[2], -1)
        # ========================================================
        batch_size=80
        split_lw=lw.reshape(lw.shape[0]*lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        split_lw = torch.split(split_lw, batch_size, dim=0)
        temp=[]
        for batch in split_lw:
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * alpha_l.squeeze(0) + (1 - mask) * batch * alpha_u.squeeze(0), dim=-2))
        _lw=torch.cat(temp, dim=0).reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)
        #
        split_uw=uw.reshape(uw.shape[0]*uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        split_uw = torch.split(split_uw, batch_size, dim=0)
        temp=[]
        for batch in split_uw:
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * alpha_u.squeeze(0) + (1 - mask) * batch * alpha_l.squeeze(0), dim=-2))
        _uw=torch.cat(temp, dim=0).reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)

        del(mask)

        self.a.backward_buffer(_lw, _uw)

        del(_lw)
        del(_uw)
        if self.empty_cache:
            torch.cuda.empty_cache()


        batch_size=80
        split_lw=lw.reshape(lw.shape[0]*lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        split_lw = torch.split(split_lw, batch_size, dim=0)
        temp=[]
        for batch in split_lw:
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * beta_l.squeeze(0) + (1 - mask) * batch * beta_u.squeeze(0), dim=-4).transpose(
                1, 2))
        _lw2 = torch.cat(temp, dim=0).reshape(lw.shape[0], lw.shape[1], self.b.length, -1)

        split_uw=uw.reshape(uw.shape[0]*uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        split_uw = torch.split(split_uw, batch_size, dim=0)
        temp=[]
        for batch in split_uw:
            mask = torch.gt(batch, 0.).to(torch.float)
            temp.append(torch.sum(mask * batch * beta_u.squeeze(0) + (1 - mask) * batch * beta_l.squeeze(0), dim=-4).transpose(
                1, 2))
        _uw2 = torch.cat(temp, dim=0).reshape(uw.shape[0], uw.shape[1], self.b.length, -1)

        #============================================================
        # _lw2 = lw\
        #     .reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        # mask = torch.gt(_lw2, 0.).to(torch.float)
        # # (20,80,20,4,20,1)（1,1,20,4,20,64）
        # _lw2 = torch.sum(mask * _lw2 * beta_l + (1 - mask) * _lw2 * beta_u, dim=-4)\
        #     .transpose(2, 3)
        # _lw2 = _lw2.reshape(_lw2.shape[0], _lw2.shape[1], _lw2.shape[2], -1)
        #
        # #
        # _uw2 = uw\
        #     .reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        # mask = torch.gt(_uw2, 0.).to(torch.float)
        # _uw2 = torch.sum(mask * _uw2 * beta_u + (1 - mask) * _uw2 * beta_l, dim=-4)\
        #     .transpose(2, 3)
        # _uw2 = _uw2.reshape(_uw2.shape[0], _uw2.shape[1], _uw2.shape[2], -1)
        # ============================================================
        self.b.backward_buffer(_lw2, _uw2)
        del(mask)
        del(_lw2)
        del(_uw2)
        if self.empty_cache:
            torch.cuda.empty_cache()

    def backward_get_vars(self, device=None):
        if device is None:
            device = self.args.device
        if self.opt_able:
            print(self.version)
            if self.version=="originPlus":
                self.X0_1_opt = self.X0_1_opt.clone().detach().to(device)
                self.X0_1_opt.requires_grad_()
                # self.X1_1_opt = self.X0_1_opt.clone().detach().to(device)
                # self.X1_1_opt.requires_grad_()
                self.controller.opt_vars.append(self.X0_1_opt)
                # self.controller.opt_vars.append(self.X1_1_opt)
            else:
                self.X0_1_opt=self.X0_1_opt.clone().detach().to(device)
                self.X0_2_opt =self.X0_2_opt.clone().detach().to(device)
                self.X0_1_opt.requires_grad_()
                self.X0_2_opt.requires_grad_()
                # self.X1_1_opt=self.X1_1_opt.clone().detach().to(device)
                # self.X1_2_opt =self.X1_2_opt.clone().detach().to(device)
                # self.X1_1_opt.requires_grad_()
                # self.X1_2_opt.requires_grad_()
                self.controller.opt_vars.append(self.X0_1_opt)
                self.controller.opt_vars.append(self.X0_2_opt)
                # self.controller.opt_vars.append(self.X1_1_opt)
                # self.controller.opt_vars.append(self.X1_2_opt)
        self.a.need_pass_buffer()
        self.b.need_pass_buffer()
class EdgeTranspose(Edge):
    def __init__(self, args, controller, par, num_attention_heads):
        super(EdgeTranspose, self).__init__(args, controller)

        assert(args.method != "baf")

        self.par = par
        self.num_attention_heads = num_attention_heads

    def transpose(self, w):
        w = w.reshape(
            w.shape[0], w.shape[1], w.shape[2], 
            self.num_attention_heads, -1
        ).transpose(2, 4)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], -1)    
        return w

    def backward(self, lw, uw,flag=False):
        #print("EdgeTranspose_backward")
        lw = self.transpose(lw)
        uw = self.transpose(uw)

        self.par.backward_buffer(lw, uw)
    def backward_get_vars(self, device=None):
        self.par.need_pass_buffer()

class EdgeMultiply(EdgeActivation):
    def __init__(self, args, controller, a, b):
        super(EdgeMultiply, self).__init__(args, controller, par=a, par2=b)

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = get_bounds_xy(
            a.l.reshape(-1),
            a.u.reshape(-1),
            b.l.reshape(-1),
            b.u.reshape(-1)
        )
        alpha_l = alpha_l.reshape(a.l.shape)
        beta_l = beta_l.reshape(a.l.shape)
        gamma_l = gamma_l.reshape(a.l.shape)
        alpha_u = alpha_u.reshape(a.l.shape)
        beta_u = beta_u.reshape(a.l.shape)
        gamma_u = gamma_u.reshape(a.l.shape)

        self.add_linear(mask=None, type="lower", k=alpha_l, x0=0, y0=gamma_l)
        self.add_linear(mask=None, type="lower", k=beta_l, x0=0, y0=0, second=True)
        self.add_linear(mask=None, type="upper", k=alpha_u, x0=0, y0=gamma_u)
        self.add_linear(mask=None, type="upper", k=beta_u, x0=0, y0=0, second=True)

class EdgeSqr(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeSqr, self).__init__(args, controller, par)

        k = self.par.u + self.par.l
        self.add_linear(mask=None, type="upper", k=k, x0=self.par.l, y0=self.par.l.pow(2))
        m = torch.max((self.par.l + self.par.u) / 2, 2 * self.par.u)
        self.add_linear(mask=self.mask_neg, type="lower", k=2*m, x0=m, y0=m.pow(2))
        m = torch.min((self.par.l + self.par.u) / 2, 2 * self.par.l)
        self.add_linear(mask=self.mask_pos, type="lower", k=2*m, x0=m, y0=m.pow(2))

class EdgeSqrt(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeSqrt, self).__init__(args, controller, par)

        assert(torch.min(self.par.l) >= 0)
        k = (torch.sqrt(self.par.u) - torch.sqrt(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=None, type="lower", k=k, x0=self.par.l, y0=torch.sqrt(self.par.l) + epsilon)
        m = (self.par.l + self.par.u) / 2
        k = 0.5 / torch.sqrt(m)
        self.add_linear(mask=None, type="upper", k=k, x0=m, y0=torch.sqrt(m) + epsilon)

class EdgeReciprocal(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeReciprocal, self).__init__(args, controller, par)
        assert(torch.min(self.par.l))
        m = (self.par.l + self.par.u) / 2
        kl = -1 / m.pow(2)
        self.add_linear(mask=None, type="lower", k=kl, x0=m, y0=1. / m)
        ku = -1. / (self.par.l * self.par.u)
        self.add_linear(mask=None, type="upper", k=ku, x0=self.par.l, y0=1. / self.par.l)

class EdgeLinear(EdgeActivation):
    def __init__(self, args, controller, par, w, b): 
        super(EdgeLinear, self).__init__(args, controller, par)

        self.add_linear(mask=None, type="lower", k=w, x0=0., y0=b)
        self.add_linear(mask=None, type="upper", k=w, x0=0., y0=b)

class EdgeExp(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeExp, self).__init__(args, controller, par)

        m = torch.min((self.par.l + self.par.u) / 2, self.par.l + 0.99)
        k = torch.exp(m)
        self.add_linear(mask=None, type="lower", k=k, x0=m, y0=torch.exp(m))
        k = (torch.exp(self.par.u) - torch.exp(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=None, type="upper", k=k, x0=self.par.l, y0=torch.exp(self.par.l))

class EdgeDivide(EdgeComplex):
    def __init__(self, args, controller, a, b):
        super(EdgeDivide, self).__init__(args, controller)
        b_reciprocal = b.next(EdgeReciprocal(args, controller, b))
        self.res = a.next(EdgeMultiply(args, controller, a, b_reciprocal))

class EdgeRelu(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeRelu, self).__init__(args, controller, par)

        self.add_linear(mask=self.mask_neg, type="lower", k=0., x0=0, y0=0)
        self.add_linear(mask=self.mask_neg, type="upper", k=0., x0=0, y0=0)        
        self.add_linear(mask=self.mask_pos, type="lower", k=1., x0=0, y0=0)
        self.add_linear(mask=self.mask_pos, type="upper", k=1., x0=0, y0=0)

        k = self.par.u / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=self.par.l, y0=0)

        k = torch.gt(torch.abs(self.par.u), torch.abs(self.par.l)).to(torch.float)
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=0, y0=0)

class EdgeTanh(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeTanh, self).__init__(args, controller, par)

        def dtanh(x):
            return 1. / torch.cosh(x).pow(2)
            
        # lower bound for negative
        m = (self.par.l + self.par.u) / 2
        k = dtanh(m)
        self.add_linear(mask=self.mask_neg, type="lower", k=k, x0=m, y0=torch.tanh(m))
        # upper bound for positive
        self.add_linear(mask=self.mask_pos, type="upper", k=k, x0=m, y0=torch.tanh(m))

        # upper bound for negative
        k = (torch.tanh(self.par.u) - torch.tanh(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_neg, type="upper", k=k, x0=self.par.l, y0=torch.tanh(self.par.l))
        # lower bound for positive
        self.add_linear(mask=self.mask_pos, type="lower", k=k, x0=self.par.l, y0=torch.tanh(self.par.l))

        # bounds for both
        max_iter = 10

        # lower bound for both
        diff = lambda d: (torch.tanh(self.par.u) - torch.tanh(d)) / (self.par.u - d + epsilon) - dtanh(d)
        d = self.par.l / 2
        _l = self.par.l
        if torch.cuda.is_available():
            _u = torch.zeros(self.par.l.shape).cuda()
        else:
            _u = torch.zeros(self.par.l.shape)
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * mask_p + _l * (1 - mask_p)
            _u = d * (1 - mask_p) + _u * mask_p
            d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
        k = (torch.tanh(d) - torch.tanh(self.par.u)) / (d - self.par.u + epsilon)
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=d, y0=torch.tanh(d))

        # upper bound for both
        diff = lambda d: (torch.tanh(d) - torch.tanh(self.par.l))/ (d - self.par.l + epsilon) - dtanh(d)
        d = self.par.u / 2
        if torch.cuda.is_available():
            _l = torch.zeros(self.par.l.shape).cuda()
        else:
            _l = torch.zeros(self.par.l.shape)
        _u = self.par.u
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * (1 - mask_p) + _l * mask_p
            _u = d * mask_p + _u * (1 - mask_p)
            d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
        k = (torch.tanh(d) - torch.tanh(self.par.l)) / (d - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=d, y0=torch.tanh(d))        
