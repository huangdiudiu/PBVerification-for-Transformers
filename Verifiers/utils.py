# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.
from Verifiers.config import use_mean_eps
import torch
import math
from Verifiers.Bounds import Bounds,cvx
import numpy as np
def check(name, bounds=None, l=None, u=None, std=None, verbose=False):
    if verbose:
        print("Check ", name)
    eps = 1e-4
    if bounds is not None:
        l, u = bounds.concretize()
    if len(l.shape) == 3:
        l, u, std = l[0], u[0], std[0]
    if len(l.shape) == 2 and len(std.shape) == 3:
        std=std.squeeze(0)
    c = torch.gt(l - eps, std).to(torch.float) + torch.lt(u + eps, std).to(torch.float)
    if bounds is not None:
        c += torch.gt(bounds.lb[0] - eps, std).to(torch.float) + torch.lt(bounds.ub[0] + eps, std).to(torch.float)
    errors = torch.sum(c)
    score = float(torch.mean(u - l))
    if verbose:
        print("%d errors, %.5f average range" % (errors, score))
        if errors > 0:
            cnt = 0
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    if c[i,j] > 0:
                        print(i, j)
                        print(l[i,j], u[i,j], std[i,j])
                        cnt += 1
                        if cnt >= 10: 
                            assert(0)
    assert(errors == 0)



def adv_checker(QK_output,l,u):
    adv_in=torch.tensor(QK_output["in"]).to(l.device)
    eps = 1e-4
    c = torch.gt(l - eps, adv_in).to(torch.float) + torch.lt(u + eps, adv_in).to(torch.float)

    errors = torch.sum(c)
    score = float(torch.mean(u - l))
    if errors > 0:
        print("adv out of the range %d errors, %.5f average range" % (errors, score))
        assert(0)
    print("check adv..... pass")
    assert(errors == 0)

def parall_get_bounds_QK_bi(l_x, u_x,W_Q,W_K,QK_output=None):
    if QK_output is None:
        print("QK_output is None")
    else:
        print("QK_output is not  None")
    num_pos= l_x.shape[0]
    d_in=W_Q.shape[2]
    num_attention_heads=W_Q.shape[0]
    d_model=W_Q.shape[1]

    num_attention_heads1=W_K.shape[0]
    d_model1=W_K.shape[1]

    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"
    l_a=l_x.reshape(num_pos,1,d_in,1).repeat(1,num_pos,1,d_in)
    u_a=u_x.reshape(num_pos,1,d_in,1).repeat(1,num_pos,1,d_in)
    l_b=l_x.reshape(1,num_pos,1,d_in).repeat(num_pos,1,d_in,1)
    u_b=u_x.reshape(1,num_pos,1,d_in).repeat(num_pos,1,d_in,1)
    alpha_l1, beta_l1, gamma_l1, alpha_u1, beta_u1, gamma_u1=get_bounds_xy(l_a, u_a, l_b, u_b)

    alpha_l1,beta_l1,alpha_u1,beta_u1=alpha_l1.unsqueeze(1),beta_l1.unsqueeze(1),alpha_u1.unsqueeze(1),beta_u1.unsqueeze(1)
    gamma_l1,gamma_u1=gamma_l1.unsqueeze(1),gamma_u1.unsqueeze(1)

    A=torch.bmm(W_Q.transpose(1, 2),W_K)/math.sqrt(d_model)
    A=A.unsqueeze(0).unsqueeze(2)
    l_a,u_a,l_b,u_b=l_a.unsqueeze(1),u_a.unsqueeze(1),l_b.unsqueeze(1),u_b.unsqueeze(1)
    #========
    mask = torch.isclose(l_a, u_a,atol=1e-6).to(torch.float)
    alpha_l=alpha_l1*A
    gamma_l_add=mask*l_a*alpha_l
    alpha_l=alpha_l*(1-mask)
    alpha_l=torch.sum(alpha_l,dim=-1)

    alpha_u=alpha_u1*A
    gamma_u_add=mask*u_a*alpha_u
    alpha_u=alpha_u*(1-mask)
    alpha_u=torch.sum(alpha_u,dim=-1)

    mask = torch.isclose(l_b, u_b,atol=1e-6).to(torch.float)
    beta_l=beta_l1*A
    gamma_l_add=gamma_l_add+mask*l_b*beta_l
    beta_l=beta_l*(1-mask)
    beta_l=torch.sum(beta_l,dim=-2)

    beta_u=beta_u1*A
    gamma_u_add=gamma_u_add+mask*u_b*beta_u
    beta_u = beta_u * (1 - mask)
    beta_u=torch.sum(beta_u,dim=-2)

    gamma_l=gamma_l1*A
    gamma_l=gamma_l+gamma_l_add
    gamma_l=torch.sum(gamma_l,dim=[-1,-2])

    gamma_u=gamma_u1*A
    gamma_u=gamma_u+gamma_u_add
    gamma_u=torch.sum(gamma_u,dim=[-1,-2])

    input_m, input_eps = (u_x + l_x) / 2, (u_x - l_x) / 2
    input_eps_numpy=input_eps.cpu().numpy()
    SDP = SDP4QK(d_in, cvx.MOSEK)

    use_mean_eps=True

    if use_mean_eps:
        eps_norms = input_eps.norm(dim=1)
        k = int(0.1 * eps_norms.size(0))
        top_k_values, top_k_indices = torch.topk(eps_norms, k)
        top_k_indices=top_k_indices.detach().cpu().tolist()
    else:
        top_k_indices=list(range(u_x.size(0)))

    mask = torch.ones(u_x.size(0), dtype=torch.bool)
    mask[top_k_indices] = False
    others=input_eps[mask]
    assert others.size(0) !=0, "others.size(0) is 0!!"
    eps_mean = others.mean(dim=0)

    # generate all the SDP problems
    params_list=[]
    use_mean_list=[[]for i in range(num_attention_heads)]
    for h in range(num_attention_heads):
        W_q = W_Q[h, :, :].clone().cpu().numpy()
        W_q = W_q.T
        W_k = W_K[h, :, :].clone().cpu().numpy()
        W_k = W_k.T
        W = W_q @ W_k.T / np.sqrt(d_model)

        if use_mean_eps:
            params={}
            params['W'] = W
            params['eps4_query'] = eps_mean.cpu().numpy()
            params['eps4_key'] = eps_mean.cpu().numpy()
            params['ij'] = (-1, -1)
            params['h'] = h
            params_list.append(params)
            params={}
            params['W'] = W
            params['eps4_query'] = eps_mean.cpu().numpy()
            params['eps4_key'] = eps_mean.cpu().numpy()
            params['ij'] = (-1, -2)
            params['h'] = h
            params_list.append(params)

        for pos1 in range(num_pos):
            for pos2 in range(num_pos):
                if (not torch.allclose(l_x[pos1], u_x[pos1], atol=1e-6)) and (not torch.allclose(l_x[pos2], u_x[pos2], atol=1e-6)):
                    if (pos1 in top_k_indices or pos2 in top_k_indices):
                        params={}
                        params['W'] = W
                        params['eps4_query']=input_eps[pos1].cpu().numpy()
                        params['eps4_key'] = input_eps[pos2].cpu().numpy()
                        params['ij']=(pos1,pos2)
                        params['h']=h
                        params_list.append(params)
                    elif use_mean_eps and (pos1 not in top_k_indices and pos2 not in top_k_indices):
                        use_mean_list[h].append([pos1,pos2])

    num_workers=min(cpu_count()-1,7)
    print("num_workers:{}, SDPs:{}".format(num_workers,len(params_list)))
    start=time.time()
    with Pool(num_workers) as pool:
        adp_results = pool.map(SDP.sdp_sover, params_list)
    end=time.time()
    print("solving all sdps in {} seconds".format(end-start))

    dig_mask=torch.zeros(num_pos, num_attention_heads, num_pos,dtype=torch.bool).to(l_x.device)
    offdig_mask=torch.zeros(num_pos, num_attention_heads, num_pos,dtype=torch.bool).to(l_x.device)
    D_query=torch.zeros(2,num_pos, num_attention_heads, num_pos, d_in).to(l_x.device)
    D_key=torch.zeros(2,num_pos, num_attention_heads, num_pos, d_in).to(l_x.device)
    obj=torch.zeros(2,num_pos, num_attention_heads, num_pos).to(l_x.device)

    mean_dig_re=[None] * num_attention_heads
    mean_off_dig_re=[None] * num_attention_heads
    for re in adp_results:
        i=re["i"]
        j=re["j"]
        h=re["h"]
        if i<0 and j<0:
            if i==j:
                #mean_dig_re[h].append(re["UNL_list"])
                mean_dig_re[h]=re["UNL_list"]
            else:
                #mean_off_dig_re[h].append(re["UNL_list"])
                mean_off_dig_re[h]=re["UNL_list"]
        else:
            if i==j:
                dig_mask[i, h, j] = True
            else:
                offdig_mask[i, h, j] = True
            UNL_list=re["UNL_list"]
            for bnd in [0,1]:
                Q_bound=UNL_list[bnd]
                D_query[bnd,i,h,j,:]=torch.tensor(Q_bound["D_query"], dtype=torch.float32)
                D_key[bnd,i,h,j,:]=torch.tensor(Q_bound["D_key"], dtype=torch.float32)
                obj[bnd, i, h, j]=torch.tensor(Q_bound["Obj_val"], dtype=torch.float32)

    if use_mean_eps:
        for h in range(num_attention_heads):
            for pos_tuple in use_mean_list[h]:
                i,j=pos_tuple[0],pos_tuple[1]
                if not dig_mask[i, h, j] and not offdig_mask[i, h, j]:
                    if i==j:
                        dig_mask[i, h, j] = True
                        UNL_list=mean_dig_re[h]
                    else:
                        offdig_mask[i, h, j] = True
                        UNL_list=mean_off_dig_re[h]
                    for bnd in [0, 1]:
                        Q_bound = UNL_list[bnd]
                        #print(bnd,i,h,j)
                        D_query[bnd, i, h, j,:] = torch.tensor(Q_bound["D_query"], dtype=torch.float32)
                        D_key[bnd, i, h, j, :] = torch.tensor(Q_bound["D_key"], dtype=torch.float32)
                        if i == j:
                            obj[bnd, i, h, j] = torch.tensor(Q_bound["D_query"] @ input_eps_numpy[i] ** 2,
                                                             dtype=torch.float32)
                        else:
                            obj[bnd, i, h, j] = torch.tensor(
                                Q_bound["D_query"] @ input_eps_numpy[i] ** 2 + Q_bound["D_key"] @ input_eps_numpy[j] ** 2,
                                dtype=torch.float32)

    W=W_Q.transpose(1,2)@W_K/math.sqrt(d_model)
    WW=W+W.transpose(1,2)
    W=W.unsqueeze(1).unsqueeze(0)#[1,num_attention_heads,1,d_in,d_in]
    WW=WW.unsqueeze(1).unsqueeze(0)
    query_m=input_m.unsqueeze(1).repeat(1,num_attention_heads,1)
    key_m=input_m.unsqueeze(1).repeat(1,num_attention_heads,1)
    query_m=query_m.repeat(1, 1, num_pos).reshape(num_pos, num_attention_heads, num_pos, d_in)
    key_m=key_m.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos, d_in)

    # for diagonal case
    XWX = -query_m.unsqueeze(-2) @ WW @ query_m.unsqueeze(-1)
    XWX=XWX.squeeze(-1).squeeze(-1)
    alpha_l = torch.where(dig_mask.unsqueeze(-1), (WW @ query_m.unsqueeze(-1)).squeeze(-1), alpha_l)
    temp = 0.5 * (XWX - obj[0])
    gamma_l = torch.where(dig_mask,temp,gamma_l)
    beta_l = torch.where(dig_mask.unsqueeze(-1),beta_l*0,beta_l)

    alpha_u=torch.where(dig_mask.unsqueeze(-1),alpha_l,alpha_u)
    temp = 0.5 * (XWX + obj[1])
    gamma_u = torch.where(dig_mask, temp, gamma_u)
    beta_u= torch.where(dig_mask.unsqueeze(-1),beta_u*0,beta_u)
    # for off-diagonal case
    XWX = -query_m.unsqueeze(-2) @ W @ key_m.unsqueeze(-1)
    XWX = XWX.squeeze(-1).squeeze(-1)
    alpha_l = torch.where(offdig_mask.unsqueeze(-1),(W @ key_m.unsqueeze(-1)).squeeze(-1),alpha_l)
    beta_l = torch.where(offdig_mask.unsqueeze(-1),(W.transpose(-2,-1) @ query_m.unsqueeze(-1)).squeeze(-1),beta_l)
    temp = XWX - 0.5 * obj[0]
    gamma_l = torch.where(offdig_mask, temp, gamma_l)
    alpha_u = torch.where(offdig_mask.unsqueeze(-1), alpha_l, alpha_u)
    beta_u=torch.where(offdig_mask.unsqueeze(-1), beta_l, beta_u)
    temp=XWX + 0.5 * obj[1]
    gamma_u = torch.where(offdig_mask, temp, gamma_u)

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u


def get_bounds_QK_bi(l_x, u_x,W_Q,W_K,QK_output=None):
    QK_output=None
    if QK_output is None:
        print("QK_output is None")
    else:
        print("QK_output is not  None")
    #adv_checker(QK_output,l_x,u_x)
    num_pos= l_x.shape[0]
    d_in=W_Q.shape[2]
    num_attention_heads=W_Q.shape[0]
    d_model=W_Q.shape[1]

    num_attention_heads1=W_K.shape[0]
    d_model1=W_K.shape[1]

    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"



    # num_pos1=num_pos
    # l_a = l_x.reshape(num_pos, d_in) \
    #     .repeat(1, num_pos1).reshape(num_pos,num_pos1,d_in,1)
    # l_a=l_a.repeat(1, 1,1,d_in)
    # u_a = u_x.reshape(num_pos, d_in) \
    #     .repeat(1, num_pos1).reshape(num_pos,num_pos1,d_in,1)
    # u_a = u_a.repeat(1, 1, 1, d_in)
    #
    # l_b = l_x.reshape(num_pos1,1,d_in) \
    #     .transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos,num_pos1,1,d_in)
    # l_b=l_b.repeat(1,1,d_in,1)
    # u_b = u_x.reshape(num_pos1,1,d_in) \
    #     .transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos,num_pos1,1,d_in)
    # u_b=u_b.repeat(1,1,d_in,1)

    # l_x = torch.tensor([[2,3],[-1,-1]])
    # u_x = torch.tensor([[2, 3], [2, 1]])
    # num_pos=2
    # d_in=2


    l_a=l_x.reshape(num_pos,1,d_in,1).repeat(1,num_pos,1,d_in)
    u_a=u_x.reshape(num_pos,1,d_in,1).repeat(1,num_pos,1,d_in)
    l_b=l_x.reshape(1,num_pos,1,d_in).repeat(num_pos,1,d_in,1)
    u_b=u_x.reshape(1,num_pos,1,d_in).repeat(num_pos,1,d_in,1)


    # l_a=l_x.repeat(1, num_pos).unsqueeze(-1).reshape(num_pos,num_pos,d_in,1).repeat(1, 1,1,d_in).reshape(-1)
    # u_a=u_x.repeat(1, num_pos).unsqueeze(-1).reshape(num_pos,num_pos,d_in,1).repeat(1, 1, 1,d_in).reshape(-1)
    # l_b=l_x.unsqueeze(0).repeat(num_pos, 1, 1).reshape(num_pos,num_pos,1,d_in).repeat(1, 1, d_in,1).reshape(-1)
    # u_b=u_x.unsqueeze(0).repeat(num_pos, 1, 1).reshape(num_pos,num_pos,1,d_in).repeat(1, 1, d_in,1).reshape(-1)

    alpha_l1, beta_l1, gamma_l1, alpha_u1, beta_u1, gamma_u1=get_bounds_xy(l_a, u_a, l_b, u_b)


    # for i in range(num_pos):
    #     for j in range(num_pos):
    #         print("i*j:")
    #         print("alpha_l1:{}",alpha_l1[i,j])
    #         print("beta_l1:{}",beta_l1[i,j])
    #         print("gamma_l1:{}",gamma_l1[i,j])
    #         print("alpha_u1:{}",alpha_u1[i,j])
    #         print("beta_u1:{}",beta_u1[i,j])
    #         print("gamma_u1:{}",gamma_u1[i,j])
    #         print("====================================")

    alpha_l1=alpha_l1.unsqueeze(1)
    beta_l1=beta_l1.unsqueeze(1)
    alpha_u1=alpha_u1.unsqueeze(1)
    beta_u1=beta_u1.unsqueeze(1)
    gamma_l1=gamma_l1.unsqueeze(1)
    gamma_u1=gamma_u1.unsqueeze(1)

    A=torch.bmm(W_Q.transpose(1, 2),W_K)/math.sqrt(d_model)
    # A=torch.tensor([[2,-3],[4,-2]])
    # A=A.unsqueeze(0)
    A=A.unsqueeze(0).unsqueeze(2)

    l_a=l_a.unsqueeze(1)
    u_a=u_a.unsqueeze(1)
    l_b=l_b.unsqueeze(1)
    u_b=u_b.unsqueeze(1)


    #========
    # mask = torch.isclose(l_a, u_a,atol=1e-6).to(torch.float)
    # alpha_l=alpha_l1*A
    # gamma_l_add=mask*l_a*alpha_l
    # alpha_l=alpha_l*(1-mask)
    # alpha_l=torch.sum(alpha_l,dim=-1)
    #
    # alpha_u=alpha_u1*A
    # gamma_u_add=mask*u_a*alpha_u
    # alpha_u=alpha_u*(1-mask)
    # alpha_u=torch.sum(alpha_u,dim=-1)
    #
    # mask = torch.isclose(l_b, u_b,atol=1e-6).to(torch.float)
    # beta_l=beta_l1*A
    # gamma_l_add=gamma_l_add+mask*l_b*beta_l
    # beta_l=beta_l*(1-mask)
    # beta_l=torch.sum(beta_l,dim=-2)
    #
    # beta_u=beta_u1*A
    # gamma_u_add=gamma_u_add+mask*u_b*beta_u
    # beta_u = beta_u * (1 - mask)
    # beta_u=torch.sum(beta_u,dim=-2)
    #
    # gamma_l=gamma_l1*A
    # gamma_l=gamma_l+gamma_l_add
    # gamma_l=torch.sum(gamma_l,dim=[-1,-2])
    #
    # gamma_u=gamma_u1*A
    # gamma_u=gamma_u+gamma_u_add
    # gamma_u=torch.sum(gamma_u,dim=[-1,-2])
    # #=============
    #
    mask = torch.gt(A, 0.).to(torch.float)
    mask1 = torch.isclose(l_a, u_a,atol=1e-6).to(torch.float)
    alpha_l=mask*A*alpha_l1+(1-mask)*A*alpha_u1
    gamma_l_add=mask1*l_a*alpha_l
    alpha_l=alpha_l*(1-mask1)
    alpha_l=torch.sum(alpha_l,dim=-1)

    alpha_u=alpha_u1*mask*A+alpha_l1*(1-mask)*A
    gamma_u_add=mask1*u_a*alpha_u
    alpha_u=alpha_u*(1-mask1)
    alpha_u=torch.sum(alpha_u,dim=-1)

    mask1 = torch.isclose(l_b, u_b,atol=1e-6).to(torch.float)
    beta_l=beta_l1*mask*A+beta_u1*(1-mask)*A
    gamma_l_add=gamma_l_add+mask1*l_b*beta_l
    beta_l=beta_l*(1-mask1)
    beta_l=torch.sum(beta_l,dim=-2)

    beta_u=beta_u1*mask*A+beta_l1*(1-mask)*A
    gamma_u_add=gamma_u_add+mask1*u_b*beta_u
    beta_u = beta_u * (1 - mask1)
    beta_u=torch.sum(beta_u,dim=-2)

    gamma_l=gamma_l1*mask*A+gamma_u1*(1-mask)*A
    gamma_l = gamma_l + gamma_l_add
    gamma_l=torch.sum(gamma_l,dim=[-1,-2])

    gamma_u=gamma_u1*mask*A+gamma_l1*(1-mask)*A
    gamma_u = gamma_u + gamma_u_add
    gamma_u=torch.sum(gamma_u,dim=[-1,-2])

    # for i in range(num_pos):
    #     for j in range(num_pos):
    #         print("i*j:")
    #         print("alpha_l:{}",alpha_l[i,0,j])
    #         print("beta_l:{}",beta_l[i,0,j])
    #         print("gamma_l:{}",gamma_l[i,0,j])
    #         print("alpha_u:{}",alpha_u[i,0,j])
    #         print("beta_u:{}",beta_u[i,0,j])
    #         print("gamma_u:{}",gamma_u[i,0,j])
    #         print("====================================")
    # exit(0)



    # device=l_x.device
    # alpha_l=torch.zeros((num_pos,num_attention_heads,num_pos,d_in),dtype=torch.float32).to(device)
    # alpha_u = torch.zeros((num_pos,num_attention_heads, num_pos, d_in), dtype=torch.float32).to(device)
    # beta_l = torch.zeros((num_pos,num_attention_heads, num_pos, d_in), dtype=torch.float32).to(device)
    # beta_u = torch.zeros((num_pos,num_attention_heads, num_pos, d_in), dtype=torch.float32).to(device)
    # gamma_l=torch.zeros((num_pos,num_attention_heads, num_pos), dtype=torch.float32).to(device)
    # gamma_u=torch.zeros((num_pos,num_attention_heads, num_pos), dtype=torch.float32).to(device)

    input_m, input_eps = (u_x + l_x) / 2, (u_x - l_x) / 2
    SDP = SDP4QK(d_in, cvx.MOSEK)  # cvx.SCS
    if use_mean_eps:
        # diff=torch.abs(input_eps)
        # equal_mask=torch.all(diff >= 1e-6, dim=1)
        #
        # masked_eps=input_eps[equal_mask]
        # if masked_eps.numel() > 0:
        #     eps_mean = masked_eps.mean(dim=0)
        # else:
        #     eps_mean=input_eps[0]
        eps_mean=input_eps.mean(dim=0)

        eps_mean=eps_mean.cpu().numpy()
        input_m_np=input_m.clone().cpu().numpy()

        for h in range(num_attention_heads):
            W_q = W_Q[h, :, :].clone().cpu().numpy()
            W_q = W_q.T
            W_k = W_K[h, :, :].clone().cpu().numpy()
            W_k = W_k.T
            W = W_q @ W_k.T / np.sqrt(d_model)
            WW = W + W.T
            # for diagonal case
            # if np.all(np.abs(eps_mean) < 1e-8):
            #     eps_mean*=0

            UNL_list=SDP.diagSolve(W,eps_mean)
            for pos in range(num_pos):
                if not torch.allclose(l_x[pos], u_x[pos], atol=1e-8):
                    for bnd in [0, 1]:
                        Q_bound=UNL_list[bnd]
                        if bnd == 0:
                            XWX = -input_m_np[pos] @ WW @ input_m_np[pos]
                            if QK_output is None:
                                alpha_l[pos, h, pos, :]=torch.tensor(WW @ input_m_np[pos] ,dtype=torch.float32)
                                temp=0.5*(XWX-Q_bound["Obj_val"])
                                gamma_l[pos, h, pos] = torch.tensor(temp, dtype=torch.float32)
                                beta_l[pos, h, pos, :] = beta_l[pos, h, pos, :] * 0
                            else: # ==============optimize==========
                                #mDm = input_m_np[pos] @ np.diag(Q_bound["D_query"]) @ input_m_np[pos]
                                #QK_output["in"][pos]=input_m_np[pos]######
                                X0=QK_output["in"][pos]
                                A = np.diag(Q_bound["D_query"]) + WW
                                B=-Q_bound["D_query"]*input_m_np[pos]
                                XAX =X0 @ A @ X0
                                alpha_l[pos, h, pos, :]=torch.tensor(X0@A+B,dtype=torch.float32)
                                # alpha_l1=torch.tensor(WW @ input_m_np[pos],dtype=torch.float32).to(alpha_l[pos, h, pos, :].device) ################
                                # if not torch.allclose(alpha_l1, alpha_l[pos, h, pos, :], atol=1e-6):
                                #     print("alpha_l errors")
                                #     exit(0)
                                c=-0.5*(B@input_m_np[pos]+Q_bound["Obj_val"])
                                temp=-0.5*XAX+c
                                gamma_l[pos, h, pos] = torch.tensor(temp, dtype=torch.float32)
                                # temp=0.5*(XWX-Q_bound["Obj_val"]) ######################
                                # gamma_l1= torch.tensor(temp, dtype=torch.float32).to(gamma_l[pos, h, pos].device)##############
                                # if not torch.allclose(gamma_l1, gamma_l[pos, h, pos], atol=1e-6):
                                #     print("gamma_l errors")
                                #     exit(0)
                                beta_l[pos, h, pos, :] = beta_l[pos, h, pos, :] * 0

                        if bnd == 1:
                            XWX = -input_m_np[pos] @ WW @ input_m_np[pos]
                            if QK_output is None:
                                alpha_u[pos, h, pos, :] = torch.tensor(WW @ input_m_np[pos],dtype=torch.float32)
                                temp=0.5*(XWX+Q_bound["Obj_val"])
                                gamma_u[pos, h, pos] = torch.tensor(temp, dtype=torch.float32)
                                beta_u[pos, h, pos, :] = beta_u[pos, h, pos, :] * 0
                            else: # ==============optimize==========
                                #QK_output["in"][pos] = input_m_np[pos]  ######
                                X0 = QK_output["in"][pos]
                                A = np.diag(Q_bound["D_query"]) - WW
                                B = -Q_bound["D_query"] * input_m_np[pos]
                                XAX = X0 @ A @ X0
                                alpha_u[pos, h, pos, :] = -torch.tensor(X0@A+B,dtype=torch.float32)
                                #alpha_u1=torch.tensor(WW @ input_m_np[pos],dtype=torch.float32).to(alpha_u[pos, h, pos, :].device) ################
                                # if not torch.allclose(alpha_u1, alpha_u[pos, h, pos, :], atol=1e-6):
                                #     print("alpha_u errors")
                                #     exit(0)
                                c=-0.5*(B@input_m_np[pos]+Q_bound["Obj_val"])
                                temp = 0.5*XAX-c
                                gamma_u[pos, h, pos] = torch.tensor(temp, dtype=torch.float32)
                                # temp=0.5*(XWX+Q_bound["Obj_val"]) ######################
                                # gamma_u1= torch.tensor(temp, dtype=torch.float32).to(gamma_u[pos, h, pos].device)##############
                                # if not torch.allclose(gamma_u1, gamma_u[pos, h, pos], atol=1e-6):
                                #     print("gamma_u errors")
                                #     exit(0)
                                beta_u[pos, h, pos, :] = beta_u[pos, h, pos, :] * 0

            # for off-diagonal case
            UNL_list = SDP.off_diagSolve(W,eps_mean,eps_mean)
            for pos1 in range(num_pos):
                for pos2 in range(num_pos):
                    if pos1 != pos2 and (not (torch.allclose(l_x[pos1], u_x[pos1], atol=1e-8)) and(not torch.allclose(
                        l_x[pos2], u_x[pos2], atol=1e-8))):
                        XWX = -input_m_np[pos1] @ W @ input_m_np[pos2]
                        Q_bound = UNL_list[0]
                        if QK_output is None:
                            alpha_l[pos1, h, pos2, :] = torch.tensor(W @ input_m_np[pos2],dtype=torch.float32)
                            beta_l[pos1, h, pos2, :] = torch.tensor(W.T @ input_m_np[pos1],dtype=torch.float32)
                            temp = XWX - 0.5 * Q_bound["Obj_val"]
                            gamma_l[pos1, h, pos2] = torch.tensor(temp, dtype=torch.float32)

                            alpha_u[pos1, h, pos2, :] = alpha_l[pos1, h, pos2, :]
                            beta_u[pos1, h, pos2, :] = beta_l[pos1, h, pos2, :]
                            temp = XWX + 0.5 * Q_bound["Obj_val"]
                            gamma_u[pos1, h, pos2] = torch.tensor(temp, dtype=torch.float32)
                        else:
                            mDm_q = input_m_np[pos1] @ np.diag(Q_bound["D_query"]) @ input_m_np[pos1]
                            mDm_k = input_m_np[pos2] @ np.diag(Q_bound["D_key"])@ input_m_np[pos2]
                            # QK_output["in"][pos1]=input_m_np[pos1] ####
                            # QK_output["in"][pos2]=input_m_np[pos2] #####
                            X0_1=QK_output["in"][pos1]
                            X0_2=QK_output["in"][pos2]
                            alpha_l[pos1, h, pos2, :] = torch.tensor(
                                Q_bound["D_query"] * (X0_1 - input_m_np[pos1]) + W @ X0_2,dtype=torch.float32)
                            beta_l[pos1, h, pos2, :] = torch.tensor(
                                Q_bound["D_key"] * (X0_2 - input_m_np[pos2]) + W.T @ X0_1,dtype=torch.float32)
                            #===========================================
                            # alpha_l1=torch.tensor(W @ input_m_np[pos2],dtype=torch.float32).to(alpha_l[pos1, h, pos2, :].device) #####
                            # beta_l1=torch.tensor(W.T @ input_m_np[pos1],dtype=torch.float32).to(beta_l[pos1, h, pos2, :].device)########
                            # if not torch.allclose(alpha_l1, alpha_l[pos1, h, pos2, :], atol=1e-6):##############
                            #     print("alpha_loff errors")
                            #     exit(0)
                            # if not torch.allclose(beta_l1, beta_l[pos1, h, pos2, :], atol=1e-6):##############
                            #     print("beta_l_off errors")
                            #     exit(0)
                            # ===========================================
                            c = 0.5 * (mDm_q + mDm_k - Q_bound["Obj_val"])
                            XAX =X0_1 * Q_bound["D_query"] @ X0_1+X0_2 * Q_bound["D_key"] @ X0_2-2*XWX
                            temp=-0.5*XAX+c
                            gamma_l[pos1, h, pos2] = torch.tensor(temp, dtype=torch.float32)
                            #=========================================
                            # temp = XWX - 0.5 * Q_bound["Obj_val"]
                            # gamma_l1 = torch.tensor(temp, dtype=torch.float32).to(gamma_l.device)
                            # if not torch.allclose(gamma_l1, gamma_l[pos1, h, pos2], atol=1e-6):##############
                            #     print("gamma_l_off errors")
                            #     exit(0)
                            # =========================================
                            alpha_u[pos1, h, pos2, :] = torch.tensor(
                                Q_bound["D_query"] * (input_m_np[pos1] - X0_1) + W @ X0_2,dtype=torch.float32)
                            beta_u[pos1, h, pos2, :] = torch.tensor(
                                Q_bound["D_key"] * (input_m_np[pos2] - X0_2) + W.T @ X0_1,dtype=torch.float32)
                            #============================================
                            # alpha_u1=alpha_l[pos1, h, pos2, :]##########
                            # beta_u1=beta_l[pos1, h, pos2, :] ###############
                            # if not torch.allclose(alpha_u1, alpha_u[pos1, h, pos2, :], atol=1e-6):##############
                            #     print("alpha_u_off errors")
                            #     exit(0)
                            # if not torch.allclose(beta_u1, beta_u[pos1, h, pos2, :], atol=1e-6):##############
                            #     print("beta_u_off errors")
                            #     exit(0)
                            #==============================================
                            XAX=X0_1 * Q_bound["D_query"] @ X0_1+X0_2 * Q_bound["D_key"] @ X0_2+2*XWX
                            temp = 0.5*XAX -c
                            gamma_u[pos1, h, pos2] = torch.tensor(temp, dtype=torch.float32)
                            #================================
                            # temp = XWX + 0.5 * Q_bound["Obj_val"]
                            # gamma_u1= torch.tensor(temp, dtype=torch.float32)
                            # if not torch.allclose(gamma_u1, gamma_u[pos1, h, pos2], atol=1e-6):##############
                            #     print("gamma_u_off errors")
                            #     exit(0)

                            #=============================
    else:

        # ll_x=l_x.cpu().numpy()
        # uu_x=u_x.cpu().numpy()
        # for pos in range(num_pos):
        #     if np.allclose(ll_x[pos],uu_x[pos],atol=1e-8):
        #         uu_x[pos]+=0.01
        #         ll_x[pos]-=0.01
        #
        # C_obj_val,C_D_query,C_D_key=SDP_custom(ll_x, uu_x, W_Q.cpu().numpy(), W_K.cpu().numpy())
        #
        # SDP_list = {}
        #
        # for h in range(num_attention_heads):
        #     for pos1 in range(num_pos):
        #         for pos2 in range(num_pos):
        #             if pos1 != pos2:
        #                 UNL_list=[]
        #                 for bnd in [0, 1]:
        #                     Q_bound = {}
        #                     Q_bound["D_query"] = C_D_query[pos1,h,pos2].cpu().numpy()
        #                     Q_bound["D_key"] = C_D_key[pos1,h,pos2].cpu().numpy()
        #                     Q_bound["Time"] = 0
        #                     Q_bound["Obj_val"] = C_obj_val[pos1,h,pos2].cpu().numpy().reshape(-1)
        #                     UNL_list.append(Q_bound)
        #                     SDP_list[(pos1,h,pos2)]=UNL_list


        input_eps=input_eps.cpu().numpy()
        input_m_np=input_m.cpu().numpy()
        for h in range(num_attention_heads):
            # obj_val = torch.zeros((2, num_pos, num_pos), dtype=torch.float32)
            # D_query = torch.zeros((2, num_pos, num_pos, d_model), dtype=torch.float32)
            # D_key = torch.zeros((2, num_pos, num_pos, d_model), dtype=torch.float32)
            # dignal_pos = torch.zeros((2, num_pos, num_pos), dtype=torch.bool)
            # offDiagnal_pos = torch.zeros((2, num_pos, num_pos), dtype=torch.bool)
            W_q = W_Q[h, :, :].clone().cpu().numpy()
            W_q = W_q.T
            W_k = W_K[h, :, :].clone().cpu().numpy()
            W_k = W_k.T
            W = W_q @ W_k.T / np.sqrt(d_model)
            WW = W + W.T
            for pos1 in range(num_pos):
                for pos2 in range(num_pos):
                    UNL_list=None
                    if pos1==pos2:
                        if not torch.allclose(l_x[pos1], u_x[pos1], atol=1e-6):
                            #UNL_list=test_solve(d_in, W, input_eps[pos1])
                            UNL_list = SDP.diagSolve(W, input_eps[pos1])
                    else:
                        if (not torch.allclose(l_x[pos1], u_x[pos1], atol=1e-6)) and (not torch.allclose(
                        l_x[pos2], u_x[pos2], atol=1e-6)):
                            #UNL_list=SDP_list[(pos1,h,pos2)]
                            UNL_list = SDP.off_diagSolve(W,input_eps[pos1],input_eps[pos2])
                    if UNL_list is not None:
                        for bnd in [0,1]:
                            Q_bound = UNL_list[bnd]
                            # if pos1==pos2:
                            #     dignal_pos[bnd,pos1,pos2]=True
                            # else:
                            #     offDiagnal_pos[bnd,pos1,pos2]=True
                            # obj_val[bnd,pos1,pos2]=torch.tensor(Q_bound["Obj_val"])
                            # D_query[bnd,pos1,pos2]=torch.tensor(Q_bound["D_query"])
                            # D_key[bnd, pos1, pos2] = torch.tensor(Q_bound["D_key"])
                            if pos1==pos2 and bnd==0:
                                #lower
                                XWX = -input_m_np[pos1] @ WW @ input_m_np[pos1]
                                alpha_l[pos1, h, pos1, :] = torch.tensor(WW @ input_m_np[pos1], dtype=torch.float32)
                                temp = 0.5 * (XWX - Q_bound["Obj_val"])
                                gamma_l[pos1, h, pos1] = torch.tensor(temp, dtype=torch.float32)
                                beta_l[pos1, h, pos1, :] = beta_l[pos1, h, pos1, :] * 0

                            if pos1==pos2 and bnd==1:
                                XWX = -input_m_np[pos1] @ WW @ input_m_np[pos1]
                                alpha_u[pos1, h, pos1, :] = torch.tensor(WW @ input_m_np[pos1],dtype=torch.float32)
                                temp=0.5*(XWX+Q_bound["Obj_val"])
                                gamma_u[pos1, h, pos1] = torch.tensor(temp, dtype=torch.float32)
                                beta_u[pos1, h, pos1, :] = beta_u[pos1, h, pos1, :] * 0
                            if pos1!=pos2 and bnd==0:
                                XWX = -input_m_np[pos1] @ W @ input_m_np[pos2]
                                alpha_l[pos1, h, pos2, :] = torch.tensor(W @ input_m_np[pos2], dtype=torch.float32)
                                beta_l[pos1, h, pos2, :] = torch.tensor(W.T @ input_m_np[pos1], dtype=torch.float32)
                                temp = XWX - 0.5 * Q_bound["Obj_val"]
                                gamma_l[pos1, h, pos2] = torch.tensor(temp, dtype=torch.float32)
                            if pos1 != pos2 and bnd == 1:
                                XWX = -input_m_np[pos1] @ W @ input_m_np[pos2]
                                alpha_u[pos1, h, pos2, :] = alpha_l[pos1, h, pos2, :]
                                beta_u[pos1, h, pos2, :] = beta_l[pos1, h, pos2, :]
                                temp = XWX + 0.5 * Q_bound["Obj_val"]
                                gamma_u[pos1, h, pos2] = torch.tensor(temp, dtype=torch.float32)

            # W=torch.tensor(W).to(W_Q.device)
            # WW=torch.tensor(WW).to(W_Q.device)
            # input_m_t=input_m_t.repeat(1, num_pos).unsqueeze(-1).reshape(num_pos,num_pos,d_in,1)
            # co_l=WW@input_m[dignal_pos[0]]
            # alpha_l[:,h,:,:]=WW@

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

'''
def get_bounds_QK_bi(l_x, u_x,W_Q,W_K,QK_output=None):


    #===========
    # l_x=torch.tensor([[1, 1], [2, 2]])
    # u_x=torch.tensor([[1, 1], [2, 2]])
    # W_Q=torch.ones((2, 1, 1))
    # W_K = torch.ones((2, 1, 1))

    l_xnp=l_x.cpu().numpy()
    u_xnp=u_x.cpu().numpy()
    W_Qnp=W_Q.cpu().numpy()
    W_Knp=W_K.cpu().numpy()
    np.savez('arrays.npz',lx=l_xnp,ux=u_xnp,W_Q=W_Qnp,W_K=W_Knp)
    #========



    #solver_name=cvx.CVXOPT
    solver_name = cvx.MOSEK
    #solver_name = cvx.SCS
    num_pos= l_x.shape[0]
    d_in=W_Q.shape[2]
    num_attention_heads=W_Q.shape[0]
    d_model=W_Q.shape[1]

    num_attention_heads1=W_K.shape[0]
    d_model1=W_K.shape[1]


    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"


    l_a=l_x.repeat(1, num_pos).unsqueeze(-1).reshape(num_pos,num_pos,d_in,1).repeat(1, 1,1,d_in).reshape(-1)
    u_a=u_x.repeat(1, num_pos).unsqueeze(-1).reshape(num_pos,num_pos,d_in,1).repeat(1, 1, 1,d_in).reshape(-1)
    l_b=l_x.unsqueeze(0).repeat(num_pos, 1, 1).reshape(num_pos,num_pos,1,d_in).repeat(1, 1, d_in,1).reshape(-1)
    u_b=u_x.unsqueeze(0).repeat(num_pos, 1, 1).reshape(num_pos,num_pos,1,d_in).repeat(1, 1, d_in,1).reshape(-1)

    alpha_l1, beta_l1, gamma_l1, alpha_u1, beta_u1, gamma_u1=get_bounds_xy(l_a, u_a, l_b, u_b)

    alpha_l1=alpha_l1.reshape((num_pos,num_pos, d_in,d_in)).unsqueeze(1)
    beta_l1=beta_l1.reshape((num_pos,num_pos, d_in,d_in)).unsqueeze(1)
    alpha_u1=alpha_u1.reshape((num_pos,num_pos, d_in,d_in)).unsqueeze(1)
    beta_u1=beta_u1.reshape((num_pos,num_pos, d_in,d_in)).unsqueeze(1)
    gamma_l1=gamma_l1.reshape((num_pos,num_pos, d_in,d_in)).unsqueeze(1)
    gamma_u1=gamma_u1.reshape((num_pos,num_pos, d_in,d_in)).unsqueeze(1)

    A=torch.bmm(W_Q.transpose(1, 2),W_K)/torch.sqrt(torch.tensor(d_model,dtype=torch.float32))
    A=A.unsqueeze(0).unsqueeze(2)


    mask = torch.gt(A, 0.).to(torch.float)

    alpha_l=alpha_l1*mask*A+alpha_u1*(1-mask)*A
    alpha_l=torch.sum(alpha_l,dim=-1)

    alpha_u=alpha_u1*mask*A+alpha_l1*(1-mask)*A
    alpha_u=torch.sum(alpha_u,dim=-1)

    beta_l=beta_l1*mask*A+beta_u1*(1-mask)*A
    beta_l=torch.sum(beta_l,dim=-2)

    beta_u=beta_u1*mask*A+beta_l1*(1-mask)*A
    beta_u=torch.sum(beta_u,dim=-2)

    gamma_l=gamma_l1*mask*A+gamma_u1*(1-mask)*A
    gamma_l=torch.sum(gamma_l,dim=[-1,-2])

    gamma_u=gamma_u1*mask*A+gamma_l1*(1-mask)*A
    gamma_u=torch.sum(gamma_u,dim=[-1,-2])


    diff=torch.abs(u_x-l_x)
    equal_mask=torch.all(diff >= 1e-6, dim=1)

    input_m,input_eps = (u_x + l_x) / 2,(u_x - l_x) / 2

    masked_eps=input_eps[equal_mask]
    eps_mean = masked_eps.mean(dim=0)

    #eps_mean=input_eps.mean(dim=0)


    eps_mean=eps_mean.clone().cpu().numpy()
    input_m_np=input_m.clone().cpu().numpy()

    # alpha_l=np.zeros((num_pos, num_attention_heads, num_pos, d_in))
    # beta_l=np.zeros((num_pos, num_attention_heads, num_pos, d_in))
    # alpha_u=np.zeros((num_pos, num_attention_heads, num_pos, d_in))
    # beta_u=np.zeros((num_pos, num_attention_heads, num_pos, d_in))
    # gamma_l=np.zeros((num_pos, num_attention_heads, num_pos))
    # gamma_u=np.zeros((num_pos, num_attention_heads, num_pos))

    D_query_var = cvx.Variable(d_in, nonneg=True)
    W_param = cvx.Parameter((d_in, d_in))
    cons_PSD_d = cvx.diag(D_query_var) + W_param + W_param.T >> 0
    eps2_query = cvx.Parameter(d_in, nonneg=True)
    obj_d = cvx.Minimize(eps2_query @ D_query_var)
    prob_d = cvx.Problem(obj_d, [cons_PSD_d])
    eps2_query.value = eps_mean ** 2
    # for off-diagonal situation
    D_key_var = cvx.Variable(d_in, nonneg=True)
    W_param1 = cvx.Parameter((d_in, d_in))
    cons_PSD_od = cvx.bmat([[cvx.diag(D_query_var), W_param1], [W_param1.T, cvx.diag(D_key_var)]]) >> 0
    eps2_key = cvx.Parameter(d_in, nonneg=True)
    obj_od = cvx.Minimize(eps2_query @ D_query_var + eps2_key @ D_key_var)
    prob_od = cvx.Problem(obj_od, [cons_PSD_od])
    eps2_key.value = eps_mean ** 2

    for h in range(num_attention_heads):
        W_q=W_Q[h,:,:].clone().cpu().numpy()
        W_q=W_q.T
        W_k = W_K[h, :, :].clone().cpu().numpy()
        W_k=W_k.T
        W=W_q @ W_k.T / np.sqrt(d_model)
        # for diagonal situation

        # for lb
        W_param.value = -W
        time_start = time.time()
        obj_val = prob_d.solve(solver=solver_name)
        time_end = time.time()
        print("{}-diagnal case time: {:.3f}".format(h,(time_end - time_start)))
        D_query = D_query_var.value

        for pos in range(num_pos):
            b_query_l=D_query_var.value * input_m_np[pos]
            c=-0.5*obj_val
            if not torch.allclose(l_x[pos], u_x[pos], atol=1e-6):

                alpha_l[pos, h, pos, :] = torch.from_numpy((W+W.T) @ input_m_np[pos]).to(torch.float32)
                temp =-0.5 * input_m_np[pos] @ (W+W.T) @ input_m_np[pos]
                temp+=c
                gamma_l[pos, h, pos] = torch.tensor(temp,dtype=torch.float32)
                beta_l[pos, h, pos, :]=beta_l[pos, h, pos, :]*0
        #for ub
        W_param.value = W
        obj_val = prob_d.solve(solver=solver_name)
        D_query = D_query_var.value
        for pos in range(num_pos):
            b_query_u = D_query_var.value * input_m_np[pos]
            c=- 0.5*obj_val
            if not torch.allclose(l_x[pos], u_x[pos], atol=1e-6):
                alpha_u[pos, h, pos, :] = torch.from_numpy((W+W.T) @ input_m_np[pos]).to(torch.float32)
                temp = -0.5 * input_m_np[pos] @ (W+W.T) @ input_m_np[pos]
                temp-=c
                gamma_u[pos, h, pos] = torch.tensor(temp,dtype=torch.float32)
                beta_u[pos, h, pos, :]=beta_u[pos, h, pos, :]*0

        #for off-diagonal situation

        W_param1.value = W
        time_start = time.time()
        obj_val1 = prob_od.solve(solver=solver_name)
        time_end = time.time()
        print("{}-off-diagnal case time: {:.3f}".format(h,(time_end - time_start)))
        # Solutions for lower and upper bounds are the same
        for pos1 in range(num_pos):
            for pos2 in range(num_pos):
                if pos1 != pos2 and not torch.allclose(l_x[pos1], u_x[pos1], atol=1e-6) and not torch.allclose(l_x[pos2], u_x[pos2], atol=1e-6):
                    D_query=D_query_var.value
                    D_key=D_key_var.value
                    c = - 0.5*obj_val1
                    XAX = -input_m_np[pos1] @ W @  input_m_np[pos2]
                    alpha_l[pos1, h, pos2, :] = torch.from_numpy(W @ input_m_np[pos2]).to(torch.float32)
                    beta_l[pos1, h, pos2, :]=torch.from_numpy(W.T @ input_m_np[pos1]).to(torch.float32)
                    temp=XAX+c
                    gamma_l[pos1, h, pos2]=torch.tensor(temp,dtype=torch.float32)
                    alpha_u[pos1, h, pos2, :] = torch.from_numpy(W @ input_m_np[pos2]).to(torch.float32)
                    beta_u[pos1, h, pos2, :]=torch.from_numpy(W.T @ input_m_np[pos1]).to(torch.float32)
                    temp=XAX-c
                    gamma_u[pos1, h, pos2]=torch.tensor(temp,dtype=torch.float32)

    # for pos1 in range(num_pos):
    #     for pos2 in range(num_pos):
    #         for h in range(num_attention_heads):
    #             QK_checker(l_x, W_Q, W_K, alpha_l, beta_l, gamma_l, pos1, h, pos2)

    return alpha_l, beta_l, gamma_l,alpha_u, beta_u, gamma_u
'''

def get_bounds_xy_bi1(l_x, u_x, l_y, u_y,QK_output=None,V_flag=False):

    num_pos,num_attention_heads, d_model=l_x.shape[0],l_x.shape[1],l_x.shape[2]
    num_pos1,num_attention_heads1,d_model1=l_y.shape[0],l_y.shape[1],l_y.shape[2]

    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"

    query_m,query_eps = (u_x + l_x) / 2,(u_x - l_x) / 2
    key_m,key_eps = (u_y + l_y) / 2,(u_y - l_y) / 2

    query_eps=query_eps.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    query_m = query_m.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_eps=key_eps.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_m = key_m.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)

    # l_a=l_x.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    # u_a=u_x.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    # l_b=l_y.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    # u_b=u_y.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    #
    # alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = \
    #     get_bounds_xy(l_a, u_a, l_b, u_b)
    # gamma_l = gamma_l \
    #     .reshape(num_pos, num_attention_heads, num_pos1, d_model) \
    #     .sum(dim=-1)
    # gamma_u = gamma_u \
    #     .reshape(num_pos, num_attention_heads, num_pos1, d_model) \
    #     .sum(dim=-1)

    #zero_tensor=torch.zeros(num_pos, num_attention_heads, num_pos1, d_model).to(l_x.device)
    D_query=torch.where(torch.abs(query_eps)<1e-6,0,key_eps/query_eps)
    D_key=torch.where(torch.abs(key_eps)<1e-6,0,query_eps/key_eps)

    obj=torch.sum(query_eps*key_eps,dim=-1,keepdim=False)
    #c=-(query_eps.unsqueeze(-2)@key_eps.unsqueeze(-1))

    alpha_l=key_m
    alpha_u=alpha_l

    beta_l =query_m
    beta_u=beta_l
    # alpha_lXq + Beta_l Xk+Gmma_l

    XWX=torch.sum(query_m*key_m,dim=-1,keepdim=False)

    gamma_l=-XWX-obj
    gamma_u=-XWX+obj
    if QK_output is not None:
        print("tangent plane move")
        # D_query=D_query.reshape(num_pos, num_attention_heads,num_pos1, d_model)
        # D_key=D_key.reshape(num_pos, num_attention_heads,num_pos1, d_model)

        mDm_q = query_m.unsqueeze(-2) @ torch.diag_embed(D_query) @ query_m.unsqueeze(-1)
        mDm_k = key_m.unsqueeze(-2) @ torch.diag_embed(D_key) @ key_m.unsqueeze(-1)
        mDm_q=mDm_q.squeeze(-1).squeeze(-1)
        mDm_k=mDm_k.squeeze(-1).squeeze(-1)

        if V_flag:
            X0_1 =QK_output["attention_probs"].squeeze(0).transpose(0, 1)
            X0_1=X0_1.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
            X0_2 =QK_output["value"].squeeze(0).permute(2,0,1)

            X0_2=X0_2.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
        else:
            X0_1 =QK_output["query"].squeeze(0).transpose(0, 1)
            X0_1=X0_1.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
            X0_2 =QK_output["key"].squeeze(0).transpose(0, 1)
            X0_2 =X0_2.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)

        alpha_l = D_query * (X0_1 - query_m) + X0_2
        beta_l = D_key * (X0_2 - key_m) + X0_1
        # if torch.any(torch.isnan(X0_2 - key_m)):
        #     print("X0_2 - key_m has NAN")
        c = 0.5 * (mDm_q + mDm_k) - obj
        XAX = ((X0_1 * D_query).unsqueeze(-2) @ X0_1.unsqueeze(-1) + (X0_2 * D_key).unsqueeze(-2) @ X0_2.unsqueeze(-1)).squeeze(-1).squeeze(-1) +\
              2*torch.sum(X0_1*X0_2,dim=-1,keepdim=False)
        gamma_l = -0.5 * XAX + c

        alpha_u = D_query * (query_m-X0_1) + X0_2
        beta_u = D_key * (key_m-X0_2) + X0_1

        XAX = ((X0_1 * D_query).unsqueeze(-2) @ X0_1.unsqueeze(-1) + (X0_2 * D_key).unsqueeze(-2) @ X0_2.unsqueeze(-1)).squeeze(-1).squeeze(-1) - \
              2*torch.sum(X0_1*X0_2,dim=-1,keepdim=False)
        gamma_u = 0.5 * XAX - c
        #
        # if torch.any(torch.isnan(D_query)):
        #     print("D_query has NAN")


    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

def get_bounds_xy_bi_adj(l_x, u_x, l_y, u_y,X0_1,X0_2,X1_1,X1_2):
    num_pos,num_attention_heads, d_model=l_x.shape[0],l_x.shape[1],l_x.shape[2]
    num_pos1,num_attention_heads1,d_model1=l_y.shape[0],l_y.shape[1],l_y.shape[2]

    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"

    query_m,query_eps = (u_x + l_x) / 2,(u_x - l_x) / 2
    key_m,key_eps = (u_y + l_y) / 2,(u_y - l_y) / 2

    query_eps=query_eps.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    query_m = query_m.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X0_1=X0_1.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X1_1=X1_1.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_eps=key_eps.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_m = key_m.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X0_2=X0_2.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X1_2 = X1_2.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)

    D_query=torch.where(torch.abs(query_eps)<1e-6,0,key_eps/query_eps)
    D_key=torch.where(torch.abs(key_eps)<1e-6,0,query_eps/key_eps)

    obj=torch.sum(query_eps*key_eps,dim=-1,keepdim=False)

    # D_query=D_query.reshape(num_pos, num_attention_heads,num_pos1, d_model)
    # D_key=D_key.reshape(num_pos, num_attention_heads,num_pos1, d_model)

    mDm_q = query_m.unsqueeze(-2) @ torch.diag_embed(D_query) @ query_m.unsqueeze(-1)
    mDm_k = key_m.unsqueeze(-2) @ torch.diag_embed(D_key) @ key_m.unsqueeze(-1)
    mDm_q = mDm_q.squeeze(-1).squeeze(-1)
    mDm_k = mDm_k.squeeze(-1).squeeze(-1)


    alpha_l = D_query * (X0_1 - query_m) + X0_2
    beta_l = D_key * (X0_2 - key_m) + X0_1

    c = 0.5 * (mDm_q + mDm_k) - obj
    XAX1=((X0_1 * D_query).unsqueeze(-2) @ X0_1.unsqueeze(-1) + (X0_2 * D_key).unsqueeze(-2) @ X0_2.unsqueeze(-1)).squeeze(
        -1).squeeze(-1)
    XAX2=2 * torch.sum(X0_1 * X0_2, dim=-1, keepdim=False)

    gamma_l = -0.5 * (XAX1+XAX2) + c

    XAX11=((X1_1 * D_query).unsqueeze(-2) @ X1_1.unsqueeze(-1) + (X1_2 * D_key).unsqueeze(-2) @ X1_2.unsqueeze(-1)).squeeze(
        -1).squeeze(-1)
    XAX22=2 * torch.sum(X1_1 * X1_2, dim=-1, keepdim=False)

    alpha_u = D_query * (query_m - X1_1) + X1_2
    beta_u = D_key * (key_m - X1_2) + X1_1

    gamma_u = 0.5 * (XAX11-XAX22) - c


    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

def get_bounds_xy_adj(l_x, u_x, l_y, u_y,X0,X1):
    num_pos,num_attention_heads, d_model=l_x.shape[0],l_x.shape[1],l_x.shape[2]
    num_pos1,num_attention_heads1,d_model1=l_y.shape[0],l_y.shape[1],l_y.shape[2]

    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"

    query_m,query_eps = (u_x + l_x) / 2,(u_x - l_x) / 2
    key_m,key_eps = (u_y + l_y) / 2,(u_y - l_y) / 2

    query_eps=query_eps.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    query_m = query_m.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X0_1=X0.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X1_1 = X1.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_eps=key_eps.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_m = key_m.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    #X0_2=X0.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)

    alpha_l = X0_1*key_eps +key_m
    beta_l = X0_1*query_eps+query_m
    gamma_l=-query_m*key_m-query_eps*key_eps-X0_1*(query_m*key_eps+key_m*query_eps)
    gamma_l=gamma_l.sum(dim=-1)

    alpha_u=-X1_1*key_eps +key_m
    beta_u=X1_1*query_eps+query_m
    gamma_u = -query_m * key_m + query_eps * key_eps + X1_1 * (query_m * key_eps - key_m * query_eps)
    gamma_u=gamma_u.sum(dim=-1)

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

def get_bounds_xy_bi(l_x, u_x, l_y, u_y):
    num_pos, d_model=l_x.shape[0],l_x.shape[1]
    num_pos1,d_model1=l_y.shape[0],l_y.shape[1]

    # assert num_pos==num_pos1, "the length of query and key should be the same"
    # assert d_model==d_model1,"the dim of query and key should be the same"

    query_l=l_x.to('cpu').numpy()
    query_u=u_x.to('cpu').numpy()
    key_l=l_y.to('cpu').numpy()
    key_u=u_y.to('cpu').numpy()
    query_m = (query_u + query_l) / 2
    query_eps = (query_u - query_l) / 2
    key_m = (key_u + key_l) / 2
    key_eps = (key_u - key_l) / 2

    D_query_var = cvx.Variable(d_model, nonneg=True)
    D_key_var = cvx.Variable(d_model, nonneg=True)
    W_param = cvx.Parameter((d_model, d_model))
    W=np.eye(d_model)*2

    eps2_query = cvx.Parameter(d_model, nonneg=True)
    eps2_key = cvx.Parameter(d_model, nonneg=True)

    cons_PSD_od = cvx.bmat([[cvx.diag(D_query_var), W_param], [W_param.T, cvx.diag(D_key_var)]]) >> 0

    obj_od = cvx.Minimize(eps2_query @ D_query_var + eps2_key @ D_key_var)
    obj_od = cvx.Minimize(0)
    # Optimization problem
    #prob_od = cvx.Problem(obj_od, [cons_PSD_od])
    prob_od = cvx.Problem(obj_od, [cons_PSD_od])
    #prob.is_dpp()

    D_query = np.empty((num_pos, num_pos, 2, d_model))
    D_key = np.empty((num_pos, num_pos, 2, d_model))
    b_query = np.empty((num_pos, num_pos, 2, d_model))
    b_key = np.empty((num_pos, num_pos, 2, d_model))

    alpha_l= np.zeros((num_pos, num_pos1, d_model))
    alpha_u= np.zeros((num_pos, num_pos1, d_model))
    beta_l= np.zeros((num_pos, num_pos1, d_model))
    beta_u= np.zeros((num_pos, num_pos1, d_model))
    gamma_l=np.zeros((num_pos, num_pos1))
    gamma_u= np.zeros((num_pos, num_pos1))



    # Constant terms
    c = np.empty((num_pos, num_pos, 2))
    # Mean gap
    gapMean = np.empty((num_pos, num_pos, 2))

    # Iterate over queries
    for query in range(num_pos):
        # Iterate over keys
        for key in range(num_pos1):

            # gamma_l[query, key] = query_l[query] @ key_l[key]
            # gamma_u[query, key] = query_l[query] @ key_l[key]
            # continue
            if np.all(np.isclose(query_eps[query], 0)) and np.all(np.isclose(key_eps[key], 0)):
                gamma_l[query,key]=query_l[query]@key_l[key]
                gamma_u[query,key]=query_l[query]@key_l[key]
                continue
            # Instantiate objective function weights
            eps2_query.value = query_eps[query] ** 2
            eps2_key.value = key_eps[key] ** 2


            # Solutions for lower and upper bounds are the same
            for bnd in [0, 1]:
                # Instantiate W
                W_param.value = -W if bnd == 1 else W
                time_start=time.time()
                # Solve problem
                obj_val = prob_od.solve()
                time_end = time.time()
                print("({},{})SDP solving time: {:.3f}".format(query,key,time_end - time_start))
                # Mean gap
                gapMean[query, key, bnd] = obj_val / 3
                # Diagonal matrices
                D_query[query, key, bnd] = D_query_var.value
                D_key[query, key, bnd] = D_key_var.value
                # Linear term coefficients
                b_query[query, key, bnd] = -D_query_var.value * query_m[query]
                b_key[query, key, bnd] = -D_key_var.value * key_m[key]
                # Constant term
                c[query, key, bnd] = -(b_query[query, key, bnd] @ query_m[query] + b_key[query, key, bnd] @ key_m[key] + obj_val) / 2

                #1/2(X^T A X)+B^TX+C
                A =np.block([[np.diag(D_query[query, key, bnd]) , (-1) ** bnd *W],[(-1) ** bnd *W.T, np.diag(D_key[query, key, bnd])]])
                B=np.concatenate((b_query[query, key, bnd],b_key[query, key, bnd]))
                C=c[query, key, bnd]
                #====getMiddlePlane====
                x0=np.concatenate((query_m[query],key_m[key]))
                f_x0=(-1) ** bnd*0.5* x0 @ A @ x0+(-1) ** bnd*B @ x0 + (-1) ** bnd*C
                grad_f_x0=(-1) ** bnd*A @ x0+(-1) ** bnd*B
                constLine=f_x0-(grad_f_x0 @ x0)

                if bnd == 0:
                    alpha_l[query,key]=grad_f_x0[:d_model]
                    beta_l[query,key]=grad_f_x0[d_model:]
                    gamma_l[query,key]=constLine
                else:
                    alpha_u[query,key]=grad_f_x0[:d_model]
                    beta_u[query,key]=grad_f_x0[d_model:]
                    gamma_u[query,key]=constLine

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u
    num_pos,num_attention_heads, d_model=l_x.shape[0],l_x.shape[1],l_x.shape[2]
    num_pos1,num_attention_heads1,d_model1=l_y.shape[0],l_y.shape[1],l_y.shape[2]

    assert num_attention_heads == num_attention_heads1, "the number of attention heads should be the same"
    assert d_model == d_model1, "the dim of query and key should be the same"

    query_m,query_eps = (u_x + l_x) / 2,(u_x - l_x) / 2
    key_m,key_eps = (u_y + l_y) / 2,(u_y - l_y) / 2

    query_eps=query_eps.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    query_m = query_m.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X0_1=X0_1.repeat(1, 1, num_pos1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_eps=key_eps.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    key_m = key_m.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)
    X0_2=X0_2.transpose(0, 1).repeat(num_pos, 1, 1).reshape(num_pos, num_attention_heads, num_pos1, d_model)


def get_bounds_xy(l_x, u_x, l_y, u_y):
    alpha_l = l_y
    beta_l = l_x
    gamma_l = -alpha_l * beta_l

    alpha_u = u_y
    beta_u = l_x
    gamma_u = -alpha_u * beta_u

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

def get_bounds_xy1(l_x, u_x, l_y, u_y):
    alpha_l = u_y
    beta_l = u_x
    gamma_l = -alpha_l * beta_l

    alpha_u = l_y
    beta_u = u_x
    gamma_u = -alpha_u * beta_u

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u