#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:03:42 2024

Custom parallelized SDP solver for optimizing quadratic bounds on bilinear functions
"""

import time
import numpy as np
import pandas as pd
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #else "mps" if torch.backends.mps.is_available()

device=torch.device("cpu")


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    # 创建输出张量的副本以避免原地修改
    out = input.clone()

    shape = input.shape
    # 计算对角线长度
    diag_len = min(shape[dim1], shape[dim2]) - abs(offset)
    if diag_len <= 0:
        # 如果偏移过大导致无有效对角线长度，则直接返回
        return out

    # 创建对角线索引序列
    idx = torch.arange(diag_len, device=input.device)

    # 根据offset确定dim1与dim2维度的索引对应关系
    if offset >= 0:
        idx1 = idx
        idx2 = idx + offset
    else:
        idx1 = idx - offset
        idx2 = idx

    # 构建索引列表，用切片表示除dim1和dim2之外的维度全部取全
    indices = [slice(None)] * input.ndim
    indices[dim1] = idx1
    indices[dim2] = idx2

    # 将src的值赋给output对角线位置
    out[indices] = src

    return out


def SDP_custom(l_x, u_x, W_Q, W_K, index_tuple=None):
    """
    Solve SDPs using custom parallelized solver and return outputs according to API

    Parameters
    ----------
    l_x : (n, d_model) array
        Lower bounds on inputs to bilinear function.
    u_x : (n, d_model) array
        Upper bounds on inputs to bilinear function.
    W_Q : (num_heads, d_key, d_model) array
        Weight matrices multiplying queries.
    W_K : (num_heads, d_key, d_model) array
        Weight matrices multiplying keys.
    index_tuple : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    obj_val : (n, num_heads, n, 1) Tensor
        Optimal objective values.
    D_query : (n, num_heads, n, d_model) Tensor
        Optimal D_query parameters
    D_key : (n, num_heads, n, d_model) Tensor
        Optimal D_key parameters

    """
    # Dimensions
    num_heads, d_key, d_model = W_Q.shape
    n = l_x.shape[0]
    # Interval half-widths for inputs
    eps = (u_x - l_x) / 2

    # Initialize outputs
    obj_val = torch.zeros(n, num_heads, n)
    D_query = torch.zeros(n, num_heads, n, d_model)
    D_key = torch.zeros(n, num_heads, n, d_model)
    # Mask for off-diagonal elements
    ind_off_diag = ~torch.eye(n, dtype=bool)
    
    # Iterate over heads
    for h in range(num_heads):
        # Call custom SDP solver
        qboc = QuadBoundOptimizerCustom(W_Q[h].T, W_K[h].T, eps, scale_eps=True)
        qboc.optimize()

        # Save objective value and D parameters
        obj_val[:, h, :][ind_off_diag] = -qboc.obj_dual.flatten()
        D_query[:, h, :][ind_off_diag] = qboc.d.view(-1, 2 * d_model)[:, :d_model]
        D_key[:, h, :][ind_off_diag] = qboc.d.view(-1, 2 * d_model)[:, d_model:]

    return obj_val.unsqueeze(-1), D_query, D_key


class QuadBoundOptimizerCustom:
    def __init__(self, W_Q, W_K, eps, scale_eps=False, nu=10):
        """
        Class for optimizing quadratic bounds on bilinear functions using custom parallelized SDP solver

        Parameters
        ----------
        W_Q : (d_model, d_key) array
            Weight matrix multiplying queries.
        W_K : (d_model, d_key) array
            Weight matrix multiplying keys.
        eps : (n, d_model) array
            Interval half-widths for inputs to bilinear function.
        scale_eps : bool
            Scale W_Q, W_K by eps and then set eps equal to ones.
        nu : float
            Potential function parameter

        Returns
        -------
        None.

        """
        self.scale_eps = scale_eps
        # Dimensions
        self.d_model, self.d_key = W_Q.shape
        self.n = eps.shape[0]
        
        # PARAMETERS FOR ALL n x (n-1) OFF-DIAGONAL SDPs
        
        # Potential function parameter
        self.C_nu = 2 * self.d_model + nu * np.sqrt(2 * self.d_model)
        
        # Half-widths for query and key inputs for all off-diagonal SDPs
        # First construct position indices corresponding to queries and keys
        idx_query_key = (~torch.eye(self.n, dtype=bool)).nonzero()
        idx_query = idx_query_key[:, 0].reshape(self.n, self.n - 1)
        idx_key = idx_query_key[:, 1].reshape(self.n, self.n - 1)
        self.eps_query = torch.from_numpy(eps)[idx_query].to(device)
        self.eps_key = torch.from_numpy(eps)[idx_key].to(device)
        
        # Product of query and key weight matrices (unscaled)
        self.W_QK = torch.from_numpy(W_Q @ W_K.T / np.sqrt(self.d_key)).float().to(device)
        # Product of query and key weight matrices scaled by epsilons
        self.Wtilde_QK = self.eps_query.unsqueeze(-1) * self.W_QK * self.eps_key.unsqueeze(-2)

        if scale_eps:
            # Use only scaled W_QK
            self.W_QK = self.Wtilde_QK
            # Set epsilon^2 cost vector to ones
            self.eps2 = torch.ones(self.n, self.n - 1, 2 * self.d_model, device=device)
        else:
            # No need to expand unscaled W_QK
            # epsilon^2 cost vector
            self.eps2 = torch.cat((self.eps_query ** 2, self.eps_key ** 2), dim=-1)
        
        return
    
    def init_solutions(self, delta=0.01):
        """
        Initialize solutions to primal and dual SDPs

        Parameters
        ----------
        delta : float
            Small factor to ensure positive definiteness
        
        Returns
        -------
        None.

        """
        # Initial solutions are based on SVD of Wtilde
        Wtilde_SVD = torch.linalg.svd(self.Wtilde_QK)
        
        # Initial dual solution based on largest singular value of Wtilde
        self.d = (1 + delta) * Wtilde_SVD.S.select(-1, 0).unsqueeze(-1).expand_as(self.eps2)
        if not self.scale_eps:
            # Divide by epsilon^2
            self.d /= self.eps2
        
        # Construct matrices W and S
        self.W = torch.diag_embed(torch.zeros_like(self.d))
        self.W[:, :, :self.d_model, self.d_model:] = self.W_QK
        self.W[:, :, self.d_model:, :self.d_model] = self.W_QK.transpose(-2, -1)


        self.S = torch.diagonal_scatter(self.W, self.d, dim1=-2, dim2=-1)
        
        # Initial primal solution based on singular vectors of Wtilde
        self.X = torch.diag_embed(self.eps2)
        X_QK = -(1 - delta) * Wtilde_SVD.U @ Wtilde_SVD.Vh
        if not self.scale_eps:
            # Scale by epsilons
            X_QK = self.eps_query.unsqueeze(-1) * X_QK * self.eps_key.unsqueeze(-2)
        self.X[:, :, :self.d_model, self.d_model:] = X_QK
        self.X[:, :, self.d_model:, :self.d_model] = X_QK.transpose(-2, -1)
        
        # Compute Cholesky factorizations and check positive definiteness
        self.L_S, info_S = torch.linalg.cholesky_ex(self.S)
        self.L_X, info_X = torch.linalg.cholesky_ex(self.X)
        assert not info_S.any()
        assert not info_X.any()
        
        # Compute objective values and duality gap
        self.eval_solutions()

        return
    
    def eval_solutions(self, check_duality_gap=False):
        """
        Evaluate primal and dual objective values and duality gap

        Parameters
        ----------
        check_duality_gap : bool
            Check that duality gap is equal to difference between primal and dual objectives
        
        Returns
        -------
        None.

        """
        # Dual objective
        self.obj_dual = -(self.eps2 * self.d).sum(dim=-1)
        # Primal objective
        self.obj_primal = (self.W * self.X).sum(dim=[-2, -1])
        # Duality gap
        self.duality_gap = (self.S * self.X).sum(dim=[-2, -1])
        if check_duality_gap:
            assert torch.allclose(self.duality_gap, self.obj_primal - self.obj_dual)
        
        return
    
    def optimize(self, tol=1e-6, it_max=40, delta=0.1, verbose=True):
        """
        Optimize bound parameters using primal-dual interior-point algorithm

        Parameters
        ----------
        tol : float
            Tolerance on largest duality gap.
        it_max : int
            Maximum number of iterations.
        delta : float
            Small factor to ensure updated solutions are positive definite.
        verbose : bool
            Print statistics for each iteration.

        Returns
        -------
        None.

        """
        # Initialize statistics DataFrame
        stats = pd.DataFrame(index=range(it_max + 1), columns=["max duality gap", "time"])
        stats.index.name = "iter"
        
        # Initialize solutions to primal and dual SDPs
        it = 0
        start_time = time.perf_counter()
        self.init_solutions(delta=delta)
        iter_time = time.perf_counter() - start_time
        # Save statistics
        stats.loc[it] = self.duality_gap.max().item(), iter_time
        if verbose:
            print(stats.loc[[it]])
        
        # Primal-dual interior-point iterations
        for it in range(1, it_max + 1):
            start_time = time.perf_counter()
            self.optimize_one_step(delta=delta)
            iter_time = time.perf_counter() - start_time
            
            # Save statistics
            stats.loc[it] = self.duality_gap.max().item(), iter_time
            if verbose:
                print(stats.loc[[it]])
            
            if self.duality_gap.max() < tol:
                # Largest duality gap below tolerance, stop
                stats = stats.loc[:it]
                break
        
        return
    
    def optimize_one_step(self, delta=0.1):
        """
        Perform one step of primal-dual optimization

        Parameters
        ----------
        delta : float
            Small factor to ensure updated solutions are positive definite.

        Returns
        -------
        None.

        """
        # Compute primal and dual search directions
        Delta_d, Delta_X = self.compute_search_dirs()
        
        # Compute parameters and instantiate change-in-potential-function object for plane search
        pot_obj, step_sizes_max = self.prepare_plane_search(Delta_d, Delta_X, delta=delta)

        # Perform plane search using gradient descent and Armijo line search
        step_sizes = plane_search_GD(pot_obj, step_sizes_max)
        
        # Update solutions
        self.update_dual(Delta_d, step_sizes.select(-1, 0), delta=delta)
        self.update_primal(Delta_X, step_sizes.select(-1, 1), delta=delta)
        # Update objective values and duality gap
        self.eval_solutions()
        
        return
    
    def compute_search_dirs(self):
        """
        Compute primal and dual search directions

        Returns
        -------
        Delta_d : (n, n-1, 2 * d_model) Tensor
            Dual search directions.
        Delta_X : (n, n-1, 2 * d_model, 2 * d_model) Tensor
            Primal search directions.

        """
        # Inverse of S given Cholesky factor
        S_inv = torch.cholesky_inverse(self.L_S)

        # Compute dual search direction
        L_SX, info = torch.linalg.cholesky_ex(S_inv * self.X)
        assert (info == 0).all(), f"S^{-1} * X is not positive definite, info = {info}"
        rho = self.C_nu / self.duality_gap
        Delta_d = torch.cholesky_solve((torch.diagonal(S_inv, dim1=-2, dim2=-1) - rho.unsqueeze(-1) * torch.diagonal(self.X, dim1=-2, dim2=-1)).unsqueeze(-1), L_SX).squeeze(-1)
        
        # Compute primal search direction
        Delta_X = S_inv - rho.unsqueeze(-1).unsqueeze(-1) * self.X - (S_inv * Delta_d.unsqueeze(-2)) @ self.X
        # Set diagonal to zero and symmetrize
        Delta_X = torch.diagonal_scatter(Delta_X, torch.zeros(Delta_X.shape[:-1]), dim1=-2, dim2=-1)
        Delta_X = (Delta_X + Delta_X.transpose(-2, -1)) / 2
    
        return Delta_d, Delta_X        
    
    def prepare_plane_search(self, Delta_d, Delta_X, delta=0.1):
        """
        Compute parameters and instantiate change-in-potential-function object for plane search

        Parameters
        ----------
        Delta_d : (n, n-1, 2 * d_model) Tensor
            Dual search directions.
        Delta_X : (n, n-1, 2 * d_model, 2 * d_model) Tensor
            Primal search directions.
        delta : float
            Small factor to ensure updated solutions are positive definite.

        Returns
        -------
        pot_obj : PotentialChange
            Object for computing change in potential function
        step_sizes_max : (n, n-1, 2) Tensor
            Upper bounds on step sizes
        
        """
        # Constants for plane search
        c_step = torch.stack(((self.eps2 * Delta_d).sum(dim=-1) / self.duality_gap, (self.W * Delta_X).sum(dim=[-2, -1]) / self.duality_gap), dim=-1)
    
        # Generalized eigenvalues for plane search
        eigvals = torch.stack((gen_eigvalsh(torch.diag_embed(Delta_d), self.L_S), gen_eigvalsh(Delta_X, self.L_X)), dim=-1)
    
        # Instantiate change-in-potential-function object
        pot_obj = PotentialChange(c_step, eigvals, self.C_nu)
        
        # Upper bounds on step sizes
        step_sizes_max = torch.full_like(c_step, torch.inf)
        eigvals_min = eigvals.select(-2, 0)
        step_sizes_max[eigvals_min < 0] = -(1 - delta) / eigvals_min[eigvals_min < 0]
        
        return pot_obj, step_sizes_max
    
    def update_dual(self, Delta_d, step_sizes, delta=0.05):
        """
        Update dual solution

        Parameters
        ----------
        Delta_d : (n, n-1, 2 * d_model) Tensor
            Dual search directions.
        step_sizes : (n, n-1) Tensor
            Dual step sizes.
        delta : float
            Decrease in step size if new solution is not positive definite.

        Returns
        -------
        None.

        """
        # Default update
        beta = 1
        d_new = self.d + beta * step_sizes.unsqueeze(-1) * Delta_d
        self.S = torch.diagonal_scatter(self.S, d_new, dim1=-2, dim2=-1)
        # Cholesky factorization
        self.L_S, info = torch.linalg.cholesky_ex(self.S)
        
        # Check for non-positive definite S
        while info.any():
            print(f"S is not positive definite at indices {info.nonzero()}")
            # Decrease step size
            beta -= delta
            d_new[info > 0] = self.d[info > 0] + beta * step_sizes[info > 0].unsqueeze(-1) * Delta_d[info > 0]
            self.S = torch.diagonal_scatter(self.S, d_new, dim1=-2, dim2=-1)
            self.L_S, info = torch.linalg.cholesky_ex(self.S)
        
        # Updated solution
        self.d = d_new
        
        return
    
    def update_primal(self, Delta_X, step_sizes, delta=0.05):
        """
        Update primal solution

        Parameters
        ----------
        Delta_X : (n, n-1, 2 * d_model, 2 * d_model) Tensor
            Primal search directions.
        step_sizes : (n, n-1) Tensor
            Primal step sizes.
        delta : float
            Decrease in step size if new solution is not positive definite.

        Returns
        -------
        None.

        """
        # Default update
        beta = 1
        X_new = self.X + beta * step_sizes.unsqueeze(-1).unsqueeze(-1) * Delta_X
        # Cholesky factorization
        self.L_X, info = torch.linalg.cholesky_ex(X_new)
        
        # Check for non-positive definite X
        while info.any():
            print(f"X is not positive definite at indices {info.nonzero()}")
            print((X_new - self.X)[info > 0].norm(dim=[-2, -1]) / Delta_X[info > 0].norm(dim=[-2, -1]))
            # Decrease step size
            beta -= delta
            X_new[info > 0] = self.X[info > 0] + beta * step_sizes[info > 0].unsqueeze(-1).unsqueeze(-1) * Delta_X[info > 0]
            self.L_X, info = torch.linalg.cholesky_ex(X_new)
        
        # Updated solution
        self.X = X_new
        
        return
    


def gen_eigvalsh(A, L_B):
    """
    Compute generalized eigenvalues of (A, B) given Cholesky factor L_B of B
    
    Parameters
    ----------
    A : (*, 2 * d_model, 2 * d_model) Tensor
        Batch of square A matrices.
    L_B : (*, 2 * d_model, 2 * d_model) Tensor
        Batch of lower-triangular Cholesky factors.
    
    Returns
    -------
    eigvals : (*, 2 * d_model) Tensor
        Batch of generalized eigenvalues

    """
    
    # Compute L_B^{-1} A L_B^{-T}
    #temp=scipy.linalg.solve_triangular(L_B, A, upper=False)
    #temp=scipy.linalg.solve_triangular(torch.transpose(L_B, -2, -1), temp, upper=True, left=False)
    temp = torch.linalg.solve_triangular(L_B, A, upper=False)
    temp = torch.linalg.solve_triangular(torch.transpose(L_B, -2, -1), temp, upper=True, left=False)
    # Compute eigenvalues of L_B^{-1} A L_B^{-T}
    eigvals = torch.linalg.eigvalsh(temp)

    return eigvals


class PotentialChange:
    def __init__(self, c_step, eigvals, C_nu):
        """
        Class for computing change in potential function
        
        Parameters
        ----------
        c_step : (*, 2) Tensor
            Potential function parameters corresponding to changes in dual and primal objectives ("*" denotes batch dimensions)
        eigvals : (*, 2 * d_model, 2)
            Potential function parameters that are eigenvalues
        C_nu : float
            Potential function parameter that multiplies the log(change in objectives) term

        Returns
        -------
        None.

        """
        self.c_step = c_step
        self.eigvals = eigvals
        self.C_nu = C_nu

    def eval(self, step_sizes, grad=False):
        """
        Evaluate change in potential function and its gradient
        
        Parameters
        ----------
        step_sizes : (*, 2) Tensor
            Dual and primal step sizes at which to evaluate change in potential
        grad : bool
            Also evaluate gradient at step_sizes
            
        Returns
        -------
        pot : (*) Tensor
            Change in potential function
        pot_grad : (*, 2) Tensor
            Gradient of potential function at step_sizes
        """
        # Compute building blocks
        c_denom = 1 + (self.c_step * step_sizes).sum(dim=-1)
        eigvals_denom = 1 + step_sizes.unsqueeze(-2) * self.eigvals
        # Potential function
        pot = self.C_nu * torch.log(c_denom) - torch.log(eigvals_denom).sum(dim=[-2, -1])

        if not grad:
            return pot
        else:
            # Compute fractions
            c_frac = self.c_step / c_denom.unsqueeze(-1)
            eigvals_frac = self.eigvals / eigvals_denom
            # Gradient
            pot_grad = self.C_nu * c_frac
            pot_grad -= eigvals_frac.sum(dim=-2)

            return pot, pot_grad


def plane_search_GD(pot_obj, step_sizes_max, it_max=5, alpha_min=1e-18):
    """
    Perform plane search using gradient descent and Armijo line search

    Parameters
    ----------
    pot_obj : PotentialChange
        Object for computing change in potential function
    step_sizes_max : (*, 2) Tensor
        Upper bounds on step sizes
    it_max : int
        Maximum number of plane search iterations
    alpha_min : float
        Minimum value of alpha
    
    Returns
    -------
    step_sizes : (*, 2) Tensor
        Dual and primal step sizes
    
    """
    # Initialize step sizes
    step_sizes = torch.zeros_like(step_sizes_max)
    # Initial change in potential function and its gradient
    pot, pot_grad = pot_obj.eval(step_sizes, grad=True)
    
    for it in range(it_max):
        # Determine new step sizes using Armijo line search
        step_sizes, pot, pot_grad, alpha = armijo_line_search(step_sizes, -pot_grad, step_sizes_max, pot, pot_grad, pot_obj, alpha_min=alpha_min)
        if (alpha <= alpha_min).all():
            break

    return step_sizes


def armijo_line_search(step_sizes, search_dir, step_sizes_max, pot, pot_grad, pot_obj, sigma=0.5, tau=0.5, alpha_min=1e-18):
    """
    Perform line search to satisfy Armijo condition
    
    Parameters
    ----------
    step_sizes : (*, 2) Tensor
        Current dual and primal step sizes
    search_dir : (*, 2) Tensor
        Search direction in step size space
    step_sizes_max : (*, 2) Tensor
        Upper bounds on step sizes
    pot : (*) Tensor
        Change in potential function at step_sizes
    pot_grad : (*, 2) Tensor
        Gradient of potential function at step_sizes
    pot_obj : PotentialChange
        Object for computing change in potential function
    sigma : float
        Armijo condition threshold parameter
    tau : float
        Back-tracking factor
    alpha_min : float
        Minimum value of alpha
    
    Returns
    -------
    step_sizes_new : (*, 2) Tensor
        Updated step sizes
    pot_new : (*) Tensor
        Change in potential function at step_sizes_new
    pot_grad_new : (*, 2) Tensor
        Gradient of potential function at step_sizes_new
    alpha : float
        Multiplier of search_dir

    """
    # Armijo condition threshold
    thresh = -sigma * (pot_grad * search_dir).sum(dim=-1)
    # Initial alpha
    alpha = ((step_sizes_max - step_sizes) / search_dir).min(dim=-1).values
    step_sizes_new = step_sizes + alpha.unsqueeze(-1) * search_dir
    # Function value at initial alpha
    pot_new, pot_grad_new = pot_obj.eval(step_sizes_new, grad=True)
    # Iterate until decrease in function is above threshold
    ind_decr = (pot - pot_new < alpha * thresh) & (alpha > alpha_min)
    while ind_decr.any():
        # Decrease alpha and re-evaluate function
        alpha[ind_decr] *= tau
        step_sizes_new = step_sizes + alpha.unsqueeze(-1) * search_dir
        pot_new, pot_grad_new = pot_obj.eval(step_sizes_new, grad=True)
        ind_decr = (pot - pot_new < alpha * thresh) & (alpha > alpha_min)

    return step_sizes_new, pot_new, pot_grad_new, alpha
