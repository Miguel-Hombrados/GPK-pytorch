# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:18:44 2022

@author: mahom
"""
import torch
def correcting_factor_cov(model,W,Yval,Xval,option_lv):
    # W: T x K
    # Yval: Nval x T
    # Xval  Nval x F
    # Kh : Nval x T  times  N x T
    Nval = Yval.size(1)
    n_task = Yval.size(0)
    yval = Yval.reshpe(-1,1)
    Wp = torch.kron(torch.eye(Nval),W)
    if option_lv == "ind" or "mt":
        SigmaH = model.likelihood(model.forward(Xval)).covariance_matrix
    if option_lv == "ind_ori":
        SigmaH_list = []
        for task in range(0,n_task):
            mk = model[task]
            SigmaH_k = mk.likelihood(mk.forward(Xval)).covariance_matrix
            SigmaH_list.append(SigmaH_k)
        SigmaH = torch.block_diag(SigmaH_list)

    
    Kyi = torch.inverse(Wp@SigmaH@Wp.T+1e-6*torch.eye(n_task*Nval))
    a = (yval.T@Kyi@yval)/(Nval*n_task)
    
    return a
    
    
    