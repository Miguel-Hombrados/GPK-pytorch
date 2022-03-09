# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:18:44 2022

@author: mahom
"""
import numpy as np
import torch
from to_torch import to_torch
from DeStandarizeData import DeStandarizeData
def correcting_factor_cov(model,W,Yval,Xval,option_lv,scalerY_K):
    # W: T x K
    # Yval: Nval x T
    # Xval  Nval x F
    # Kh : Nval x T  times  N x T
    
    Nval = Yval.size(0)
    n_task = Yval.size(1)
    yval = Yval.reshape(-1,1)
    Wp = to_torch(np.kron(torch.eye(Nval),W))
    if option_lv == "ind" or option_lv == "mt":
        SigmaH = model.likelihood(model.forward(Xval)).covariance_matrix
    if option_lv == "gp_ind_ori":
        SigmaH_list = []
        for task in range(0,n_task):
            mk = model['task{}'.format(task+1)]
            SigmaH_k = mk.likelihood(mk.forward(Xval)).covariance_matrix
            SigmaH_list.append(SigmaH_k)
        SigmaH = torch.block_diag(*SigmaH_list)

    Kyi = torch.inverse(Wp@SigmaH@Wp.T+1e-6*torch.eye(24*Nval))
    a = (yval.T@Kyi@yval)/(Nval*24)
    
    return a
    
    
    