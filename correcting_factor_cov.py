# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:18:44 2022

@author: mahom
"""
import numpy as np
import torch
from to_torch import to_torch
def correcting_factor_cov(model,Ws,Yval,Xval,option_lv,scalerY_K,Var_ErrorNMF,Std_nmf):
    # W: T x K
    # Yval: Nval x T
    # Xval  Nval x F
    # Kh : Nval x T  times  N x T
    
    Nval = Yval.size(0)
    n_task = len(model)
    yval = Yval.reshape(-1,1)
    W = Std_nmf*Ws
    #Wp = to_torch(np.kron(torch.eye(Nval),W))
    Wp = to_torch(np.kron(W,torch.eye(Nval)))
    VarK = scalerY_K.var_
    StdK = torch.sqrt(torch.Tensor(scalerY_K.var_)).reshape(-1,1)
    meanK = torch.Tensor(scalerY_K.mean_).reshape(-1,1)
    if option_lv == "ind" or option_lv == "mt":
        SigmaH = model.likelihood(model.forward(Xval)).covariance_matrix
    if option_lv == "gp_ind_ori":
        SigmaH_list = []
        meansH_list = []
        for task in range(0,n_task):
            mk = model['task{}'.format(task+1)]
            SigmaH_k = VarK[task]*(mk.likelihood(mk.forward(Xval)).covariance_matrix) # Varianzas solo en lugar de Cov? Rearrange de manera que se puedan operar diferente.
            SigmaH_list.append(SigmaH_k)
            meansH_list.append(model['task{}'.format(task+1)].mean_module.constant)
        SigmaH = torch.block_diag(*SigmaH_list)
        
    meansH = torch.Tensor(meansH_list).reshape(-1,1)
    Mstd = meansH.repeat(1,Nval)
    M = (W@(Mstd*StdK+meanK)).T  # N x 24
    m = M.reshape(-1,1)
    yval_cent = yval - m
 # Las matrices deberian ser positivas definidas.
    #Kyi = torch.inverse(Wp@SigmaH@Wp.T+1e-3*torch.eye(24*Nval)) # Noise estimation en lugar de eye
    Kyi = torch.inverse(Wp@SigmaH@Wp.T+torch.diag(Var_ErrorNMF.repeat(Nval))) 
    a = (yval_cent.T@Kyi@yval_cent)/(Nval*24)
    
    return a
    
    
    