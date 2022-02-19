# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:18:44 2022

@author: mahom
"""
import torch
def correcting_factor_cov(model,W,Yval,Xval):
    # W: T x K
    # Yval: T x Nval
    # Xval  F x Nval
    # K : Nval x T  times  N x T
    Nval = Yval.size(1)
    n_task = Yval.size(0)
    SigmaH = model.likelihood(model.forward(Xval)).covariance_matrix
    
    yval = Yval.reshpe(-1,1)
    Wp = torch.kron(torch.eye(Nval),W)
    Kyi = torch.inverse(Wp@SigmaH@Wp.T+1e-6*torch.eye(n_task*Nval))
    a = (yval.T@Kyi@yval)/(Nval*n_task)
    
    return a
    
    
    