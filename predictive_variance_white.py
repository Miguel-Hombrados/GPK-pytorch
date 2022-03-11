# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:30:34 2022

@author: mahom
"""
import torch
import scipy as sp
def predictive_variance_white(Variances,W,Var_ErrorNMF,VarNMF):
    
    # Variacnces: Ntest x K
    # W: F x K
    # StdNMF: F x 1
    # Var_ErrorNMF: F
    
    Ntest = Variances.size(0)
    Variances24 = torch.zeros(Ntest,24)
    SigInv = torch.inverse(W.T@W)
    A = torch.tensor(sp.linalg.sqrtm(SigInv)).float()
    W = W.float()
    Variances = Variances.float()
    VarNMF = VarNMF.ravel()
    
    for ss in range(0,Ntest):
       var_un = torch.diag(W@A@torch.diag(Variances[ss,:])@A.T@W.T) + Var_ErrorNMF
       Variances24[ss,:] = var_un*VarNMF
    return Variances24
    
    
    