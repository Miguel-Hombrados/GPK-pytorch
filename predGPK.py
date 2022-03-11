# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:50:45 2021

@author: mahom
"""


import torch
from to_torch import to_torch
def predGPK(mean_k,var_covar_k,Wtrain_s,**kwargs):
    
    
    mean_k = to_torch(mean_k)
    var_covar_k = to_torch(var_covar_k)
    Wtrain_s = to_torch(Wtrain_s)
    K = Wtrain_s.size(dim=1)
    F = Wtrain_s.size(dim=0)
    Nt = mean_k.size(dim=0)
    
    #Stds_train_load = kwargs["Stds_train_load"]
    Stds_train_load = kwargs.get("Stds_train_load", None)
    if(Stds_train_load is not None):
        Stds_train_load = to_torch(Stds_train_load)
        Wtrain = Stds_train_load.repeat(1,K)*Wtrain_s
    if(Stds_train_load is None):
        Wtrain = Wtrain_s

    mean = mean_k@Wtrain.T                  # F x N

    variances = torch.zeros(Nt,F)
    if var_covar_k.shape == torch.Size([Nt,K]):     # N x K
        for nt in range(0,Nt):    
            variances[nt,:] = torch.diag(Wtrain@torch.diag(var_covar_k[nt,:])@Wtrain.T)
    elif var_covar_k.shape == torch.Size([Nt,K,K]):   # N x K x K
        for nt in range(0,Nt):    
            variances[nt,:] = torch.diag(Wtrain@torch.diag(torch.diag(var_covar_k[nt,:,:]))@Wtrain.T)
    else:
        print("Input error in 'predGPK. Wrong shape'")

    return mean,variances