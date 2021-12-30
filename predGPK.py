# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:50:45 2021

@author: mahom
"""
import torch
import gpytorch

def predGPK(test_x,likelihood,model,Wtrain_s,option_k,**kwargs):
    
    
    Wtrain_s= torch.from_numpy(Wtrain_s)
    F = Wtrain_s.size(dim=0)
    K = Wtrain_s.size(dim=1)
    Nt = test_x.size(dim=0)
    
    Stds_train_load = kwargs.get('Stds_train_load', None)
    if(Stds_train_load is not None):
        Stds_train_load = torch.from_numpy(Stds_train_load).T
        Wtrain = Stds_train_load.repeat(K,1)*Wtrain_s
    if(Stds_train_load is None):
        Wtrain = Wtrain_s
    
    
    if option_k == "ind":
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(test_x))
            meanK = predictions.mean
            stdK = predictions.std
                 
        mean = Wtrain@meanK.T                    # F x N
        std = torch.zeros(F,F,Nt)
        for nt in range(0,Nt):    
            std[:,:,nt] = Wtrain@torch.diag(stdK[:,nt])@Wtrain.T   # F x F x N 
    #if option_k == "mt":   
    return mean,std