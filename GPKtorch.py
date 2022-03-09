#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:05:03 2021

@author: Miguel A Hombrados
"""

import torch
from GPind_ori import GPind_ori
from GPind import GPind
from GPMT import GPMT
from to_torch import to_torch
from sklearn.model_selection import train_test_split
def GPKtorch(x,y,W,n_tasks,kernel_type,option_lv,opt_parameters):
    
    trainsize = opt_parameters['trainsize']  
    valsize = 1 - trainsize
    x = to_torch(x)
    y = to_torch(y)
    
     
    x = torch.cat((x,torch.Tensor(range(0,x.size(0))).reshape(-1,1)),dim=1)
    [train_x,val_x, train_y,val_y] = train_test_split(x,y, test_size=valsize, train_size=trainsize, random_state=47, shuffle=True, stratify=None)
    train_x = train_x[:,0:-1]
    val_x  = val_x[:,0:-1]
    ind_val = val_x[:,-1]
    
    if option_lv == "gp_ind_ori":
        [MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood] = GPind_ori(train_x,train_y,n_tasks,kernel_type,opt_parameters)
    if option_lv == "gp_ind":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPind(train_x,train_y,n_tasks,kernel_type,opt_parameters)       
    if option_lv == "gpmt":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPMT(train_x,train_y,n_tasks,kernel_type,option_lv,opt_parameters)

    return MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood,ind_val
