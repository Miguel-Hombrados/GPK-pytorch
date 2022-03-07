#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:05:03 2021

@author: Miguel A Hombrados
"""

import torch
import gpytorch
from GPind_ori import GPind_ori
from GPind import GPind
from GPMT import GPMT
from to_torch import to_torch
from sklearn.model_selection import train_test_split
from correcting_factor_cov import correcting_factor_cov
def GPKtorch(x,y,W,n_tasks,kernel_type,option_lv):
    
    valtsize = 0.1
    trainsize = 0.9
    train_x = to_torch(x)
    train_y = to_torch(y)
    
    [train_x,val_x, train_y,val_y] = train_test_split(x,y, test_size=valtsize, train_size=trainsize, random_state=47, shuffle=True, stratify=None)
      
    
    if option_lv == "ind":
        [MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood] = GPind_ori(train_x,train_y,n_tasks,kernel_type)
        MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood
    if option_lv == "ind_ori":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPind(train_x,train_y,n_tasks,kernel_type)
        
    if option_lv == "mt":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPMT(train_x,train_y,n_tasks,kernel_type,option_lv)


    a = correcting_factor_cov(model,W,val_y,val_x)
    return Opt_model,Opt_likelihood
