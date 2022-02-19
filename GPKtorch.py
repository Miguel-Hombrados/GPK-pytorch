#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:05:03 2021

@author: Miguel A Hombrados
"""

import torch
import gpytorch
from GPind import GPind
from GPind_ori import GPind_ori
from GPMT import GPMT
from to_torch import to_torch
def GPKtorch(train_x,train_y,n_tasks,kernel_type,option_lv):
    
    
    train_x = to_torch(train_x)
    train_y = to_torch(train_y)
    
    
    if option_lv == "ind":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPind(train_x,train_y,n_tasks,kernel_type)
        
    if option_lv == "ind_ori":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPind(train_x,train_y,n_tasks,kernel_type)
        
    if option_lv == "mt":
        [model,likelihood,n_opt_niter,min_valid_loss] = GPMT(train_x,train_y,n_tasks,kernel_type,option_lv)


    a = correcting_factor_cov(model,W,Yval,Xval)
    return model,likelihood,n_opt_niter,min_valid_loss
