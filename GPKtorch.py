#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:05:03 2021

@author: Miguel A Hombrados
"""

import torch
import gpytorch
from GPindK import GPindK

def GPKtorch(train_x,train_y,n_tasks,Wtrain_s,option_lv):
    
    
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    
    training_iterations = 50
    
   
    
    if option_lv == "ind":
        [model,likelihood] = GPindK(train_x,train_y,n_tasks)
        
    if option_lv == "mt":
