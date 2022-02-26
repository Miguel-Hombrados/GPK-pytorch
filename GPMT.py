# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:37:27 2022

@author: mahom
"""

import torch
import gpytorch
from to_torch import to_torch
from MTGPclasses import MultitaskGPModel
from epoch_tv import train_epoch,valid_epoch
from sklearn.model_selection import train_test_split
from fix_constraints import fix_constraints
from my_initialization import my_initialization
def GPMT(x,y,n_tasks,kernel_type,option_lv):
    
    x = to_torch(x)
    y = to_torch(y)
    
    num_iter = 200
    learning_rate = 0.01
    
    best_niter = {} 
    [train_x,val_x, train_y,val_y] = train_test_split(x,y, test_size=0.2, train_size=0.8, random_state=47, shuffle=True, stratify=None)
    data_train = (train_x,train_y)
    data_val = (val_x,val_y)
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = MultitaskGPModel(train_x, train_y, likelihood,n_tasks,kernel_type,option_lv)
    
    fix_constraints(model,likelihood,kernel_type,n_tasks,"mt")
    hypers = my_initialization(model,likelihood,kernel_type,n_tasks,"mt")
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters
    
    history = {'train_loss': [], 'valid_loss': []}

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for it in range(0,num_iter):
        optimizer.zero_grad()
        train_loss,output = train_epoch(model,data_train,mll,optimizer)
        [valid_loss,valid_error] = valid_epoch(model,likelihood,output,data_val,mll)
        
        
        train_loss = train_loss / data_train[0].size()[0]
        valid_loss = valid_loss / data_val[0].size()[0]
        
        print("Iter:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f}".format(it + 1,
                                                                         num_iter,
                                                                         train_loss,
                                                                         valid_loss,
                                                                          ))
        
        if it> 1  and valid_loss < torch.min(history['valid_loss']):
            min_valid_loss = valid_loss
            n_opt_niter = it + 1
            best_params = model.state_dict()
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

    return model, likelihood
    
    