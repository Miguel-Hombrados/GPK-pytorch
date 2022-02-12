# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:05:02 2021

@author: mahom
"""

import numpy as np
import torch
import gpytorch
from to_torch import to_torch
from fix_constraints import fix_constraints
from MTGPclasses import BatchIndependentMultitaskGPModel
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from fix_parameter import fix_parameter
from my_initialization import my_initialization
from epoch_tv import train_epoch,valid_epoch


def GPind(x,y,n_tasks,kernel_type):

    
    x= to_torch(x)
    y = to_torch(y)
    #dataset = TensorDataset(train_x, train_y)
    #dataset = DataLoader((train_x,train_y))

    num_iter = 50
    learning_rate = 0.05


    best_niter = {}
    
    
    [train_x,val_x, train_y,val_y] = train_test_split(x,y, test_size=0.2, train_size=0.8, random_state=47, shuffle=True, stratify=None)
    data_train = (train_x,train_y)
    data_val = (val_x,val_y)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = BatchIndependentMultitaskGPModel(train_x,train_y, likelihood,n_tasks,kernel_type)
    
    fix_constraints(model,likelihood,kernel_type,n_tasks)
    hypers = my_initialization(model,kernel_type,n_tasks)
        #model.initialize(**hypers)
         
        
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
        
    
    # Fix redundant parameters
    [model,new_parameters] = fix_parameter(model,kernel_type)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(new_parameters, lr=learning_rate)  # Includes GaussianLikelihood parameters
    
    history = {'train_loss': [], 'valid_loss': []}

    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
    for it in range(0,num_iter):
        train_loss,output = train_epoch(model,data_train,mll,optimizer)
        optimizer.zero_grad()
        valid_loss = valid_epoch(model,likelihood,output,data_val,mll)
    
        train_loss = train_loss / data_train[0].size()[0]
        valid_loss = valid_loss / data_val[0].size()[0]
        
        print("Iter:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f}".format(it + 1,
                                                                         num_iter,
                                                                         train_loss,
                                                                         valid_loss,
                                                                          ))
        
        if it> 1  and valid_loss < np.min(history['valid_loss']):
            min_valid_loss = valid_loss
            n_opt_niter = it + 1
            best_params = model.state_dict()
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

 
    return model,likelihood,n_opt_niter,min_valid_loss


