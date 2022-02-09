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
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from fix_parameter import fix_parameter
from my_initialization import my_initialization
from epoch_tv import train_epoch,valid_epoch
def GPind(train_x,train_y,n_tasks,kernel_type):
    validation = True
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    if validation == True:
        k=5
        splits=KFold(n_splits=k,shuffle=True,random_state=42)
    
    
    train_x= to_torch(train_x)
    train_y = to_torch(train_y)
    #dataset = TensorDataset(train_x, train_y)
    #dataset = DataLoader((train_x,train_y))

    num_iter = 21
    learning_rate = 0.1

    foldperf = {}
    best_models = {}
    best_niter = {}
    for fold, (train_idx,val_idx) in enumerate(splits.split(train_x)):

        print('Fold {}'.format(fold + 1))
        #device = torch.device("cpu")
        
        
        train_x_k = train_x[train_idx,:]
        train_y_k = train_y[train_idx,:]
        data_train = (train_x_k,train_y_k)
        valid_x_k = train_x[val_idx,:]
        valid_y_k = train_y[val_idx,:]
        data_val = (valid_x_k,valid_y_k)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
        model = BatchIndependentMultitaskGPModel(train_x_k,train_y_k, likelihood,n_tasks,kernel_type)
        
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
        
        for it in range(num_iter):
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
                opt_niter = it
                best_params = model.state_dict()
            
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
        foldperf['fold{}'.format(fold+1)] = history  
                                
        # for i in range(training_iterations):
        #     optimizer.zero_grad()
        #     output = model(train_x)
        #     loss = -mll(output, train_y)
        #     loss.backward()
        #     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        #     optimizer.step()
        
        best_models['fold{}'.format(fold+1)] = best_params
        best_niter['fold{}'.format(fold+1)] = opt_niter
    n_opt = np.median(list(best_niter.values()))
    for it in range(n_opt):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
        model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood,n_tasks,kernel_type) 
        fix_constraints(model,likelihood,kernel_type,n_tasks)
        hypers = my_initialization(model,kernel_type,n_tasks)
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        
        
        # Fix redundant parameters
        [model,new_parameters] = fix_parameter(model,kernel_type)
        # Use the adam optimizer
        optimizer = torch.optim.Adam(new_parameters, lr=learning_rate)  # Includes GaussianLikelihood parameters
        
    return model,likelihood#, history, foldperf, best_models


