# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 02:28:59 2022

@author: mahom
"""


import numpy as np
import torch
import gpytorch
from to_torch import to_torch
from fix_constraints import fix_constraints
from sklearn.model_selection import train_test_split

from fix_parameter import fix_parameter
from my_initialization import my_initialization
from epoch_tv import train_epoch,valid_epoch
from MTGPclasses import ApproxGPModel_Lap_single

def GPind_lap(x,y,n_tasks,kernel_type):

    
    x= to_torch(x)
    y = to_torch(y)

    num_iter = 200
    learning_rate = 0.01
    
    [train_x,val_x, train_y,val_y] = train_test_split(x,y, test_size=0.25, train_size=0.75, random_state=47, shuffle=True, stratify=None)
    

    history = {}
    best_params = {}
    models = {}
    likelihoods = {}
    for task in range(0,n_tasks):
        n_opt_iter = 0
        min_valid_loss = np.Inf
        min_train_loss = np.Inf
        train_x_t = train_x
        train_y_t = train_y[:,task].ravel()
        val_x_t = val_x
        val_y_t = val_y[:,task].ravel()
        data_train_t = (train_x_t,train_y_t)
        data_val_t = (val_x_t,val_y_t)
        
        likelihood = gpytorch.likelihoods.LaplaceLikelihood()
        model = ApproxGPModel_Lap_single(train_x_t,kernel_type) # FUNCIONARIA SIN train_x, como en la gaussiana?
        
        fix_constraints(model,likelihood,kernel_type,1,"gpi_lap_var")
        hypers = my_initialization(model,likelihood,kernel_type,1,"gpi_lap_var")
            
        # Fix redundant parameters
        #[model,new_parameters] = fix_parameter(model,kernel_type)
        # Use the adam optimizer
        #new_parameters = model.parameters()
        
        
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=learning_rate)  # Includes GaussianLikelihood parameters
        
        history_t = {'train_loss': [], 'valid_loss': [], 'n_opt_iter': [], 'min_valid_loss': []}
    
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y_t.numel())
            
        for it in range(0,num_iter):
            train_loss,output = train_epoch(model,data_train_t,mll,optimizer)
            optimizer.zero_grad()
            valid_loss,_ = valid_epoch(model,likelihood,output,data_val_t,mll)
        
            train_loss = train_loss / data_train_t[0].size()[0]
            valid_loss = valid_loss / data_val_t[0].size()[0]
            
            print("Iter:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f}".format(it + 1,
                                                                             num_iter,
                                                                             train_loss,
                                                                             valid_loss,
                                                                              ))
            if it> 1  and train_loss < np.min(history_t['valid_loss']):
           # if it> 1  and valid_loss < np.min(history_t['valid_loss']):
                min_valid_loss = valid_loss
                #min_train_loss = train_loss
                n_opt_iter = it + 1
                best_params_k = model.state_dict()
            
            history_t['train_loss'].append(train_loss)
            history_t['valid_loss'].append(valid_loss)
        history_t ['n_opt_iter'] = n_opt_iter
        history_t ['min_valid_loss'] = min_valid_loss
        history['task{}'.format(task+1)] = history_t  
        best_params['task{}'.format(task+1)] = best_params_k
        models['task{}'.format(task+1)] = model
        likelihoods['task{}'.format(task+1)] = likelihood

    return models,likelihoods,history,best_params
