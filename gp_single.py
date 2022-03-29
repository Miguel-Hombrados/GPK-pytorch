# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:29:24 2022

@author: mahom
"""
import numpy as np
import torch
import gpytorch
from to_torch import to_torch
from fix_constraints import fix_constraints
from MTGPclasses import ExactGPModel_single
from sklearn.model_selection import train_test_split
from fix_parameter import fix_parameter
from my_initialization import my_initialization
from random_initialization import random_initialization
from epoch_tv import train_epoch,valid_epoch


def gp_single(x,y,kernel_type,opt_parameters):
    
    
    learning_rate = opt_parameters['lr1']
    learning_rate2 = opt_parameters['lr2'] 
    n_restarts = opt_parameters['n_restarts'] 
    num_iter = opt_parameters['num_iter'] 
    
    trainsize = opt_parameters['trainsize']  
    valsize = 1.0 - trainsize
    
    if trainsize !=0:
        Ntrain  = x.size(0)
        ind_train = torch.Tensor(range(0,Ntrain)).reshape(-1,1)
        x = torch.cat((x,ind_train),dim=1)
        [train_x,val_x, train_y,val_y] = train_test_split(x,y, test_size=valsize, train_size=trainsize, random_state=47, shuffle=True, stratify=None)
        train_x = train_x[:,:-1]
        ind_val_t = val_x[:,-1]
        val_x = val_x[:,:-1]
    elif trainsize == 1:
         train_x = x
         train_y = y
    
    Results = {}
    MODELS = {}
    LIKELIHOODS = {}
    for rest in range(0,n_restarts):
        print()
        print("RESTART{}/{}:".format(rest+1,n_restarts))
        history_t = {}
        best_params_t = {}
        models = {}
        likelihoods = {}
        n_opt_iter_t = 0
        min_train_loss = np.Inf
        data_train_t = (train_x,train_y)
        n_batch = 1
        min_valid_loss = np.Inf
        min_train_loss = np.Inf
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel_single(train_x, train_y, likelihood, kernel_type)# FUNCIONARIA SIN train_x, como en la gaussiana?
        if trainsize !=0:
            data_val_t = (val_x,val_y)
        
        
        fix_constraints(model,likelihood,kernel_type,1,"gpk_sp")
        
        if rest == 0:
            my_initialization(model,likelihood,kernel_type,1,"gpk_sp")
        else: 
            random_initialization(model,likelihood,kernel_type,1,"gpk_sp")
        # Fix redundant parameters
        [model,new_parameters] = fix_parameter(model,kernel_type,"gpk_sp")
        model.mean_module.initialize(constant=0.)
    
        # Use the adam optimizer
        #new_parameters = model.parameters()
        
        
        optimizer = torch.optim.Adam(new_parameters , lr=learning_rate)  # Includes GaussianLikelihood parameters
    

        if trainsize !=0:
            history_t = {'train_loss': [], 'valid_loss': [], 'n_opt_iter': [], 'min_valid_loss': []}
        elif trainsize == 1:
            history_t = {'train_loss': [], 'n_opt_iter': []}
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
        for it in range(0,num_iter):
            optimizer.zero_grad()
            train_loss,output = train_epoch(model,data_train_t,mll,optimizer)
        
            train_loss = train_loss / data_train_t[0].size()[0]
            
            
            if trainsize == 1:
                print("Restart: {} Iter:{}/{} AVG Training Loss:{:.3f} ".format(rest+1,it + 1,
                                                                                 num_iter,
                                                                                 train_loss))
            elif trainsize !=0:
                valid_loss,validation_error,vp_error = valid_epoch(model,likelihood,output,data_val_t,mll)
                valid_loss = valid_loss / data_val_t[0].size()[0]
                print("Restart: {} Iter:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.6f}".format(rest+1,it + 1,
                                                                                   num_iter,
                                                                                   train_loss,
                                                                                   valid_loss))
            if it ==0:
                best_params_t = model.state_dict()
                if trainsize !=0:
                      min_valid_loss = valid_loss
                      min_validation_error = validation_error
                      min_validation_predictive_error = vp_error
            if it >int(0.8*num_iter):
                optimizer.param_groups[0]['lr'] =  learning_rate2 
            
            if it> 1  and train_loss < np.min(history_t['train_loss']):
            #if it> 1  and valid_loss < np.min(history_t['valid_loss']):
                min_valid_loss = valid_loss
                min_validation_error = validation_error
                min_validation_predictive_error = vp_error
                min_train_loss = train_loss
                n_opt_iter_t = it + 1
                best_params_t = model.state_dict()
            history_t['train_loss'].append(train_loss)
            if trainsize !=0:
                history_t['valid_loss'].append(valid_loss)
        history_t ['min_valid_loss'] = min_valid_loss
        history_t ['min_valid_error'] = min_validation_error
        history_t ['min_validation_predictive_error'] = min_validation_predictive_error            
        history_t ['min_train_loss'] = min_train_loss
        history_t ['n_opt_iter_t'] = n_opt_iter_t
        Results['restart{}'.format(rest+1)] = {'history_t':history_t,'best_params_t':best_params_t,'model':model,'likelihood':likelihood,
                                         'configuration':{' lr': learning_rate,'max_iter':num_iter,'num_restarts':n_restarts}}
        MODELS['restart{}'.format(rest+1)] = model
        LIKELIHOODS['restart{}'.format(rest+1)] = likelihood
    
    Opt_model = {}
    Opt_likelihood = {}
    print()
    print()
    Opt_loss = torch.inf
    for rest in range(0,n_restarts):
        min_train_t_r = Results['restart{}'.format(rest+1)]['history_t']['min_train_loss']
        min_val_error_t_r = Results['restart{}'.format(rest+1)]['history_t']['min_valid_error']
        min_val_predictive_error_t_r = Results['restart{}'.format(rest+1)]['history_t']['min_validation_predictive_error']
        if min_train_t_r<Opt_loss:
            Opt_loss = min_train_t_r
            Opt_model= MODELS['restart{}'.format(rest+1)]
            Opt_likelihood = LIKELIHOODS['restart{}'.format(rest+1)]
            min_val_error_t = min_val_error_t_r
            min_val_predictive_error_t = min_val_predictive_error_t_r
        print(' Restart:{}'.format(rest)+' Min train loss:{:.5f}'.format(Opt_loss))
    Validation_Errors_t  = min_val_error_t
    Validation_Predictive_Errors_t =     min_val_predictive_error_t
    return MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood,Validation_Errors_t,Validation_Predictive_Errors_t,ind_val_t