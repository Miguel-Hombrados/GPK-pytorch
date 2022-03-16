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
from MTGPclasses import ExactGPModel_single
from sklearn.model_selection import train_test_split
from fix_parameter import fix_parameter
from my_initialization import my_initialization
from epoch_tv import train_epoch,valid_epoch


def GPind_ori_aux2(x,y,n_tasks,kernel_type,opt_parameters):

      x= to_torch(x)
      y = to_torch(y)
      n_tasks = y.size(1)

      learning_rate = opt_parameters['lr1']
      learning_rate2 = opt_parameters['lr2'] 
      n_restarts = opt_parameters['n_restarts'] 
      num_iter = opt_parameters['num_iter'] 
   
      
      train_x = x
      train_y = y


      Results = {}
      MODELS = {}
      LIKELIHOODS = {}
      for rest in range(0,n_restarts):
          print()
          print("RESTART{}/{}:".format(rest+1,n_restarts))
          history = {}
          best_params = {}
          models = {}
          likelihoods = {}
          for task in range(0,n_tasks):
              n_opt_iter = 0
              min_train_loss = np.Inf
              train_x_t = train_x
              train_y_t = train_y[:,task].ravel()
              data_train_t = (train_x_t,train_y_t)
              likelihood = gpytorch.likelihoods.GaussianLikelihood()
              model = ExactGPModel_single(train_x, train_y_t, likelihood, kernel_type)# FUNCIONARIA SIN train_x, como en la gaussiana?
             
                
              fix_constraints(model,likelihood,kernel_type,1,"gpi_ori")
              hypers = my_initialization(model,likelihood,kernel_type,1,"gpi_ori")
              # Fix redundant parameters
              [model,new_parameters] = fix_parameter(model,kernel_type,"gpi_ori")

              optimizer = torch.optim.Adam(new_parameters , lr=learning_rate)  # Includes GaussianLikelihood parameters

              history_t = {'train_loss': [], 'n_opt_iter': []}
          
              
              # "Loss" for GPs - the marginal log likelihood
              mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                  
              for it in range(0,num_iter):
                  optimizer.zero_grad()
                  train_loss,output = train_epoch(model,data_train_t,mll,optimizer)
                  #valid_loss,validation_error,vp_error = valid_epoch(model,likelihood,output,None,mll)
              
                  train_loss = train_loss / data_train_t[0].size()[0]
                  
                  print("Task: {} Iter:{}/{} AVG Training Loss:{:.3f} ".format(task+1,it + 1,
                                                                                   num_iter,
                                                                                   train_loss,           
                                                                                   ))
                  if it ==0:
                      best_params_k = model.state_dict()
                  if it >int(0.8*num_iter):
                      optimizer.param_groups[0]['lr'] =  learning_rate2 
                  
                  if it> 1  and train_loss < np.min(history_t['train_loss']):
                      min_train_loss = train_loss
                      n_opt_iter = it + 1
                      best_params_k = model.state_dict()
                  
                  history_t['train_loss'].append(train_loss)
              history_t ['n_opt_iter'] = n_opt_iter
              history_t ['min_train_loss'] = min_train_loss
              history['task{}'.format(task+1)] = history_t  
              best_params['task{}'.format(task+1)] = best_params_k
              models['task{}'.format(task+1)] = model
              likelihoods['task{}'.format(task+1)] = likelihood
          Results['restart{}'.format(rest+1)] = {'history':history,'best_params':best_params,'models':models,'likelihoods':likelihoods,
                                               'configuration':{' lr': learning_rate,'max_iter':num_iter,'num_restarts':n_restarts,
                                                        'ratioTRTST':"all training"}}
          MODELS['restart{}'.format(rest+1)] = models
          LIKELIHOODS['restart{}'.format(rest+1)] = likelihoods
          
          Opt_model = {}
          Opt_likelihood = {}
      print()
      print()
      for task in range(0,n_tasks):  
          Opt_loss = torch.inf
          for rest in range(0,n_restarts):
              min_train_t_r = Results['restart{}'.format(rest+1)]['history']['task{}'.format(task+1)]['min_train_loss']
              if min_train_t_r<Opt_loss:
                  Opt_loss = min_train_t_r
                  Opt_model['task{}'.format(task+1)]= MODELS['restart{}'.format(rest+1)]['task{}'.format(task+1)]
                  Opt_likelihood['task{}'.format(task+1)]= LIKELIHOODS['restart{}'.format(rest+1)]['task{}'.format(task+1)]
              print('Task:{}'.format(task+1)+' Restart:{}'.format(rest)+' Min train loss:{:.5f}'.format(Opt_loss))

      return MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood