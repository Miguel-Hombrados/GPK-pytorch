# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:05:02 2021

@author: mahom
"""


import torch
import gpytorch
from to_torch import to_torch
from fix_constraints import fix_constraints
from MTGPclasses import BatchIndependentMultitaskGPModel

from fix_parameter import fix_parameter
from my_initialization import my_initialization

def GPind(train_x,train_y,n_tasks,kernel_type):
    
    train_x = to_torch(train_x)
    train_y = to_torch(train_y)
    

    training_iterations = 60
    learning_rate = 0.1

    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood,n_tasks,kernel_type)
    
    
    hypers = my_initialization(kernel_type,n_tasks)
    model.initialize(**hypers)
    fix_constraints(model,likelihood,kernel_type,n_tasks)

    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    
    # Fix redundant parameters
    [model,new_parameters] = fix_parameter(model,kernel_type)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(new_parameters, lr=learning_rate)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    
    return model, likelihood