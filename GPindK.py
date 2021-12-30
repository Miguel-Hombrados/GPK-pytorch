# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:05:02 2021

@author: mahom
"""


import torch
import gpytorch
from MTGPclasses import BatchIndependentMultitaskGPModel
def GPindK(train_x,train_y,n_tasks,kernel_type):
    
    
    train_x= torch.from_numpy(train_x)
    train_y= torch.from_numpy(train_y)
    
    training_iterations = 50
    
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood,n_tasks,kernel_type)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

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