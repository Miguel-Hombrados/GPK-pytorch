# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:37:27 2022

@author: mahom
"""

import torch
import gpytorch
from to_torch import to_torch
from MTGPclasses import MultitaskGPModel
def GPMT(train_x,train_y,n_tasks,kernel_type,option_lv):
    
    train_x = to_torch(train_x)
    train_y = to_torch(train_y)
    
    training_iterations = 200
    learning_rate = 0.01
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = MultitaskGPModel(train_x, train_y, likelihood,n_tasks,kernel_type,option_lv)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

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
    
    