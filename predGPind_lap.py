# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:50:45 2021

@author: mahom
"""
import torch
import gpytorch
from to_torch import to_torch
def predGPind_lap(test_x,likelihoods,models):
    
    test_x = to_torch(test_x)
    n_test = test_x.size(0)
    n_task = 24
    
    means = torch.zeros(n_test,n_task)
    scales = torch.zeros(n_test,n_task)
    
    for task in range(0,n_task):
        model_k = models['task{}'.format(task+1)]
        likelihood_k = likelihoods['task{}'.format(task+1)]

    # Make predictions
        model_k.eval()
        likelihood_k.eval()


        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood_k(model_k(test_x))
            mean_k = predictions.mean
            scale_k = predictions.scale
        means[:,task] = torch.mean(mean_k)
        scales[:,task] = torch.mean(scale_k)
        
    return means,scales