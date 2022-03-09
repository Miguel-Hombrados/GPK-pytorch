# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:50:45 2021

@author: mahom
"""
import torch
import gpytorch
from to_torch import to_torch
from opt_example import opt_example
def predGPind_ori(test_x,likelihoods,models):
    
    test_x = to_torch(test_x)
    n_test = test_x.size(0)
    n_task = len(models)
    
    means = torch.zeros(n_test,n_task)
    variances = torch.zeros(n_test,n_task)
    
    
    #opt_example(models,likelihoods)
    for task in range(0,n_task):
        model_k = models['task{}'.format(task+1)]
        likelihood_k = likelihoods['task{}'.format(task+1)]


    
    # Make predictions
        model_k.eval()
        likelihood_k.eval()


        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood_k(model_k(test_x))
            mean_k = predictions.mean
            var_k = predictions.variance
        means[:,task] = mean_k
        variances[:,task] = var_k
        
    return means,variances
