# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:50:45 2021

@author: mahom
"""
import torch
import gpytorch
from to_torch import to_torch
from opt_example import opt_example
def predGPK_sp(Ntest,K, models,likelihoods, IDRegressorTypes, IDRegressorTypes_test,X_test,Labels,Labels_test,Indices_test):
    
    n_task = len(models)
    
    means = torch.zeros(Ntest,K)
    variances = torch.zeros(Ntest,K)

    #opt_example(models,likelihoods)
    for task in range(0,n_task):

        indices_test_samples_r = Indices_test['task{}'.format(task+1)]
        test_x = X_test['task{}'.format(task+1)]
        test_x = to_torch(test_x)

        
        model_k = models['task{}'.format(task+1)]
        likelihood_k = likelihoods['task{}'.format(task+1)]
    
    # Make predictions
        model_k.eval()
        likelihood_k.eval()


        with torch.no_grad(): #, gpytorch.settings.fast_pred_var():
            predictions = likelihood_k(model_k(test_x))
            mean_k = predictions.mean
            var_k = predictions.variance
        means[indices_test_samples_r,:] = mean_k.reshape(-1,1)
        variances[indices_test_samples_r,:] = var_k.reshape(-1,1)
        
        
        
    return means,variances
