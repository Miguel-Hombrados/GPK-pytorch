# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:50:45 2021

@author: mahom
"""
import torch
import gpytorch
from to_torch import to_torch
def predGPind(test_x,likelihood,model):
    
    test_x = to_torch(test_x)
    Ntest = torch.Tensor.size(test_x,0)
    # Make predictions
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        cov = predictions.covariance_matrix
        var = cov.diag().reshape(Ntest,-1)
    return mean,var