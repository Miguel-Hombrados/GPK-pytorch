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
    task_num = 24
    model.likelihood.task_noises = 0.5*torch.ones(task_num)                                                         # 0.5
    model.covar_module.kernels[0].base_kernel.outputscale = 5e-1*torch.ones(task_num)                                #0.5
    model.covar_module.kernels[0].base_kernel.lengthscale = 15*torch.ones(task_num,1,1) # probar 15                  #3
    model.covar_module.kernels[1].bias = 1e-3*torch.ones(task_num,1,1)                                              #1e-3

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        cov = predictions.covariance_matrix
        var = cov.diag().reshape(Ntest,-1)
    return mean,var