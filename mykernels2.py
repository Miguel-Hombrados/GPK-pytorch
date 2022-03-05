# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:01:21 2022

@author: mahom
"""
# import positivity constraint
import torch
import gpytorch
from gpytorch.constraints import Positive


class bias(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = False

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        n1 = x1.size(0)
        n2 = x2.size(0)
        O = torch.ones(n1,n2)   
        diff0 = O*params['bias']
        diff0.where(diff0 == 0, torch.as_tensor(1e-20))
        
        return diff0

