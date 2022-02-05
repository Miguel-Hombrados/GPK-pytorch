#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:55:12 2022

@author: Miguel A Hombrados
"""

import gpytorch

def fix_constraints(model,likelihood,kernel_type,num_task):
    
    if kernel_type == "linear":
        likelihood.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-3,2))
        likelihood.register_constraint("raw_task_noises", gpytorch.constraints.Interval(1e-3,2))
       
        model.covar_module.kernels[0].base_kernel.register_constraint("raw_variance", gpytorch.constraints.Interval(1e-3,2))
        model.covar_module.kernels[0].register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-3,2))
        model.covar_module.kernels[1].register_constraint("raw_bias", gpytorch.constraints.Interval(1e-3,2))
    if kernel_type == "rbf":
        likelihood.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-4,1))
        likelihood.register_constraint("raw_task_noises", gpytorch.constraints.Interval(1e-1,2))
       
        model.covar_module.kernels[0].base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1.5,50))
        model.covar_module.kernels[0].register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-3,1))
        model.covar_module.kernels[1].register_constraint("raw_bias", gpytorch.constraints.Interval(1e-6,1))
## Changing the constraint after the module has been created

#print(f'Noise constraint: {likelihood.noise_covar.raw_noise_constraint}')
