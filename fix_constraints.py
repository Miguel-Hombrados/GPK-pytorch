#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:55:12 2022

@author: Miguel A Hombrados
"""

import gpytorch

def fix_constraints(model,likelihood,kernel_type,num_task):
    
    if kernel_type == "linear":
        likelihood.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-3,10))
        likelihood.register_constraint("raw_task_noises", gpytorch.constraints.Interval(1e-3,10))
       
        model.covar_module.base_kernel.register_constraint("raw_variance", gpytorch.constraints.Interval(1e-3,10))
        model.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-3,10))
    if kernel_type == "rbf":
        likelihood.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-3,10))
        likelihood.register_constraint("raw_task_noises", gpytorch.constraints.Interval(1e-3,10))
       
        model.covar_module.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1e-3,10))
        model.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-3,10))
    
## Changing the constraint after the module has been created

#print(f'Noise constraint: {likelihood.noise_covar.raw_noise_constraint}')
