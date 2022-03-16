#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:55:12 2022

@author: Miguel A Hombrados
"""

import gpytorch

def fix_constraints(model,likelihood,kernel_type,num_task,method):
    
    
    if method == "gpi":
        if kernel_type == "linear":
            likelihood.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-6,2))
            likelihood.register_constraint("raw_task_noises", gpytorch.constraints.Interval(1e-2,2))
           
            model.covar_module.kernels[0].base_kernel.register_constraint("raw_variance", gpytorch.constraints.Interval(1e-3,50))
            model.covar_module.kernels[0].register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-3,50))
            model.covar_module.kernels[1].register_constraint("raw_bias", gpytorch.constraints.Interval(1e-7,1))
        if kernel_type == "rbf":
            likelihood.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-8,5e-1))                                         #(1e-4,1)
            likelihood.register_constraint("raw_task_noises", gpytorch.constraints.Interval(1e-8, 1))                                   #(1e-1,2)
            model.covar_module.kernels[0].register_constraint("raw_outputscale", gpytorch.constraints.Interval(0.01,100))   
            #model.covar_module.kernels[0].base_kernel.register_constraint("raw_outputscale", gpytorch.constraints.Interval(1,100)) 
            model.covar_module.kernels[0].base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1,100))    #(1.5,50)               #(1e-3,1)
            model.covar_module.kernels[1].register_constraint("raw_bias", gpytorch.constraints.Interval(1e-8, 1))                       #(1e-6,1)
    elif method == "gpi_ori":
        if kernel_type == "rbf":  
            model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-4, 10))                                                            # 0.5    
            model.covar_module.kernels[0].register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-5, 200))
            model.covar_module.kernels[0].base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1,200))                  #3
            model.covar_module.kernels[1].register_constraint("raw_bias", gpytorch.constraints.Interval(1e-7, 1))   
                
                #model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-4, 10))                                                            # 0.5    
                #model.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-5, 50))
                #model.covar_module.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1,100))    
                

               
                