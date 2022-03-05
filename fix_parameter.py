# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 09:20:44 2022

@author: mahom
"""
import torch
def fix_parameter(model,kernel_type,method):
    
    
    
    if method == "gpi": 
        shared_noise_variance = 0  # 1e-4
        task_num = model.likelihood.num_tasks
        if kernel_type == "linear":
            model.covar_module.kernels[0].base_kernel.variance = torch.ones(task_num,1,1) # If selected "outputscale" the tensor should be 1d tensor of length task_num 
            model.likelihood.noise = shared_noise_variance
            new_parameters = set(list(model.parameters()))-{model.likelihood.raw_noise}-{model.covar_module.kernels[0].base_kernel.raw_variance}
            
        if kernel_type == "rbf":
            model.likelihood.noise = shared_noise_variance
            new_parameters = set(list(model.parameters()))-{model.likelihood.raw_noise}-{model.mean_module.constant}
        
        
        if kernel_type == "matern":
            model.likelihood.noise = shared_noise_variance
            new_parameters = set(list(model.parameters()))-{model.likelihood.raw_noise}
            #new_parameters = set(list(model.parameters()))
    if method == "gpi_ori": 
            if kernel_type == "rbf":
                new_parameters = set(list(model.parameters()))-{model.mean_module.constant}
    return model,new_parameters
           
        