#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:16:15 2022

@author: Miguel A Hombrados
"""
import torch
def my_initialization(kernel_type,task_num):
    
    if kernel_type == "linear":
        hypers = {
            'likelihood.noise': torch.tensor(1.),
            'likelihood.task_noises': 0.1*torch.ones(task_num),
            'covar_module.outputscale': 0.1*torch.ones(task_num),
            'covar_module.base_kernel.variance': 1*torch.ones(task_num,1,1)   # Scale Kernel
    }
        
        
    if kernel_type == "rbf":
        hypers = {
            'likelihood.noise': torch.tensor(1.),
            'likelihood.task_noises': 0.1*torch.ones(task_num),
            'covar_module.base_kernel.lengthscale': 0.1*torch.ones(task_num,1,1),
            'covar_module.outputscale': 1*torch.ones(task_num)
    }
        
    # if kernel_type == "matern":
    #     hypers = {
    #         'likelihood.noise': torch.tensor(1.),
    #         'likelihood.task_noises': 0.1*torch.ones(task_num),
    #         'covar_module.covar_module.base_kernel.lengthscale': 0.1*torch.ones(task_num,1,1),
    #         'covar_module.outputscale': 1*torch.ones(task_num)
    # }
        
    return hypers
    
    