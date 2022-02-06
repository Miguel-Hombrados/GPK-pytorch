#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:16:15 2022

@author: Miguel A Hombrados
"""
import torch
def my_initialization(model,kernel_type,task_num):
    
    if kernel_type == "linear":

         #model.likelihood.noise = torch.tensor(0.1)
         model.likelihood.task_noises = 1e-1*torch.ones(task_num)                                   
         model.covar_module.kernels[0].base_kernel.outputscale = 0.1*torch.ones(task_num)         
         model.covar_module.kernels[0].base_kernel.variance = 1*torch.ones(task_num,1,1)            
         model.covar_module.kernels[1].bias = 1*torch.ones(task_num,1,1)            
        
        


    if kernel_type == "rbf":

        #model.likelihood.noise = torch.tensor(0.01)
        model.likelihood.task_noises = 0.5*torch.ones(task_num)                                                         # 0.5
        model.covar_module.kernels[0].base_kernel.outputscale = 1e-5*torch.ones(task_num)                                #0.5
        model.covar_module.kernels[0].base_kernel.lengthscale = 10*torch.ones(task_num,1,1) # probar 15                  #3
        model.covar_module.kernels[1].bias = 1e-7*torch.ones(task_num,1,1)                                              #1e-3
    

    # if kernel_type == "matern":

    