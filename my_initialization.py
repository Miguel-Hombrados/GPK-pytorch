#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:16:15 2022

@author: Miguel A Hombrados
"""
import torch
def my_initialization(model,likelihood,kernel_type,task_num,method):
    
    if method == "gpi": 
        if kernel_type == "linear":
             #model.likelihood.noise = torch.tensor(0.1)
             model.likelihood.task_noises = 1e-1*torch.ones(task_num,requires_grad=True)                               
             model.covar_module.kernels[0].base_kernel.outputscale = 10*torch.ones(task_num,requires_grad=True)             
             model.covar_module.kernels[0].base_kernel.variance = 10*torch.ones(task_num,1,1,requires_grad=True)             
             model.covar_module.kernels[1].bias = 1e-5*torch.ones(task_num,1,1,requires_grad=True)         
        if kernel_type == "rbf":
            #model.likelihood.noise = torch.tensor(0.01)
            model.likelihood.task_noises = 0.1*torch.ones(task_num,requires_grad=True)                                                        # 0.5
            #model.covar_module.kernels[0].base_kernel.outputscale = 1000*torch.ones(task_num,requires_grad=True)      #0.5
            model.covar_module.kernels[0].base_kernel.lengthscale = 91*torch.ones(task_num,1,1,requires_grad=True) # probar 96.5                #3
            model.covar_module.kernels[1].bias = 1e-6*torch.ones(task_num,requires_grad=True)   
            
            model.covar_module.kernels[0].outputscale =  10*torch.ones(task_num,requires_grad=True)  
        # if kernel_type == "rbf":
        #     model.likelihood.noise_covar.noise = 0.1
        #     model.covar_module.kernels[0].base_kernel.lengthscale = 20
        #     model.covar_module.kernels[0].outputscale = 2
        #     model.covar_module.kernels[1].bias    =  1e-3     
        # if kernel_type == "linear":        
           # model.likelihood.noise_covar.noise = 5                       
    #if method == "gpi_ori":
    
