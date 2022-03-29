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
            model.covar_module.kernels[0].outputscale = 50*torch.ones(task_num,requires_grad=True)  
            model.likelihood.task_noises = 0.1*torch.ones(task_num,requires_grad=True)                                                        # 0.5
            model.covar_module.kernels[0].base_kernel.lengthscale = 35*torch.ones(task_num,1,1,requires_grad=True) # probar 96.5                #3
            model.covar_module.kernels[1].bias = 0.3*torch.ones(task_num,requires_grad=True)   
            
    elif method == "gpi_ori": 
        if kernel_type == "linear":
             #model.likelihood.noise = torch.tensor(0.1)
             model.likelihood.noise = 0.5*torch.ones(task_num,requires_grad=True)                               
             model.covar_module.kernels[0].base_kernel.outputscale = 10*torch.ones(task_num,requires_grad=True)             
             model.covar_module.kernels[0].base_kernel.variance = 10*torch.ones(task_num,1,1,requires_grad=True)             
             model.covar_module.kernels[1].bias = 1e-5*torch.ones(task_num,1,1,requires_grad=True)         
        if kernel_type == "rbf":
            model.covar_module.kernels[0].outputscale = 25*torch.ones(task_num,requires_grad=True)  
            model.likelihood.noise = 1*torch.ones(task_num,requires_grad=True)                                                        # 0.5
            model.covar_module.kernels[0].base_kernel.lengthscale = 50*torch.ones(task_num,1,1,requires_grad=True) # probar 96.5                #3
            model.covar_module.kernels[1].bias = 0.5*torch.ones(task_num,requires_grad=True)  
            #model.mean_module.constant = torch.nn.Parameter(torch.tensor(0.))
            model.mean_module.initialize(constant=0.)
    elif method == "gpk_sp":
        if kernel_type == "rbf":
            model.covar_module.kernels[0].outputscale = 25*torch.ones(task_num,requires_grad=True)  
            model.likelihood.noise = 1*torch.ones(task_num,requires_grad=True)                                                        # 0.5
            model.covar_module.kernels[0].base_kernel.lengthscale = 50*torch.ones(task_num,1,1,requires_grad=True) # probar 96.5                #3
            model.covar_module.kernels[1].bias = 0.5*torch.ones(task_num,requires_grad=True)  
            #model.mean_module.constant = torch.nn.Parameter(torch.tensor(0.))
            model.mean_module.initialize(constant=0.)
    elif method == "gpmt":
        if kernel_type == "rbf":
            model.likelihood.noise = torch.tensor([1e-6]).requires_grad_(True)                                            # 0.5
            model.covar_module.covar_module_list[0].data_covar_module.kernels[0].outputscale = 25*torch.ones(task_num,requires_grad=True)  
            model.likelihood.task_noises = 1*torch.ones(task_num,requires_grad=True)                                                        # 0.5
            model.covar_module.covar_module_list[0].data_covar_module.kernels[0].base_kernel.lengthscale = 50*torch.ones(task_num,1,1,requires_grad=True) # probar 96.5                #3
            model.covar_module.covar_module_list[0].data_covar_module.kernels[1].bias = 0.5*torch.ones(task_num,requires_grad=True)  
            for t in range(0,task_num):
                model.mean_module.base_means[t].constant = torch.nn.Parameter(torch.tensor(0.))
