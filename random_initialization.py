# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:10:21 2022

@author: mahom
"""
import random
from to_torch import to_torch
def random_initialization(model,likelihood,kernel_type,task_num,method):
    if method == "gpi_ori": 
        if kernel_type == "linear":
            print("Include random initialization for linear kernel!")
        if kernel_type == "rbf":
            ub = model.likelihood.noise_covar.raw_noise_constraint.upper_bound
            lb = model.likelihood.noise_covar.raw_noise_constraint.lower_bound
            noise0 = to_torch(random.uniform(lb,ub))
            model.likelihood.noise =noise0.requires_grad_(True)
            #################################################
            ub = model.covar_module.kernels[0].raw_outputscale_constraint.upper_bound
            lb = model.covar_module.kernels[0].raw_outputscale_constraint.lower_bound
            outputscale0 = to_torch(random.uniform(lb,ub))
            model.covar_module.kernels[0].outputscale =outputscale0.requires_grad_(True)
            #################################################
            ub = model.covar_module.kernels[0].base_kernel.raw_lengthscale_constraint.upper_bound
            lb = model.covar_module.kernels[0].base_kernel.raw_lengthscale_constraint.lower_bound
            lengthscale0 = to_torch(random.uniform(lb,ub))
            model.covar_module.kernels[0].base_kernel.lengthscale = lengthscale0.requires_grad_(True)
            #################################################
            ub = model.covar_module.kernels[1].raw_bias_constraint.upper_bound
            lb = model.covar_module.kernels[1].raw_bias_constraint.lower_bound
            bias0 = to_torch(random.uniform(lb,ub))
            model.covar_module.kernels[1].bias = bias0.requires_grad_(True)
            