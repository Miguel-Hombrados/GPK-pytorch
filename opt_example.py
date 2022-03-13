# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:13:29 2022

@author: mahom
"""
import torch 
from load_obj import load_obj
def opt_example(model,likelihood):    
    DATA = load_obj("C:/Users/mahom/Desktop/GPt24_Full__std_y_allLocations.pkl");
    M = DATA['MODELS_ALL']
    ct = M[3]
    AL = DATA['Alphas_ALL']
    
    for task in range(0,24):
        print(ct[task].kernel_)
    for task in range(0,24):
        with torch.no_grad():
            model['task{}'.format(task+1)].covar_module.kernels[0].outputscale = ct[task].kernel_.k1.k1.k1.constant_value
            model['task{}'.format(task+1)].covar_module.kernels[0].base_kernel.lengthscale = ct[task].kernel_.k1.k1.k2.length_scale
            model['task{}'.format(task+1)].covar_module.kernels[1].bias  = ct[task].kernel_.k1.k2.constant_value
            model['task{}'.format(task+1)].likelihood.noise_covar.noise = ct[task].kernel_.k2.noise_level + torch.tensor([1e-6])
            
            likelihood['task{}'.format(task+1)].noise_covar.noise = ct[task].kernel_.k2.noise_level+ torch.tensor([1e-6])
            
            model['task{}'.format(task+1)].mean_module.constant = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
            
    # n_tasks = 24
    # outputScales = torch.zeros(n_tasks,requires_grad=True)
    # LScales = torch.zeros(n_tasks,1,1,requires_grad=True)
    # bias = torch.zeros(n_tasks,requires_grad=True)
    # noises = torch.zeros(n_tasks,requires_grad=True)
    # for task in range(0,24):
    #     with torch.no_grad():
    #         noises[task] = ct[task].kernel_.k2.noise_level
    #         bias[task] = ct[task].kernel_.k1.k2.constant_value
    #         LScales[task] = ct[task].kernel_.k1.k1.k2.length_scale
    #         outputScales[task] = ct[task].kernel_.k1.k1.k1.constant_value
        
    # model.likelihood.task_noises = noises                                                        # 0.5
    # model.covar_module.kernels[0].base_kernel.lengthscale = LScales # probar 15                  #3
    # model.covar_module.kernels[1].bias = bias         
    # model.covar_module.kernels[0].outputscale =  outputScales
        