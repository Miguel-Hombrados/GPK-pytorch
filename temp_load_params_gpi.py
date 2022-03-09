# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:14:25 2022

@author: mahom
"""


print("lenthscale",model.covar_module.kernels[0].base_kernel.lengthscale[0:3])
print("outputscale",model.covar_module.kernels[0].outputscale[0:3])
print("bias",model.covar_module.kernels[1].bias[0:3])
print("noise",model.likelihood.task_noises[0:3])

print("outputscale_fix",model.covar_module.kernels[0].base_kernel.outputscale)


print("lenthscale",model.covar_module.kernels[0].base_kernel.lengthscale)
print("outputscale",model.covar_module.kernels[0].outputscale)
print("bias",model.covar_module.kernels[1].bias)
print("noise",model.likelihood.task_noises)


model.likelihood.raw_task_noises.grad
model.covar_module.kernels[0].base_kernel.raw_outputscale.grad




for task in range(1,8):

    #print("lenthscale",model['task{}'.format(task)].covar_module.kernels[0].base_kernel.lengthscale)
    #print("outputscale",model['task{}'.format(task)].covar_module.kernels[0].outputscale)
    #print("bias",model['task{}'.format(task)].covar_module.kernels[1].bias)
    print("noise",model['task{}'.format(task)].likelihood.noise)