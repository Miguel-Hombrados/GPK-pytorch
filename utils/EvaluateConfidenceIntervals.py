#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:19:42 2020

@author: Miguel A Hombrados
"""
import torch
def EvaluateConfidenceIntervals(YTest,YPredicted,VPredicted):
    """
    YTest : array (n_samples , n_task)
    YPredicted : array (n_samples , n_task)
    VPredicted : array - either (n_samples x n_task,n_samples x n_task ) or
                                (n_samples ,n_task)
    Returns: Intervals1Std and Intervals2Std a vector of size n_task, containing 
             the % of samples that fall within 1 or 2 Std's distance from the 
             prediction. 
             
    Torch version    
    """
    if  len(YTest.shape)==1:
        YTest = YTest.reshape(-1,1)
        YPredicted = YPredicted.reshape(-1,1)
    if  len(VPredicted.shape)==1:
        VPredicted = VPredicted.reshape(-1,1)
        
    n_tasks = torch.Tensor.size(YTest,1)
    n_samples = torch.Tensor.size(YTest,0)
    Intervals1Std= torch.zeros((n_tasks,1))
    Intervals2Std= torch.zeros((n_tasks,1))
    
    
    if VPredicted.shape == torch.Size([n_samples, n_tasks]):
<<<<<<< HEAD
            SP = np.sqrt(VPredicted)
    if VPredicted.shape == (n_samples*n_tasks , n_samples*n_tasks):
            SP = np.sqrt(np.diagonal(VPredicted).reshape(n_tasks,n_samples).T)  # REVIEW THIS !!
=======
            SP = torch.sqrt(VPredicted)
    if VPredicted.shape == torch.Size([n_samples*n_tasks , n_samples*n_tasks]):
            SP = torch.sqrt(torch.diagonal(VPredicted).reshape(n_tasks,n_samples).T)  # REVIEW THIS !!
>>>>>>> fe85a1c14081347f15f9db4147bde7738ffe5d8c
    
    for t in range(0,n_tasks):
         Yt = YTest[:,t].reshape(-1,1)
         Yp = YPredicted[:,t].reshape(-1,1)
         SSp = SP[:,t].reshape(-1,1)
          # The transposes and the matrix transformations are
          # done in order to guarantee that the shape is n_samples x n_task
          
         Uthreshold = Yp + SSp
         Lthreshold = Yp - SSp
         c = 0
         for x, y, z in zip(Yt,Lthreshold,Uthreshold):
             if x > y and x < z:   # Your code
                c = c + 1
         
         Intervals1Std[t] = 100*c/n_samples
         Uthreshold2 = Yp + SSp*2
         Lthreshold2 = Yp - SSp*2
         c2 = 0
         for x, y, z in zip(Yt,Lthreshold2,Uthreshold2):
             if x > y and x < z:   # Your code
                 c2 = c2 + 1
         Intervals2Std[t] = 100*c2/n_samples
    
    return Intervals1Std, Intervals2Std
    