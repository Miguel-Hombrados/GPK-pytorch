#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:08:04 2022

@author: Miguel
"""
import torch
def EvaluateConfidenceIntervals_Laplace(YTest,YPredicted,Bs):
    """
    YTest : array (n_samples , n_task)
    YPredicted : array (n_samples , n_task)
    Bs : array (n_samples ,n_task)

                              
    Returns: Intervals1Std and Intervals2Std a vector of size n_task, containing 
             the % of samples that fall within intervals of 68% and 95%
        
    """
    if len(YTest.shape)==1:
        YTest = YTest.reshape(-1,1)
        YPredicted = YPredicted.reshape(-1,1)
    if (len(Bs.shape))==1:  
        Bs = Bs.reshape(-1,1)
        
    n_tasks = YTest.size(1)
    n_samples = YTest.size(0)
    Intervals1Std= torch.zeros((n_tasks,1))
    Intervals2Std= torch.zeros((n_tasks,1))
    

    Lower68 = Bs*torch.log(torch.Tensor([2*0.16]))
    Upper68 = Bs*torch.log(torch.Tensor([2*(1-0.84)]))
    
    Lower95 = Bs*torch.log(torch.Tensor([2*0.025]))
    Upper95 = Bs*torch.log(torch.Tensor([2*(1-0.975)]))


    
    for t in range(0,n_tasks):
         Yt = YTest[:,t].reshape(-1,1)
         Yp = YPredicted[:,t].reshape(-1,1)
         SSl = Lower68[:,t]
         SSu = Upper68[:,t]
          # The transposes and the matrix transformations are
          # done in order to guarantee that the shape is n_samples x n_task
          
         Uthreshold = Yp - SSu.reshape(-1,1)
         Lthreshold = Yp + SSl.reshape(-1,1)
         c = 0
         for x, y, z in zip(Yt,Lthreshold,Uthreshold):
             if x > y and x < z:   # Your code
                c = c + 1
         
         SSl = Lower95[:,t]
         SSu = Upper95[:,t]
         Intervals1Std[t] = 100*c/n_samples
         Uthreshold2 = Yp - SSu.reshape(-1,1)
         Lthreshold2 = Yp + SSl.reshape(-1,1)
         c2 = 0
         for x, y, z in zip(Yt,Lthreshold2,Uthreshold2):
             if x > y and x < z:   # Your code
                 c2 = c2 + 1
         Intervals2Std[t] = 100*c2/n_samples
    
    return Intervals1Std, Intervals2Std