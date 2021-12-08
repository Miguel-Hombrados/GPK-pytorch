#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:19:42 2020

@author: Miguel A Hombrados
"""
import numpy as np
def EvaluateConfidenceIntervals(YTest,YPredicted,VPredicted):
    """
    YTest : array (n_samples , n_task)
    YPredicted : array (n_samples , n_task)
    VPredicted : array - either (n_samples x n_task,n_samples x n_task ) or
                                (n_samples ,n_task)
    Returns: Intervals1Std and Intervals2Std a vector of size n_task, containing 
             the % of samples that fall within 1 or 2 Std's distance from the 
             prediction. 
        
    """
    if np.size(np.shape(YTest))==1:
        YTest = YTest.reshape(-1,1)
        YPredicted = YPredicted.reshape(-1,1)
    if np.size(np.shape(VPredicted))==1:  
        VPredicted = VPredicted.reshape(-1,1)
        
    n_tasks = np.size(YTest,1)
    n_samples = np.size(YTest,0)
    Intervals1Std= np.zeros((n_tasks,1))
    Intervals2Std= np.zeros((n_tasks,1))
    
    
    if VPredicted.shape == (n_samples, n_tasks):
            SP = np.sqrt(VPredicted)
    if VPredicted.shape == (n_samples*n_tasks , n_samples*n_tasks):
            SP = np.sqrt(np.diagonal(VPredicted).reshape(n_tasks,n_samples).T)  # REVIEW THIS !!
    
    for t in range(0,n_tasks):
         Yt = np.matrix(YTest[:,t]).reshape(-1,1)
         Yp = np.matrix(YPredicted[:,t]).reshape(-1,1)
         SSp = np.matrix(SP[:,t]).reshape(-1,1)
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
    