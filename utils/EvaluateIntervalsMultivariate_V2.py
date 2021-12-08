# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:51:52 2021

@author: mahom

    YTrue : array (n_samples , n_task)
    YPredicted : array (n_samples , n_task)
    VPredicted : array - either (n_samples ,n_task, n_task ) or

    Returns: Intervals68 and Intervals95 are two scalars containing 
             the % of samples that fall within 68% or 95% probability.

"""
from scipy.stats import mvn
import scipy as SP
import numpy as np
from GuaranteePSD import GuaranteePSD
from scipy.stats import multivariate_normal as mvn
def EvaluateIntervalsMultivariate_V2(YPredicted,VPredicted,YTrue):
    [NSamples,NTasks] = np.shape(YPredicted)
    SamplesIn68 = np.zeros((NSamples))
    SamplesIn95 = np.zeros((NSamples))
    
    for ss in range(0,NSamples):
        Yp = np.squeeze(YPredicted[ss,:])
        Yt = np.squeeze(YTrue[ss,:])
        Vp = np.squeeze(VPredicted[ss,:,:])
        
        E = Yt-Yp
        En = -E
        mu = np.zeros((NTasks))
        low = mu
        upp = np.array(E)

        S = np.array(Vp)

        S = S + 1e-2*np.eye(NTasks)
        dist = mvn(mean=mu, cov=S)
        p = dist.cdf(np.array(E))
        print("CDF:",p)

        
        if 2*p<(0.6827):
            SamplesIn68[ss] = 1;
        if p<(0.9545):
            SamplesIn95[ss] = 1;
        
    
    Intervals68 = 100*sum(SamplesIn68)/NSamples
    Intervals95 = 100*sum(SamplesIn95)/NSamples
    
    return Intervals68,Intervals95
    
    
    
    
    
    