# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:51:52 2021

@author: mahom

    YTrue : array (n_samples , n_var)
    YPredicted : array (n_samples , n_var)
    VPredicted : array - either (n_samples ,n_var, n_var ) or

    Returns: Intervals68 and Intervals95 are two scalars containing 
             the % of samples that fall within 68% or 95% probability.

    Version: 2.0 The previous version was not generalizes to multiple variables.
"""
import scipy as SP
import numpy as np
from GuaranteePSD import GuaranteePSD
from scipy.special import ndtri
def EvaluateIntervalsMultivariate(YPredicted,VPredicted,YTrue,psd = False):
    if np.size(np.shape(YPredicted))==2:
        n_samples = np.size(YPredicted,0)
        n_var = np.size(YPredicted,1)
    if np.size(np.shape(YPredicted))==1:
        n_samples = 1
        n_var = np.size(YPredicted)
        YPredicted = YPredicted.reshape(1,-1)
        YTrue = YTrue.reshape(1,-1)   
        VPredicted  = VPredicted.reshape(1,n_var,n_var)
    SamplesIn68 = np.zeros((n_samples))
    SamplesIn95 = np.zeros((n_samples))
    
    
    pi_68 = np.power(0.68,1/(n_var))
    pi_95 = np.power(0.95,1/(n_var))
    ci_68 = ndtri((pi_68/2)+0.5)
    ci_95 = ndtri((pi_95)/2+0.5)
    
    for ss in range(0,n_samples):
        Yp = np.squeeze(YPredicted[ss,:])
        Yt = np.squeeze(YTrue[ss,:])
        Vp = np.squeeze(VPredicted[ss,:,:])
        
        E = Yt-Yp
        Vp = (Vp + Vp.T)/2
        
        if psd == True:
            VpPSD = GuaranteePSD(Vp)
        else:
            VpPSD =Vp
        Vinv = np.linalg.inv(VpPSD)    
        L = np.linalg.cholesky(Vinv)
        Ew = np.abs(np.matmul(L,E))
        
        NormEw = np.linalg.norm(Ew,axis = 0)
        
        if NormEw<ci_68:
            SamplesIn68[ss] = 1;
        if NormEw<ci_95:
            SamplesIn95[ss] = 1;
        r = np.sqrt(-2*np.log(1-0.68))
        
    
    Intervals68 = 100*sum(SamplesIn68)/n_samples
    Intervals95 = 100*sum(SamplesIn95)/n_samples
    
    return Intervals68,Intervals95
    
    
    
    
    
    