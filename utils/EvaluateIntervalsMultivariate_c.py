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
from scipy.stats.distributions import chi2
def EvaluateIntervalsMultivariate_c(YPredicted,VPredicted,YTrue,psd = False):
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
    
    
    r_68 = np.sqrt(chi2.ppf(0.68, df=n_var))
    r_95 = np.sqrt(chi2.ppf(0.95, df=n_var))

    
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
        
        if NormEw<r_68:
            SamplesIn68[ss] = 1;
        if NormEw<r_95:
            SamplesIn95[ss] = 1;

        
    Intervals68 = 100*sum(SamplesIn68)/n_samples
    Intervals95 = 100*sum(SamplesIn95)/n_samples
    
    return Intervals68,Intervals95
    
    
    
    
    
    