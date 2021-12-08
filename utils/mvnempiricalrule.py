# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:40:59 2021

@author: Miguel A Hombrados
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
# """Function that computes the confidence intervals of samples drawn from 
#    a multivariate Gaussian distribution (68%-95%-99.7%)by whitening the samples. 
#   
#   Parameters:
#   mean: 1-D array_like, of length F
#   cov: 2-D array_like, of shape (F, F)
#   x: 2-D array_like, of length (N,F)

#   Returns:
#   cis: 1-D array_like, of length 3   
# """
def mvnempiricalrule(mean,cov,x):
    if np.size(np.shape(upper))==2:
        N = np.size(upper,0)
        F = np.size(upper,1)
    if np.size(np.shape(upper))==1:
        N = 1
        F = np.size(upper)
        cis = np.zeros(N)
        upper = upper.reshape(1,-1)       
    dist = mvn(mean=mean, cov=cov)   
    for n in range(0,N):
        p = dist.cdf(upper[n,:])
        cis[n] = (p-0.25)*(2**F)
        print("CDF:",cis[n])
    return cis
       
