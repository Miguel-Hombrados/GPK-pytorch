#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:59:35 2020

@author: apple
"""
import numpy as np
from scipy.linalg import fractional_matrix_power


def VerifyVariance(Yp,Yt,Vp):
    # Yp:  Standarized predictions  Tasks x TestSamples 
    # Vp:  Standarized Covariance matrices Tasks x Tasks x TestSamples
    # Yt:  Standarized Test  matrix Tasks x TestSamples 
    
    # Returns: Return the % of samples within 1 Std, 2Std and 3Std
    
    NSamples = np.size(Yp,1)
    ErrorS =  Yp-Yt
    
    Ew = np.zeros((1,3,NSamples))
    for i in range(0,NSamples):

        Ew[:,:,i] = np.matrix(ErrorS[:,i])*np.matrix(fractional_matrix_power(Vp[:,:,i],-0.5))

    Ew = np.squeeze(Ew)
    Radious = np.linalg.norm(Ew,axis =0)
    
    S3=100*np.size(Radious[Radious<3])/NSamples
    S2=100*np.size(Radious[Radious<2])/NSamples
    S1=100*np.size(Radious[Radious<1])/NSamples
    
    return S1, S2, S3

    
    
    
    
    
    