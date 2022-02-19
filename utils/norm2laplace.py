#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:02:11 2022

@author: Miguel A Hombrados-Herrera

YTest : array (n_samples , n_task)
YPredicted : array (n_samples , n_task)
option: If 1: then returns D(laplace|gaussian)
        If 2: then returns D(gaussian|laplace) (default)
"""
import numpy as np
def norm2laplace(mu,var,option=2):
    
    # Matrices  N x F
    mu_lap = mu
    
    if option==1:
        b = np.sqrt(var/2)
    if option==2:
        b = np.sqrt((2*var)/np.pi)
    return mu_lap,b
    
    