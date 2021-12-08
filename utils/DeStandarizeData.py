#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:49:37 2020

@author: apple
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import block_diag
def DeStandarizeData(YTestS,YPredictedS,scalerY,VPredictedS = None ,Standarize = False):
    """
    Y : n_samples x n_tasks
    
    v 3.0: Introduces as input arguments scalerY from the Standarize function 
           in order to simplify the code. 
    """
    

    if np.size(np.shape(YPredictedS)) == 1:
        YPredictedS.reshape(1,-1)
    if np.size(np.shape(YTestS)) == 1:      
        YTestS.reshape(1,-1)
    n_test_samples = np.size(YTestS,0)
    n_predicted_samples = np.size(YPredictedS,0)
    n_tasks = np.size(YTestS,1)
    if Standarize == True:

        YTest = scalerY.inverse_transform(YTestS,scalerY)
        YPredicted = scalerY.inverse_transform(YPredictedS,scalerY)
# AAAAA
        if VPredictedS is not None:
              if VPredictedS.shape == (n_predicted_samples, n_tasks):
                  VPredicted = VPredictedS * scalerY.var_
              if VPredictedS.shape == (n_tasks , n_predicted_samples, n_predicted_samples):
                  VPredictedS = block_diag(*list(VPredictedS));
                  Saux = np.kron(np.outer(np.sqrt(scalerY.var_),np.sqrt(scalerY.var_)),np.eye((n_predicted_samples)))
                  VPredicted = Saux@VPredictedS
    
    if Standarize == False:
        YTest = YTestS
        YPredicted = YPredictedS
        VPredicted = VPredictedS
       
    if VPredictedS is not None:
        
        return YTest, YPredicted, VPredicted
    if VPredictedS is None:
        
         return YTest, YPredicted