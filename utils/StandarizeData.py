#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:39:21 2020

@author: Miguel A Hombrados

v2: This version returns also the scalerX and scalerY
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
def StandarizeData(XTrain,YTrain,XTest,YTest,Standarize = False):
    """
    X : n_samples x n_features
    Y : n_samples x n_tasks
    """
    
    if np.size(np.shape(XTest)) == 1:
        XTest = XTest.reshape(1,-1)
    if np.size(np.shape(YTrain)) == 1:   
        YTrain = YTrain.reshape(1,-1)
    if np.size(np.shape(XTrain)) == 1:      
        XTrain = XTrain.reshape(1,-1)
    if np.size(np.shape(YTest)) == 1:      
        YTest = YTest.reshape(1,-1)

    if Standarize == True:
        scalerX=StandardScaler().fit(XTrain)
        scalerY=StandardScaler().fit(YTrain)
    
        XTrainS = scalerX.fit_transform(XTrain)
        YTrainS = scalerY.fit_transform(YTrain)
        
        XTestS = scalerX.transform(XTest,scalerX)
        YTestS = scalerY.transform(YTest,scalerY)
  
        
        
    if Standarize == False:
        XTrainS = XTrain
        YTrainS = YTrain
        XTestS  = XTest
        YTestS  = YTest
        scalerX=0
        scalerY=0
        
    return XTrainS, YTrainS , XTestS, YTestS, scalerX, scalerY