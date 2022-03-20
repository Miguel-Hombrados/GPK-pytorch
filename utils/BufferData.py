# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:23:03 2020

@author: mahom
"""
import numpy as np
def BufferData(Data,Delay):
    """

    Parameters
    ----------
    Data : Real Matrix (features x samples).
    This function takes a matrix of temporal data, where the columns of the 
    matrix are the time stamps and returns another matrix of data, where the
    number #Delay of time stamps are buffered and stacked as features.
    
    The last sample is excluded always from the buffer to allow a match
    between X and Y elements.

    Returns
    -------
    BufferData : Real Matrix (Delay*features x samples-Delay)

    """
    NumberOfSamples = np.size(Data,1)
    NumberOfFeatures = np.size(Data,0)
    BuffData = np.zeros((Delay*NumberOfFeatures,NumberOfSamples-Delay))
    for sample in range(0, NumberOfSamples-Delay):
        BuffData[:,[sample]] = Data[:,sample:sample+Delay].reshape(-1,1)

    return BuffData
   
        

        
    
    
    
    