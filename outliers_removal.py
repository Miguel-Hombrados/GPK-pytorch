# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:02:34 2022

@author: mahom
"""
import torch
from delete_row_tensor import delete_row_tensor
def outliers_removal(Xtrain, Xtest,Ytrain,Ytest):
    # It seems GP prediction with ISO-NE struggles to estimate a propper IC
    # a good potential reason is the presence of outliers. In particular, there is 
    # one sample per year that overhoots and seems a good candidate around day
    # 300.
    
    #Note: Ideally would be to remove such samples from NMF training too, but 
    # that would take more time.
    
    # In the hour = 1  of that day on "Y"
    #Samples identified as outliers between 2011 and 2014.index: 302,666,924,1394,(1030) 
                                                    #2015      : 297
    # This indeces corespond to the data after  buffering with window = 7.
    delay = 7
    indexTrain= list(range(302,302+delay+1))+list(range(666,666+delay+1))+list(range(924,924+delay+1))+list(range(1394,1394+delay+1))
    indexTest = list(range(297,297+delay+1))

    Xtrain_c = delete_row_tensor(Xtrain, indexTrain,"cpu")
    Ytrain_c = delete_row_tensor(Ytrain, indexTrain,"cpu")
    Xtest_c = delete_row_tensor(Xtest, indexTest,"cpu")
    Ytest_c = delete_row_tensor(Ytest, indexTest,"cpu")
    
    return Xtrain_c,Xtest_c,Ytrain_c,Ytest_c



                                                    
                                                