# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:07:05 2022

@author: mahom
"""
import torch
def skill_score(Ytrue,Ypred):
    
    Yprior = Ytrue[0:-1,:]
    Ypred_s = Ypred[1:,:]
    Ytrue_s = Ytrue[1:,:]
    
    MSEforecast = torch.mean(torch.pow(Ypred_s-Ytrue_s,2))
    MSEref = torch.mean(torch.pow(Yprior-Ytrue_s,2))
    
    SS = 1-(MSEforecast/MSEref)
    
    return SS
    