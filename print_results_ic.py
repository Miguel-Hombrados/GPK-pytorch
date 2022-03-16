# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:50:48 2022

@author: mahom
"""

import torch
from EvaluateConfidenceIntervals_Laplace import EvaluateConfidenceIntervals_Laplace
from norm2laplace import norm2laplace
from EvaluateConfidenceIntervals import EvaluateConfidenceIntervals
def print_results_ic(YPred24,YTest24,VarPred24,method):
    [_,Blap1] = norm2laplace(YPred24,VarPred24 ,option=1)
    [ICS1_l1,ICS2_l1] = EvaluateConfidenceIntervals_Laplace(YTest24,YPred24,Blap1)    
    [_,Blap2] = norm2laplace(YPred24,VarPred24,option=2)
    [ICS1_l2,ICS2_l2] = EvaluateConfidenceIntervals_Laplace(YTest24,YPred24,Blap2)    
    [ICS1,ICS2] = EvaluateConfidenceIntervals(YTest24,YPred24,VarPred24) 
    print("===========================================================================")
    print("===========================================================================")
    print(method)
    print('Confidence intervals Mean  : ',(torch.mean(ICS1_l1.T),torch.mean(ICS2_l1.T)))
    print('Confidence intervals laplace 1  Mean: ', (torch.mean(ICS1_l2.T),torch.mean(ICS2_l2.T)))
    print('Confidence intervals laplace 2 Mean : ', (torch.mean(ICS1.T),torch.mean(ICS2.T)))
    print("===========================================================================")
    print("===========================================================================")
    
    
    return (ICS1,ICS2),(ICS1_l1,ICS2_l1),(ICS1_l2,ICS2_l2)