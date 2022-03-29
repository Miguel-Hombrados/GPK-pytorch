# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:18:01 2022

@author: mahom
"""


import torch
def construct_sparse_pred(N,K,labels,Y):
    YK = torch.zeros(N,K)

    
    for n in range(0,N):
       Components =  labels[n]#metaNMFsparse_test['LabelClass_test'][nt]
       for k in range(0,5):
           if k==0:
               day = Components[k]
               YK[n,day] = Y[n,0]
           if k==1:
               dow = Components[k]
               YK[n,dow+364] = Y[n,dow]
           if k==2:
               YK[n,7+364+8] = Y[n,8]
           if k==3:
               YK[n,7+364+8+1] = Y[n,9]
           if k==4:
               YK[n,7+364+8+1+1] = Y[n,10]
    return YK
        