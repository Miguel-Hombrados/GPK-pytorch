# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:18:01 2022

@author: mahom
"""


import torch
def construct_sparse_pred_error(K,labels,E):
    E_sparse = {}
    
    for t in range(0,len(labels)) :
       labels_t = labels['task{}'.format(t+1)]
       Ek = E['task{}'.format(t+1)]
       if t==0:
           for n in range(0,len(labels_t)):
               E_sparse['task{}'.format(day)].append()
       
       
    for n in range(0,len(labels_t)):
       Components =  labels_t[n]
       for k in range(0,5):
           if k==0:
               day = Components[k]
               EK[n,day] = E[n,0]
           if k==1:
               dow = Components[k]
               EK[n,dow+364] = E[n,dow]
           if k==2:
               EK[n,7+364+8] = E[n,8]
           if k==3:
               EK[n,7+364+8+1] = E[n,9]
           if k==4:
               EK[n,7+364+8+1+1] = E[n,10]
    return YK
        