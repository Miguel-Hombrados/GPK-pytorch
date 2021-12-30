#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:28:17 2021

@author: apple
"""
import torch
def getBnv(ITmatrix):
    
    T = ITmatrix.size(dim=0)
    method =  "lu"  # "sqrt"
    if method == "lu":
       B = torch.lu(ITmatrix)
    if method == "sqrt":
        [E,V] = torch.linalg.eig(ITmatrix)
        Ereal = torch.sqrt(E[:,0])
        B = V@torch.diag(Ereal)
    v = torch.min(torch.diag(ITmatrix))*torch.ones(T,1)
    
    return B,v
     