# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:50:17 2021

@author: mahom
"""
import numpy as np
def GuaranteePSD(M):
    """
    Description: This function takes a symmetric matrix and returns the closest
    Positive semidefinite matrix in a frobenius norm sense. [1]
    
    Parameters
    ----------
    M : ndarray
        
        Imput symmetric matrix.
    
    Returns
    -------
    Mpsd : ndarray
    
           Output positive definite matrix.

    [1] Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite matrix. Linear algebra and its applications, 103, 103-118.

    """
    B = (M.T + M)/2
    [A,U] = np.linalg.eig(B)
    minvalue = 0
    Apsd = np.where(A <= 0, minvalue, A)
    Jit = 1e-6
    while np.min(Apsd)<=0:
        Apsd = Apsd + Jit   
        Jit = Jit*10
    Mpsd = U*Apsd*U.T
    return Mpsd 
    
    