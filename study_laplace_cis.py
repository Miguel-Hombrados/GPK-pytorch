# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:36:00 2022

@author: mahom
"""

import numpy as np


mu = 0
b = 2
N = 1000
X = np.random.laplace(mu,b,N)


pi=0.025
ps = 0.975

li = mu + b*np.log(2*pi)
ls = mu - b*np.log(2-2*ps)

Xaux = X[X<ls]
Xint = Xaux[Xaux>li]

Nint = len(Xint)
Rate = Nint/N