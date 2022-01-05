# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 18:32:41 2022

@author: mahom
"""

import torch
import numpy as np
def to_torch(x):
    
    if isinstance(x, (np.ndarray, np.generic) ):
        y= torch.from_numpy(x).float()
    else:
        y=x.float()
    return y
