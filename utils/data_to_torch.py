# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:08:43 2022

@author: mahom
"""
import numpy as np
from to_torch import to_torch
def data_to_torch(data):
    for key, value in data.items():
            ent = data[key]
            if isinstance(ent, (float, int, np.ndarray, np.generic)):
                data[key] = to_torch(ent)
            else:
                data[key] = ent
    return data