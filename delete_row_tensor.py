# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:45:45 2022

@author: 
"""


import torch
import numpy as np
def delete_row_tensor(a, del_row, device):
    n = a.cpu().detach().numpy()
    n = np.delete(n, del_row, 0)
    n = torch.from_numpy(n).to(device)
    return n