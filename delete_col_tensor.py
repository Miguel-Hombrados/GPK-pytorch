# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:42:37 2022

@author: mahom
"""
import torch
import numpy as np
def delete_col_tensor(a, del_col, device):
    n = a.cpu().detach().numpy()
    n = np.delete(n, del_col, 1)
    n = torch.from_numpy(n).to(device)
    return n