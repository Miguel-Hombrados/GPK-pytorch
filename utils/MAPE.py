# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 00:38:58 2017

@author: MiguelAngel
"""

import torch
def MAPE(Yreal,Yestimated):
    #☺both are matrices NXS N stands for number of time instants and S by number of time series
    error=(100/torch.Tensor.size(Yestimated,0))*torch.sum(torch.abs(torch.divide(Yestimated-Yreal,Yreal)),0)
    
    return error