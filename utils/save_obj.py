# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:15:09 2020

@author: mahom
"""

import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

