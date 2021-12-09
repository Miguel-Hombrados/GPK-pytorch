# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:15:23 2020

@author: mahom  V2
"""
import pickle

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)