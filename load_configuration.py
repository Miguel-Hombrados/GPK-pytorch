# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:45:12 2022

@author: mahom
"""


def load_configuration(input_params,onlyfilesALL):
    opt_parameters ={}
    if len(input_params)==1:
        print('Running default parameters and all locations')
        opt_parameters['lr1'] = 0.01
        opt_parameters['lr2'] = 0.005
        opt_parameters['n_restarts'] = 5
        opt_parameters['num_iter']  = 5
        opt_parameters['trainsize'] = 0.9
        onlyfiles = onlyfilesALL
    elif len(input_params)==2: 
        opt_parameters['lr1'] = 0.02
        opt_parameters['lr2'] = 0.02
        opt_parameters['n_restarts'] = 10
        opt_parameters['num_iter']  = 600
        opt_parameters['trainsize'] = 0.9
        print('loading specified location')
        location = "_"+input_params[1]+"_"
        onlyfiles = [f for f in onlyfilesALL if location in f ]
    else:
        print('loading parameters from file and specified location')
        opt_parameters['lr1'] = float(input_params[3])
        opt_parameters['lr2'] = float(input_params[4])
        opt_parameters['n_restarts'] = int(input_params[5])
        opt_parameters['num_iter']  = int(input_params[6])
        opt_parameters['trainsize'] = float(input_params[7])
        location = "_"+input_params[2]+"_"
        onlyfiles = [f for f in onlyfilesALL if location in f ]
    print("LOCATIONS:", onlyfiles)
    print("OPTIMIZATION PARAMETERS:")
    print("Learning Rate:",opt_parameters['lr1'])
    print("Learning Rate 2:",opt_parameters['lr2'])
    print("Number of Restarts:",opt_parameters['n_restarts'])
    print("Number of Iterations:",opt_parameters['num_iter'])
    print("Training size:",opt_parameters['trainsize'])
    
    
    return onlyfiles,opt_parameters