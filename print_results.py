# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:03 2022

@author: mahom
"""


def print_results(folder_data_name,RESULTS):
    
    

    print('Mape Medio 24GPs indep   ', RESULTS[archivo]['mapemedio'])
    print('R2 24GPs i:    ',RESULTS[archivo]['R224'])
     
    print('Confidence intervals : ', RESULTS[archivo]['ICs'])
    print('Confidence intervals laplace 1 : ', RESULTS[archivo]['ICs_lap1'])
    print('Confidence intervals laplace 2 : ', RESULTS[archivo]['ICs_lap2'])
    print('Training time:   ',RESULTS[archivo]['training_time'] )
    print('Test time:   ', RESULTS[archivo]['test_time'])