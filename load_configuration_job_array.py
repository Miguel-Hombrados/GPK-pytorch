# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:45:03 2022

@author: mahom
"""

# Example script for job array for running multiple jobs in slurm


import sys
import pandas as pd
import pathlib as Path
from itertools import product
def load_configuration_job_array(input_params,onlyfilesALL):
  # index of the job array
    opt_parameters ={}
    ProjectPath = Path.Path.cwd()
    file_path = Path.Path.joinpath(ProjectPath,'parameters.xlsx')
    df = pd.read_excel(file_path)
    uniques = [df[i].unique().tolist() for i in df.columns ]
    dfcombinated = pd.DataFrame(product(*uniques), columns = df.columns)
    dfcombinated.dropna(inplace=True)
    index = sys.argv[1]
    forecast_method = dfcombinated[index,'methods']
    method_lv = dfcombinated[index,'method_lv']
    opt_parameters['lr1'] = float(dfcombinated[index,'lr'])
    opt_parameters['lr2'] = float(dfcombinated[index,'lr2'])
    opt_parameters['n_restarts'] = int(dfcombinated[index,'nrestarts'])
    opt_parameters['num_iter'] = int(dfcombinated[index,'niter'])
    opt_parameters['trainsize'] = float(dfcombinated[index,'trainsize'])
    location = "_"+str(dfcombinated[index,'location'])+"_"
    onlyfiles = [f for f in onlyfilesALL if location in f ]
    
    print("Forecasting method:", forecast_method)
    print("lv_method",method_lv)
    print("LOCATIONS:", onlyfiles)
    print("OPTIMIZATION PARAMETERS:")
    print("Learning Rate:",opt_parameters['lr1'])
    print("Learning Rate 2:",opt_parameters['lr2'])
    print("Number of Restarts:",opt_parameters['n_restarts'])
    print("Number of Iterations:",opt_parameters['num_iter'])
    print("Training size:",opt_parameters['trainsize'])

    return onlyfiles,opt_parameters, forecast_method, method_lv