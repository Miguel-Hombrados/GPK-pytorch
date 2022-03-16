# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:47:15 2022

@author: mahom
"""


# Script for fast reading and analyzing the results computed in CARC
import torch
import pathlib as Path
from load_obj import load_obj
from os import listdir

#====================================================================
method = "NMF"  # Full
methodfile = 'NMF'
kernel_type = "rbf"
forecast_method = "gp_ind_ori" # gp_ind_ori/gp_ind/gpk/gp_ind_laplace/gpmt
option_lv = "gp_ind_ori" # gp_ind_ori/gpmt
if forecast_method == "gpk":
    name_forecast_method = forecast_method +"_" +option_lv
else:
    name_forecast_method = forecast_method
EXPERIMENT = 2  # This has to do with the verion of the NMF generated
TaskNumber = 24
Stand = True
location = 'ME'
#folder_data_name = "Exp_"+str(EXPERIMENT)
folder_data_name = "BuenosResNMF"
#LOCATIONS = ['ME','CT','NH','RI','NEMASSBOST','SEMASS','VT','WCMASS']
#====================================================================
file_name = name_forecast_method+"_Stand_"+str(Stand)
if 'INFO' in locals():
    file_name = file_name="Exp_"+str(EXPERIMENT)


ProjectPath = Path.Path.cwd()
ResultsPath = Path.Path.joinpath(ProjectPath,"Results")
datapath = Path.Path.joinpath(ResultsPath,"Data",folder_data_name)
DATAPATH = str(datapath)
onlyfilesALL = [f for f in listdir(DATAPATH) if f.endswith('.pkl')]
onlyfiles = [f for f in onlyfilesALL if location in f ]

#====================================================================
file_results = Path.Path.joinpath(ResultsPath,file_name+"_results")
file_model = Path.Path.joinpath(ResultsPath,file_name+"_model")
file_data = Path.Path.joinpath(ResultsPath,file_name+"_data")
RESULTS = load_obj(file_results.as_posix())
M = load_obj(file_model.as_posix())
DATA = load_obj(file_data.as_posix())
#====================================================================
archivo = 0
Results = RESULTS[archivo]
print('Mape Medio 24GPs indep   ', Results['mapemedio'] )
print('R2 24GPs i:    ', Results['R224'] )
print('Training time:   ', Results['training_time'])
print('Test time:   ', Results['test_time'])
(ICS1_24,ICS2_24) = Results['ICs']
(ICS1_24_l1,ICS2_24_l1) = Results['ICs_lap1'] 
(ICS1_24_l2,ICS2_24_l2) = Results['ICs_lap2'] 
print('Confidence intervals Mean  : ',(torch.mean(ICS1_24.T),torch.mean(ICS2_24.T)))
print('Confidence intervals laplace 1  Mean: ', (torch.mean(ICS1_24_l1.T),torch.mean(ICS2_24_l1.T)))
print('Confidence intervals laplace 2 Mean : ', (torch.mean(ICS1_24_l2.T),torch.mean(ICS2_24_l2.T)))

#==========================================================================
n_tasks = len(model)
for task in range(0,n_tasks):
           print('outputscale',model['task{}'.format(task+1)].covar_module.kernels[0].outputscale)
 
for task in range(0,n_tasks):
           print('lengthscale',model['task{}'.format(task+1)].covar_module.kernels[0].base_kernel.lengthscale)
           
for task in range(0,n_tasks):
           print('bias',model['task{}'.format(task+1)].covar_module.kernels[1].bias)          

for task in range(0,n_tasks):
           print('noises',model['task{}'.format(task+1)].likelihood.noise)         

for task in range(0,n_tasks):
           print('means',model['task{}'.format(task+1)].mean_module.constant)         
           
           




    #==========================================================================

