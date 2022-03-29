# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:27:22 2022

@author: mahom
"""


import math
import sys
import numpy as np
import numpy.matlib
import time
import scipy as SP
import os
import torch
import gpytorch
from matplotlib import pyplot as plt
import pathlib as Path
from os import listdir
import pandas as pd


ProjectPath = Path.Path.cwd()
utilsPath = Path.Path.joinpath(ProjectPath,"utils")
probsUtilsPath = Path.Path.joinpath(ProjectPath,"Prob-utils")
ResultsPath = Path.Path.joinpath(ProjectPath,"Results")


UTIL_DIR = utilsPath
sys.path.append(
    str(UTIL_DIR)
)

UTIL_DIR_GEN = probsUtilsPath
sys.path.append(
    str(UTIL_DIR_GEN)
)

RESULTS_DIR_GEN = ResultsPath
sys.path.append(
    str(RESULTS_DIR_GEN)
)

from EvaluateConfidenceIntervals import EvaluateConfidenceIntervals
from StandarizeData import StandarizeData
from DeStandarizeData import DeStandarizeData
from MAPE import MAPE
from GP24I_v4 import GP24I
from GPind import GPind
from GPind_ori import GPind_ori
from predGPind_ori import predGPind_ori
from GPind_lap import GPind_lap
from predGPK import predGPK
from predGPind_lap import predGPind_lap
from GPKtorch import GPKtorch
from GPK_sparse_torch import GPK_sparse_torch
from predGPind import predGPind
from load_obj import load_obj
from save_obj import save_obj
from sklearn.metrics import r2_score
from data_to_torch import data_to_torch
from norm2laplace import norm2laplace
from EvaluateConfidenceIntervals_Laplace import EvaluateConfidenceIntervals_Laplace
from outliers_removal import outliers_removal
from load_configuration import load_configuration
from load_configuration_job_array import load_configuration_job_array
from print_configuration import print_configuration
from correcting_factor_cov import correcting_factor_cov
from correcting_factor_cov_gamma import correcting_factor_cov_gamma
from predictive_variance_white import predictive_variance_white
from  print_extra_methods import  print_extra_methods
from GP24I_v4 import GP24I
from to_torch import to_torch
from predGPK_sp import predGPK_sp
from print_results_ic import print_results_ic
from PlotDayDecomposition_sparse import PlotDayDecomposition_sparse
# #Load Power Load Data =========================================================
# #==============================================================================
method = "NMF"  # Full
methodfile = 'NMF'
kernel_type = "rbf"
forecast_method = "gpk_sp" # gp_ind_ori/gp_ind/gpk/gp_ind_laplace/gpmt/gpk_sp
option_lv = "gp_ind_ori" # gp_ind_ori/gpmt
if forecast_method == "gpk":
    name_forecast_method = forecast_method +"_" +option_lv
else:
    name_forecast_method = forecast_method
EXPERIMENT = 2  # This has to do with the verion of the NMF generated
TaskNumber = 24
Stand = False
#folder_data_name = "Exp_"+str(EXPERIMENT)
#folder_data_name = "BuenosResNMF"
folder_data_name =  "Exp_interp_1_" + str(EXPERIMENT)
#LOCATIONS = ['ME','CT','NH','RI','NEMASSBOST','SEMASS','VT','WCMASS']

datapath = Path.Path.joinpath(ProjectPath,"Data",folder_data_name,'NMF')
DATAPATH = str(datapath)
onlyfilesALL = [f for f in listdir(DATAPATH) if f.endswith('.pkl')]

[onlyfiles,opt_parameters,forecast_method] = load_configuration(sys.argv,onlyfilesALL,forecast_method)    
#[onlyfiles,opt_parameters, forecast_method, option_lv] = load_configuration_job_array(sys.argv,onlyfilesALL)


gpytorch.settings.max_cg_iterations._set_value(10000)


RESULTS = {}
for archivo in range(len(onlyfiles)):
    Results = {'R224': [],'mapes':[],'mapemedio':[],'training_time':[],'test_time':[],
               'Ypred':[],'Vpred':[],'likelihood':[],'ICs':[],'ICs_lap1':[],'ICs_lap2':[],'gpk':[]}

   
    # LOAD DATA================================================================
    file_name = onlyfiles[archivo]
    file_path = Path.Path.joinpath(datapath,file_name)
    FILE_PATH = str(file_path)
    DATA = load_obj(FILE_PATH)
    DATA = data_to_torch(DATA)
    print(FILE_PATH)
    
    if 'metaNMFsparse' in DATA:   
        metaNMFsparse = DATA['metaNMFsparse']
        metaNMFsparse_test = DATA['metaNMFsparse_test']
    
    XTrain = DATA['X_Train_Val'].T    # N x F ### torch.from_numpy
    YTrain = DATA['Y_Train_Val']           
    XTest = DATA['X_Test'].T           # N x F           
    #YTest = DATA['Y_Test']           # N x K   
    Ntest = XTest.size(0)
    n_task = YTrain.size(1)
    YTest = torch.ones(Ntest,n_task)         
    YTest_24 = DATA['Y_Test_24']     # N x T         
    YTrain_24 = DATA['Y_Train_Val_24']     
    TaskNumber = np.size(DATA['Wtrain_load'],1)
    WTrain = to_torch(DATA['Wtrain_load'])
    Stds_train_load = DATA['Stds_train_load']
    Ntest = np.size(YTest_24,0)
    Ntrain = np.size(YTrain_24,0)

    PlotDayDecomposition_sparse(YTrain_24,WTrain,YTrain,Stds_train_load)
    #[XTrain,XTest,YTrain_24,YTest_24] = outliers_removal(XTrain,XTest,YTrain_24,YTest_24)
    

    # nn = 100
    # YTrain_24_std = np.divide(YTrain_24,np.matlib.repmat(Stds_train_load.T,Ntrain,1))
    # YTrain24M  = YTrain_24[0:nn,:]
    # YTrainstd24M  = YTrain_24_std[0:nn,:]
    # XTrainM = XTrain[0:nn,:]
    # YTrainM = YTrain[0:nn,:]
    # XTrain  = XTrainM
    # YTrain = YTrainM
    # YTrain_24 = YTrain24M
    # YTrain_24_std = YTrainstd24M  

    # NORMATLIZATION================================================================
    
    if forecast_method == "gpk_sp":
        [XTrain_S, YTrain_K_S , XTest_S, YTest_K_S,scalerX, scalerY_K]=StandarizeData(XTrain,YTrain, XTest,YTest,Standarize = Stand) 
    else:
        [XTrain_S, YTrain_24_S , XTest_S, YTest_24_S,scalerX, scalerY_24]=StandarizeData(XTrain,YTrain_24, XTest,YTest_24,Standarize = Stand)

    start = time.time()
    # TRAINING================================================================
    #==========================================================================
    if forecast_method == "gp_ind_ori": 
        [M,L,RESULTS,model,like] = GPind_ori(XTrain_S,YTrain_24_S,24,kernel_type,opt_parameters)
    #elif forecast_method == "gpk": 
    end = time.time() 
    training_time = end-start
    #=========================================================================
    if forecast_method == "gpk": 
        K = YTrain.size(1)
        [M,L,RESULTS,model,like,ind_val] = GPKtorch(XTrain_S,YTrain_K_S,WTrain,K,kernel_type,option_lv,opt_parameters)
    if forecast_method == "gpk_sp": 
        K = YTrain.size(1)
        Ntest = XTest.size(0)
        [OptModel,OptLikelihood, RESULTS,IDRegressorTypes, IDRegressorTypes_test,X_test,Labels,Labels_test,Indices_test,Ind_Val] = GPK_sparse_torch(XTrain_S,YTrain_K_S,XTest_S,WTrain,K,kernel_type,option_lv,opt_parameters,metaNMFsparse,metaNMFsparse_test)

    end = time.time() 
    training_time = end-start
    # TESTING==================================================================
    #==========================================================================
    start = time.time()  
    if forecast_method == "gpk_sp": 
        [YPredictedS_KgpS,VPredicted_Kgp_S] = predGPK_sp(Ntest,K,OptModel,OptLikelihood, IDRegressorTypes, IDRegressorTypes_test,X_test,Labels,Labels_test,Indices_test)
        [_, YPredicted_24gp_K,VPredicted_24gp_K]=DeStandarizeData(YTest_K_S,YPredictedS_KgpS,scalerY_K,VPredicted_Kgp_S,Standarize = Stand)
    end = time.time() 
    testing_time = end-start


#==============================================================================
#==============================================================================
#==============================================================================
    print_configuration(file_name,name_forecast_method,kernel_type,EXPERIMENT,Stand,folder_data_name)
    if forecast_method == "gpk": 
        # TRANSFORMATION====
        S2norm = torch.pow(Stds_train_load,2)
        Snorm = Stds_train_load.T.repeat(Ntest,1)
        Snorm_tr  = Stds_train_load.T.repeat(Ntrain,1)
        #ErrorValidation_std_P = torch.stack(RESULTS['ValidationPredictiveErrors'],dim =1)
        YPredicted_24gp = (YPredicted_24gp_K@WTrain.T)*Snorm
        VPredicted_24gp = torch.zeros((Ntest,24))
        # if 'ValidationErrors' in RESULTS:
        #     ErrorValidation_std = torch.stack(RESULTS['ValidationErrors'],dim =1)
        #     Nval = ErrorValidation_std.size(0)
        #     Snorm_val = Stds_train_load.T.repeat(Nval,1)
        #     NoiseEstimation_Variance3  = torch.var((ErrorValidation_std@WTrain.T)*Snorm_val,axis=0) 
        #     ind_a = np.random.permutation(range(0,Ntrain))[0:100]
        #     a = correcting_factor_cov(model,WTrain,YTrain_24[ind_a,:],XTrain_S[ind_a,:],option_lv,scalerY_K,NoiseEstimation_Variance3,Stds_train_load )
        #     a_gamma = correcting_factor_cov_gamma(model,WTrain,YTrain_24[ind_a,:],XTrain_S[ind_a,:],option_lv,scalerY_K,NoiseEstimation_Variance3,Stds_train_load )
        #     for ss in range(0,Ntest):
        #         VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  NoiseEstimation_Variance3)*a
        #     VPredicted_24gp_white = predictive_variance_white(VPredicted_24gp_K,WTrain,NoiseEstimation_Variance3,S2norm)
        print_extra_methods(Stds_train_load,Ntest,Ntrain,WTrain,YTrain_24,YTest_24,XTrain_S,YPredicted_24gp_K,VPredicted_24gp_K,option_lv,scalerY_K,RESULTS,model,DATA)
    elif forecast_method == "gp_ind_ori":
        [_, YPredicted_24gp,VPredicted_24gp]=DeStandarizeData(YTest_24_S,YPredicted_24gp_S,scalerY_24,VPredicted_24gp_S,Standarize = Stand)
        [ICs,ICs_lap1,ICs_lap2] = print_results_ic(YPredicted_24gp,YTest_24,VPredicted_24gp,"gp_ind_ori")
    elif forecast_method == "gpk_sp":
        S2norm = torch.pow(Stds_train_load,2)
        Snorm = Stds_train_load.T.repeat(Ntest,1)
        Snorm_tr  = Stds_train_load.T.repeat(Ntrain,1)
        #ErrorValidation_std_P = torch.stack(RESULTS['ValidationPredictiveErrors'],dim =1)
        YPredicted_24gp = (YPredicted_24gp_K@WTrain.T)*Snorm
        VPredicted_24gp = torch.zeros((Ntest,24))
        
        if Stand == True:
            stdgp = torch.sqrt(torch.tensor(scalerY_K.var_)).float()
        else:
            stdgp = torch.sqrt(torch.tensor(1.0)).float()

            
        # GP predictive validation variance error============================================
        Labels_val = {}
        if 'ValidationPredictiveErrors' in RESULTS:
            for t in range(0,len(metaNMFsparse['LabelClass'])):
                Labels_val_k = metaNMFsparse['LabelClass']['task{}'.format(t+1)]
                Labels_val['task{}'.format(t+1)] = Labels_val_k
            E = construct_sparse_pred_error(Labels_val,Y,Ind_Val)

            Var_EvalPred_std = torch.tensor([torch.var(x) for x in EvalPred])
            NoiseEstimation_Variance_GPpred = (WTrain@(Var_EvalPred_std*torch.pow(stdgp,2))@WTrain.T)*torch.pow(Stds_train_load.T,2)
        # GP  validation variance error============================================
        if 'ValidationErrors' in RESULTS:
            ErrorValidation_std = torch.stack(RESULTS['ValidationErrors'],dim =1)
            ErrorValidation_aux = ErrorValidation_std*stdgp
            NoiseEstimation_Variance_GPval  = torch.var((ErrorValidation_aux@WTrain.T)*Snorm_val,axis=0) 
        
        
        for ss in range(0,Ntest):
            VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  NoiseEstimation_Variance_GPpred)
    # METRICS==================================================================
    
    
    mapes= MAPE(YTest_24,YPredicted_24gp)
    mapemedio = torch.mean(mapes)

    NTest = np.size(YTest_24,0)
    R2_all = np.zeros((NTest,1))
     
    for samp in range(0,NTest):
        R2_all[samp,0] = r2_score(YTest_24[samp,:],YPredicted_24gp[samp,:])
    r2_24gp  = np.mean(R2_all)
    
 
    # PRINT===================================================================

    
    
    print('Mape Medio 24GPs indep   ', mapemedio )
    print('R2 24GPs i:    ',r2_24gp)
    
    
    if 'ValidationErrors' in RESULTS:
        Lval = RESULTS['ValidationErrors']
        Lval_tasks = [torch.mean(x) for x in Lval]
        Lval_mean = torch.mean(torch.tensor(Lval_tasks))
        print('Mean validation loss ',Lval_mean)
    
    
    print('Training time:   ', training_time )
    print('Test time:   ',  testing_time)
    #==========================================================================
    Results['R224'] = r2_24gp
    Results['mapes'] = mapes
    Results['mapemedio'] = mapemedio 
    Results['training_time'] =   training_time
    Results['test_time'] = testing_time
    Results['Ypred'] = YPredicted_24gp
    Results['Vpred'] = VPredicted_24gp
    Results['likelihood'] = like
    #Results['ICs'] = ICs
    #Results['ICs_lap1'] = ICs_lap1
    #Results['ICs_lap2'] = ICs_lap2
    
    if forecast_method == "gpk":
        Results['Wtrain'] = WTrain
    RESULTS[archivo] = Results
    
    file_name = name_forecast_method+"_Stand_"+str(Stand)
    if 'INFO' in locals():
        file_name = file_name="Exp_"+str(EXPERIMENT)
    file_results = Path.Path.joinpath(ResultsPath,file_name+"_results")
    file_model = Path.Path.joinpath(ResultsPath,file_name+"_model")
    file_data = Path.Path.joinpath(ResultsPath,file_name+"_data")
    save_obj(RESULTS, file_results.as_posix())
    save_obj(model, file_model.as_posix())
    save_obj(DATA, file_data.as_posix())






