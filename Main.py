#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 07:15:59 2020

 @author: Miguel A Hombrados


Description: Code for testing different datasets (It is all in the noise) and ISO NE
with different multitask GP configurations from the original code of KronSum

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
from predGPind import predGPind
from load_obj import load_obj
from save_obj import save_obj
from sklearn.metrics import r2_score
from data_to_torch import data_to_torch
from norm2laplace import norm2laplace
from EvaluateConfidenceIntervals_Laplace import EvaluateConfidenceIntervals_Laplace
from outliers_removal import outliers_removal
from load_configuration import load_configuration
from print_configuration import print_configuration
from correcting_factor_cov import correcting_factor_cov
from to_torch import to_torch
# #Load Power Load Data =========================================================
# #==============================================================================
method = "NMF"  # Full
methodfile = 'NMF'
kernel_type = "rbf"
forecast_method = "gp_ind_ori" # gp_ind_ori/gp_ind/gpk/gp_ind_laplace/gpmt
option_lv = "gp_ind_ori" # gp_ind_ori/gpmt
if forecast_method == "gpk":
    name_forecast_method = forecast_method +"_" +option_lv
else:
    name_forecast_method = forecast_method
EXPERIMENT = 3  # This has to do with the verion of the NMF generated
TaskNumber = 24
Stand = True
#folder_data_name = "Exp_"+str(EXPERIMENT)
folder_data_name = "BuenosResNMF"
#LOCATIONS = ['ME','CT','NH','RI','NEMASSBOST','SEMASS','VT','WCMASS']

datapath = Path.Path.joinpath(ProjectPath,"Data",folder_data_name)
DATAPATH = str(datapath)
onlyfilesALL = [f for f in listdir(DATAPATH) if f.endswith('.pkl')]

[onlyfiles,opt_parameters] = load_configuration(sys.argv,onlyfilesALL)    
print_configuration(name_forecast_method,kernel_type,EXPERIMENT,Stand,folder_data_name)


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
    

    XTrain = DATA['X_Train_Val'].T    # N x F ### torch.from_numpy
    YTrain = DATA['Y_Train_Val']           
    XTest = DATA['X_Test'].T           # N x F           
    YTest = DATA['Y_Test']           # N x K             
    YTest_24 = DATA['Y_Test_24']     # N x T         
    YTrain_24 = DATA['Y_Train_Val_24']     
    TaskNumber = np.size(DATA['Wtrain_load'],1)
    WTrain = to_torch(DATA['Wtrain_load'])
    Stds_train_load = DATA['Stds_train_load']
    Ntest = np.size(YTest_24,0)
    Ntrain = np.size(YTrain_24,0)

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
    
    if forecast_method == "gpk":
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
    end = time.time() 
    training_time = end-start
    # TESTING==================================================================
    #==========================================================================
    start = time.time()  
    if forecast_method == "gp_ind_ori": 
        [YPredicted_24gp_S,VPredicted_24gp_S] = predGPind_ori(XTest_S,like,model)
    end = time.time() 
    testing_time = end-start
    #=========================================================================
    if forecast_method == "gpk": 
        [YPredictedS_KgpS,VPredicted_Kgp_S] = predGPind_ori(XTest_S,like,model)
        [_, YPredicted_24gp_K,VPredicted_24gp_K]=DeStandarizeData(YTest_K_S,YPredictedS_KgpS,scalerY_K,VPredicted_Kgp_S,Standarize = Stand)
        a = correcting_factor_cov(model,WTrain,YTrain[ind_val,:],XTrain_S[ind_val,:],option_lv,scalerY_K)
        [YPredictedS_KgpS,VPredicted_Kgp_S] = predGPK(YPredicted_24gp_K,VPredicted_Kgp_S,WTrain,Stds_train_load = Stds_train_load,a = a)
    end = time.time() 
    testing_time = end-start
    #=========================================================================
    #[YPredicted_24gp_ind_S,VPredicted_24gp_ind_S] = predGPind(XTest_S,like_train_24ind,model_train_24ind)
    #[YPredicted_24gp_ind_S,VPredicted_24gp_ind_S] = predGPind_lap(XTest_S,like_train_24ind,model_train_24ind)

    
    if forecast_method == "gpk": 

        # TRANSFORMATION====
        S2norm = torch.pow(Stds_train_load,2)
        Snorm = Stds_train_load.T.repeat(Ntest,1)
        Snorm_tr  = Stds_train_load.T.repeat(Ntrain,1)
     
        ErrorValidation_std = torch.stack(RESULTS['ValidationErrors'],dim =1)
        YPredicted_24gp = (YPredicted_24gp_K@WTrain.T)*Snorm
      
        Nval = ErrorValidation_std.size(0)
        Snorm_val = Stds_train_load.T.repeat(Nval,1)

        NoiseEstimation_Variance3  = torch.var((ErrorValidation_std@WTrain.T)*Snorm_val,axis=0) 
        
        VPredicted_24gp = torch.zeros((Ntest,24))
        for ss in range(0,Ntest):
            VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  NoiseEstimation_Variance3)/24
    else:
        [_, YPredicted_24gp,VPredicted_24gp]=DeStandarizeData(YTest_24_S,YPredicted_24gp_S,scalerY_24,VPredicted_24gp_S,Standarize = Stand)




    
    

    



    ############################################################################
    #[model_train_24ind,like_train_24ind,n_opt,min_valid_loss,_] = GPind(XTrain_S,YTrain_24_S,24,kernel_type)
    #[model_train_24ind,like_train_24ind,history,best_params] = GPind_lap(XTrain_S,YTrain_24_S,24,kernel_type)
    #[model_train_24ind,like_train_24ind,n_opt,min_valid_loss,ErrorValidation] = GPKtorch(XTrain_S,YTrain_24_S,24,kernel_type,option_lv)
    
    #K = ErrorValidation.size(2)
    #Nfold = ErrorValidation.size(1)
    #Nval = ErrorValidation.size(0)
 
    #EV = ErrorValidation.reshape(Nval*Nfold,K)
    #Snorm_val = np.matlib.repmat(Stds_train_load.T,Nval,1)
    #A = scipy.linalg.sqrtm(np.linalg.inv(WTrain.T@WTrain))
    #NoiseEstimation_Variance3  = np.var((EV@A@WTrain.T)*Snorm_val,axis=0) 

    #for ss in range(0,Ntest):
    #    VPredicted_24gp_gpk[ss,:] = (np.diag(WTrain@A@np.diag(VPredicted_24gp_K[ss,:])@A@WTrain.T)*(S2norm.ravel())  +  NoiseEstimation_Variance3)#
    # DENORMATLIZATION================================================================
  
    # 24GPKind================================================================
    # start_gpk_ind = time.time()
    #
    # end_gpk_ind = time.time() 
    # training_time_mtgp = end_gpk_ind-start_gpk_ind
    
    # start_gpk_ind = time.time()
    # [YPredicted_Kgp_gpk_ind_S,VPredicted_Kgp_gpk_ind_S] = predGPind(XTest_S,like_train_24gpk_ind,model_train_24gpk_ind)
    # [_, YPredicted_Kgp_gpk_ind,VPredicted_Kgp_gpk_ind]=DeStandarizeData(YTest_K_S,YPredicted_Kgp_gpk_ind_S,scalerY_K,VPredicted_Kgp_gpk_ind_S,Standarize = Stand)
    # [YPredicted_24gp_gpk_ind,VPredicted_24gp_gpk_ind] = predGPK(YPredicted_Kgp_gpk_ind,VPredicted_Kgp_gpk_ind,WTrain,Stds_train_load = Stds_train_load)
    # ## Estimation of the validation error in NMF
    # YTrain_24_est = (WTrain@YTrain.T).T
    # Etrain = YTrain_24_est - YTrain_24
    # ##=========================================
    # end_gpk_ind = time.time() 
    # testing_time_mtgp = end_gpk_ind-start_gpk_ind
    # ====================================================================  


    # METRICS==================================================================
    [YPredicted_24gp_ind,Blap1] = norm2laplace(YPredicted_24gp,VPredicted_24gp,option=1)
    [ICS1_24_l1,ICS2_24_l1] = EvaluateConfidenceIntervals_Laplace(YTest_24,YPredicted_24gp_ind,Blap1)    
    [YPredicted_24gp,Blap2] = norm2laplace(YPredicted_24gp,VPredicted_24gp,option=2)
    [ICS1_24_l2,ICS2_24_l2] = EvaluateConfidenceIntervals_Laplace(YTest_24,YPredicted_24gp,Blap2)    
    [ICS1_24,ICS2_24] = EvaluateConfidenceIntervals(YTest_24,YPredicted_24gp,VPredicted_24gp)   

    
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
    
    print('Confidence intervals Mean  : ',(torch.mean(ICS1_24.T),torch.mean(ICS2_24.T)))
    print('Confidence intervals laplace 1  Mean: ', (torch.mean(ICS1_24_l1.T),torch.mean(ICS2_24_l1.T)))
    print('Confidence intervals laplace 2 Mean : ', (torch.mean(ICS1_24_l2.T),torch.mean(ICS2_24_l2.T)))
     
    print('Confidence intervals : ',(ICS1_24.T,ICS2_24.T))
    print('Confidence intervals laplace 1 : ', (ICS1_24_l1.T,ICS2_24_l1.T))
    print('Confidence intervals laplace 2 : ', (ICS1_24_l2.T,ICS2_24_l2.T))
    
    
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
    Results['ICs'] = (ICS1_24,ICS2_24)
    Results['ICs_lap1'] = (ICS1_24_l1,ICS2_24_l1)
    Results['ICs_lap2'] = (ICS1_24_l2,ICS2_24_l2)
    
    if forecast_method == "gpk":
        Results['Wtrain'] = WTrain
    RESULTS[archivo] = Results
    
    file_name = name_forecast_method+"_Stand_"+str(Stand)
    if 'INFO' in locals():
        file_name = file_name="Exp_"+str(EXPERIMENT)
    file_results = Path.Path.joinpath(ResultsPath,file_name+"_results")
    file_model = Path.Path.joinpath(ResultsPath,file_name+"_model")
    file_data = Path.Path.joinpath(ResultsPath,file_name+"_data")
    #save_obj(RESULTS, file_results.as_posix())
    #save_obj(RESULTS, file_model.as_posix())
    #save_obj(DATA, file_data.as_posix())






