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

from pathlib import Path
ProjectPath = Path.cwd()
utilsPath = Path.joinpath(ProjectPath,"utils")


UTIL_DIR = utilsPath
sys.path.append(
    str(UTIL_DIR)
)


#project_path  = "C:\Users\mahom\Documents\GitHub\GPK-pytorch\"
#ys.path.append(project_path+'utils\')
import EvaluateConfidenceIntervals
from StandarizeData import StandarizeData
from DeStandarizeData import DeStandarizeData
from MAPE import MAPE
from GP24I_v4 import GP24I
from GPindK import GPindK
from load_obj import load_obj
from save_obj import save_obj
from sklearn.metrics import r2_score



# #Load Power Load Data =========================================================
# #==============================================================================
method = "NMF"  # Full
methodGP = 'GPt24'
EXPERIMENT = 6
TaskNumber = 24
Stand = True


methodfile = 'NMF'
datapath = Path.joinpath(ProjectPath,"Data","Exp_"+str(EXPERIMENT),str(methodfile))
DATAPATH = str(datapath)
#datapath = file_path = project_path +"Data/BuenosResNMF/"
#onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
onlyfiles = [f for f in listdir(DATAPATH) if f.endswith('.pkl')]



ALL_R2 = np.zeros((len(onlyfiles)))
ALL_MAPE = np.zeros((len(onlyfiles)))
Times = np.zeros((len(onlyfiles)))
YTEST_ALL = [None]*len(onlyfiles)
YPREDICTED_ALL = [None]*len(onlyfiles)
VPREDICTED_ALL = [None]*len(onlyfiles)
MODELS_ALL = [None]*len(onlyfiles)
CIs_ALL = [None]*len(onlyfiles)
Alphas_ALL = [None]*len(onlyfiles)
IC1_ALL = [None]*len(onlyfiles)
IC2_ALL = [None]*len(onlyfiles)
ERROR_ALL = [None]*len(onlyfiles)
R2_ALL = [None]*len(onlyfiles)
MAPE_ALL = [None]*len(onlyfiles)
VAR_ALL = [None]*len(onlyfiles)
ERROR_ALL_train = [None]*len(onlyfiles)
R2_ALL_train = [None]*len(onlyfiles)



Alphas = [1e-6]
#Alphas = np.linspace(1e-6,1e-2,5)
# ### VALIDATION LOOP ===============================================

for archivo in range(len(onlyfiles)):
    
    file_name = onlyfiles[archivo]
    #file_path = project_path +"Data/Exp_"+str(EXPERIMENT)+"/"+str(methodfile)+"/"+file_name
    file_path = Path.joinpath(datapath,file_name)
    FILE_PATH = str(file_path)
    #file_path = project_path +"Data/BuenosResNMF/"+file_name
    DATA = load_obj(FILE_PATH)
    print(FILE_PATH)
    print(list(DATA))
    INFO = DATA['Info']
    INFO['Alphas'] = Alphas
    INFO['stdGP'] = Stand
    SecCopy= DATA['RD']
    Var_Noise_NMF_val = SecCopy['OptValVar_F']
    Var_Noise_NMF_train = SecCopy['OptTrainVar_F']

    XTrain = DATA['X_Train_Val']    # F x N ### torch.from_numpy
    YTrain = DATA['Y_Train_Val']           
    XTest = DATA['X_Test']           # F x N           
    YTest = DATA['Y_Test']           # N x K             
    YTest_24 = DATA['Y_Test_24']     # N x T         
    YTrain_24 = DATA['Y_Train_Val_24']     
    TaskNumber = np.size(DATA['Wtrain_load'],1)
    WTrain = DATA['Wtrain_load']
    Stds_train_load = DATA['Stds_train_load']
    Ntest = np.size(YTest_24,0)
    Ntrain = np.size(YTrain_24,0)
    YTrain_24_std = np.divide(YTrain_24,np.matlib.repmat(Stds_train_load.T,Ntrain,1))

    
    #WtrainPP = YTrain_24_std.T@YTrain@np.linalg.inv(YTrain.T@YTrain)
    #WTrain = WtrainPP
    #nn = 900
    #YTrain24M  = YTrain_24[0:nn,:]
    #YTrainstd24M  = YTrain_24_std[0:nn,:]
    #XTrainM = XTrain[:,0:nn]
    #YTrainM = YTrain[0:nn,:]
    #XTrain  = XTrainM
    #YTrain = YTrainM
    #YTrain_24 = YTrain24M
    #YTrain_24_std = YTrainstd24M
    
    Ntrain = np.size(YTrain_24,0)
    YY =(WTrain@YTrain.T)
    E = np.divide(YTrain_24.T,np.matlib.repmat(Stds_train_load,1,Ntrain))-YY
    Var_noise_train_est = np.var(E,axis =1)
    
    PrecisionH = WTrain.T@WTrain
    Alpha = SP.linalg.sqrtm(PrecisionH)
    #YTrain = YTrain@Alpha
    #YTest  = YTest@Alpha
    
    if method == 'Full':
        [XTrainS, YTrainS , XTestS, YTestS,scalerX, scalerY]=StandarizeData(XTrain.T,YTrain_24, XTest.T,YTest_24,Standarize = Stand)
    else:
        [XTrainS, YTrainS , XTestS, YTestS,scalerX, scalerY]=StandarizeData(XTrain.T,YTrain, XTest.T,YTest,Standarize = Stand)
 

    # 24GP================================================================
    start = time.time()
    [model_train,like_train] = GPindK(XTrainS,YTrainS,TaskNumber)
    end = time.time() 
    training_test_time = end-start
    # ====================================================================


    if method == 'Full':
        [YTest, YPredicted_24gp,VPredicted_24gp]=DeStandarizeData(YTestS,YPredictedS_24gpS,scalerY,VPredictedS_24gpS,Standarize = Stand)
    else:   
        
        #[YTest_K, YPredicted_24gp_K,CovPredicted_24gp_K]=DeStandarizeData(YTestS,YPredictedS_24gpS,scalerY,Covpredicted_Best,Standarize = Stand)
        [YTest_K, YPredicted_24gp_K,VPredicted_24gp_K]=DeStandarizeData(YTestS,YPredictedS_24gpS,scalerY,VPredictedS_24gpS,Standarize = Stand)
        [YTrain_K, YPredicted_24gp_K_tr,VPredicted_24gp_K_tr]=DeStandarizeData(YTrainS,YPredictedS_24gpS_tr,scalerY,VPredictedS_24gpS_tr,Standarize = Stand)
        VarK = np.var(YTest_K-YPredicted_24gp_K,axis=0)
        VarK_tr = np.var(YTrain_K-YPredicted_24gp_K_tr,axis=0)
        
         
        [ICS1_24gp0,ICS2_24gp0] = EvaluateConfidenceIntervals(YTest_K,YPredicted_24gp_K,VPredicted_24gp_K)  
        [ICS1_24gp0_tr,ICS2_24gp0_tr] = EvaluateConfidenceIntervals(YTrain_K,YPredicted_24gp_K_tr,VPredicted_24gp_K_tr) 
        ## CONTINUAR AQUI!!!!!!!
    #[ICS1_24gp0,ICS2_24gp0] = EvaluateConfidenceIntervals(YTest_K,YPredicted_24gp_K,VPredicted_24gp_K) 

    Ntest = np.size(YPredicted_24gp_K,0)
    Ntrain = np.size(YPredicted_24gp_K_tr,0)
    YTest_24_std = YTest_K@WTrain.T 
    YPredicted_24gp_std = YPredicted_24gp_K@WTrain.T

    YPredicted_24gp_std_tr = np.matmul(YPredicted_24gp_K_tr,WTrain.T)
    Etrain = YTrain_24_std-YPredicted_24gp_std_tr
    NoiseEstimation = np.diag(np.var(Etrain,axis=0)) 

    SigInv = np.linalg.inv(WTrain.T@WTrain)
    A = SP.linalg.sqrtm(SigInv)
    Vtr = np.diagonal(WTrain@(A@np.diag(VarK_tr)@A.T)@WTrain.T).reshape(-1,1)
    Vtr2 = np.diagonal(WTrain@(A@np.diag(np.mean(VPredicted_24gp_K,0))@A.T)@WTrain.T).reshape(-1,1)
    NMFError = np.diag(NoiseEstimation).ravel() - Vtr.ravel()
    NMFError_2 = np.diag(NoiseEstimation).ravel() -Vtr2.ravel()


    Wp = np.linalg.inv(WTrain.T@WTrain)@WTrain.T
    Wpp = np.linalg.inv(Wp.T@Wp)@Wp.T
    VPredicted_24gp = np.zeros((Ntest,24))
    VPredicted_24gp_2 = np.zeros((Ntest,24))
    VPredicted_24gp_3 = np.zeros((Ntest,24))
    VPredicted_24gp_tr = np.zeros((Ntrain,24))
    VpreMat = np.zeros((Ntest,24))
    VpreMat_tr = np.zeros((Ntrain,24))
    VpreMat_2 = np.zeros((Ntest,24))


    for ss in range(0,Ntest):
        VpreMat[ss,:] =  np.diagonal(WTrain@(A@np.diag(VPredicted_24gp_K[ss,:])@A.T)@WTrain.T)
        VpreMat_tr[ss,:] =  np.diagonal(WTrain@(A@np.diag(VPredicted_24gp_K_tr[ss,:])@A.T)@WTrain.T)
        
        VPredicted_24gp[ss,:] = (np.power(Stds_train_load,2)*(VpreMat[ss,:].reshape(-1,1)  + Var_Noise_NMF_val)).T
        VPredicted_24gp_3[ss,:] = (np.power(Stds_train_load,2)*(VpreMat[ss,:].reshape(-1,1)  + NMFError_2.reshape(-1,1))).T
        VPredicted_24gp_tr[ss,:] = (np.power(Stds_train_load,2)*(VpreMat_tr[ss,:].reshape(-1,1)  + Var_Noise_NMF_train)).T
        #VPredicted_24gp[ss,:] = (np.power(Stds_train_load,2)*(NoiseEstimation[ss,:].reshape(-1,1))).T
    YPredicted_24gp = np.matlib.repmat(Stds_train_load,1,Ntest).T* YPredicted_24gp_std
    YPredicted_24gp_tr = np.matlib.repmat(Stds_train_load,1,Ntrain).T* YPredicted_24gp_std_tr

    VPredicted_24gpOPT = np.power(Stds_train_load,2).reshape(1,-1)*np.matlib.repmat(np.diag(NoiseEstimation).reshape(1,-1),Ntest,1)


    [ICS1_24gp,ICS2_24gp] = EvaluateConfidenceIntervals(YTest_24,YPredicted_24gp,VPredicted_24gp) 
    
    [ICS1_24gp_2,ICS2_24gp_2] = EvaluateConfidenceIntervals(YTest_24,YPredicted_24gp,VPredicted_24gp_2) 
    
    [ICS1_24gp_3,ICS2_24gp_3] = EvaluateConfidenceIntervals(YTest_24,YPredicted_24gp,VPredicted_24gp_3)
    
    [ICS1_24gp_OPT,ICS2_24gp_OPT] = EvaluateConfidenceIntervals(YTest_24,YPredicted_24gp,VPredicted_24gpOPT)
    
    [ICS1_24gp_tr,ICS2_24gp_tr] = EvaluateConfidenceIntervals(YTrain_24,YPredicted_24gp_tr,VPredicted_24gp_tr)
    

    
 


    
# RESULTS ===============================================================
    mm_24gp = MAPE(YTest_24,YPredicted_24gp)
    mapemedio_24gp = np.mean(mm_24gp)
    
    mm_24gp_tr = MAPE(YTrain_24,YPredicted_24gp_tr)
    mapemedio_24gp_tr = np.mean(mm_24gp_tr)
    
    
     
    NTest = np.size(YTest_24,0)
    R2_all = np.zeros((NTest,1))
     
    for samp in range(0,NTest):
        R2_all[samp,0] = r2_score(YTest_24[samp,:],YPredicted_24gp[samp,:])
    r2_24gp  = np.mean(R2_all)
     
    ALL_R2[archivo] = r2_24gp            
    ALL_MAPE[archivo] =  mapemedio_24gp
    Times[archivo] = training_test_time
    YTEST_ALL[archivo] = YTest_24
    YPREDICTED_ALL[archivo] = YPredicted_24gp
    VPREDICTED_ALL[archivo] = VPredicted_24gp
    MODELS_ALL[archivo] = model
    CIs_ALL = (ICS1_24gp,ICS2_24gp)
    Alphas_ALL[archivo] = Opt_alpha
    IC1_ALL[archivo] = IC1
    IC1_ALL[archivo] = IC2
    ERROR_ALL[archivo] = Errors
    R2_ALL[archivo] = R2s
    MAPE_ALL[archivo] = MAPEs
    VAR_ALL[archivo] = VarsALL
    ERROR_ALL_train[archivo] = Errors_train
    R2_ALL_train[archivo] = R2strain
    
    
    location = DATA['Info']['location']
    stdnmf = DATA['Info']['stdnmf']
    
    print('LOCATION:   ', location)
    print('Mape Medio 24GPs indep   ', mapemedio_24gp)
    print('R2 24GPs indep:    ',r2_24gp )
     
    print('IC1%medio 24GPs indep: ', np.mean(ICS1_24gp))
    print('IC2%medio 24GPs indep:  ', np.mean(ICS2_24gp))
    print('IC1%medio 24GPs indep alt: ', np.mean(ICS1_24gp_2))
    print('IC2%medio 24GPs indep alt:  ', np.mean(ICS2_24gp_2))

    print('Training time:   ',training_time)
    print('Validation time:   ',val_time)
    print('Test time:   ',test_time)

RESULTS = {'ALL_R2':ALL_R2,'ALL_MAPE':ALL_MAPE,'YTEST_ALL':YTEST_ALL,'YPREDICTED_ALL':YPREDICTED_ALL,'VPREDICTED_ALL':VPREDICTED_ALL,'Times':Times,'MODELS_ALL':MODELS_ALL,'CIs_ALL':CIs_ALL,'Alphas_ALL':Alphas_ALL,'Errors':Errors,'R2_ALL':R2_ALL,' MAPE_ALL': MAPE_ALL,'VAR_ALL':VAR_ALL,'IC1_ALL':IC1_ALL,'IC2_ALL':IC2_ALL,'ERROR_ALL_train':ERROR_ALL_train,'R2_ALL_train':R2_ALL_train,'INFO':INFO,'training_time':training_time,'val_time':val_time,'test_time':test_time}
save_obj(RESULTS,project_path+"Data/Exp_"+ str(EXPERIMENT) + "/"+str(method)+"/GPK_"+str(method)+"_std_"+str(stdnmf)+'_'+'allLocations')






