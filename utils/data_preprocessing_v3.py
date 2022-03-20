
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:09:33 2020

@author: mahom
"""

import sys
import numpy as np
import scipy.io

import pandas as pd
project_path = 'C:/Users/mahom/Desktop/Paper_Load_Profiling_NMF_V4/'
sys.path.append(project_path +'Forecasting/') 
sys.path.append(project_path +'utils/') 

from DimReductionNMF import DimReductionNMF
from BufferData import BufferData
from save_obj import save_obj

Delay = 7
methodGP = 'GPt24'
features = 'load' ## load// all
stdnmf = 'y'
#ME/NH/VT/CT/RI/SEMASS/WCMASS/NEMASSBOST
location = 'ME'
LOCATIONS = ['ME','NH','VT','CT','RI','SEMASS','WCMASS','NEMASSBOST']
EXPERIMENTO = 3
method = 'NMFbias'#NMF  

for location in LOCATIONS:
    # LOADING MATLAB STRUCTURE AS A PANDAS AS DICTIONARY
    file_name = "Validation_"+location+"_"+method+"_std_"+ str(stdnmf) + '_Exp_'+ str(EXPERIMENTO) + ".mat" 
    ReducedData = scipy.io.loadmat(project_path+"Data/Exp_"+str(EXPERIMENTO)+"/"+str(method)+"/"+file_name)
    RD= ReducedData['RESULTS']   # GUARDAR COMO UN ARRAY DE CELULLAS DE ESTRUCTURAS EN MATLAB
    mdtype = RD.dtype
    ReducedData = {n: RD[n][0, 0] for n in mdtype.names}

    Data_Test = scipy.io.loadmat(project_path + "Data/Preprocessed/DATA_Test_"+str(location)+".mat") # DATA USED FOR TESTING THE EXPERIMENT
    Data_Train_Val = scipy.io.loadmat(project_path + "Data/Preprocessed/"+"DATA_Train_Val_"+str(location)+".mat")   # DATA USED FOR TRAINING AND VALIDATING GPs
    

    if features == 'load':
        Load_test = Data_Test['Load'].T                             # 24 x Ntest
        Load_Train_Val = Data_Train_Val['Load'].T                   # 24 x Ntrain
    
    if features =='all':      
        Load_test = Data_Test['Load'].T                             # 24 x Ntest
        Temperature_test = Data_Test['Temperature'].T               # 24 x Ntest
        Dew_test = Data_Test['DewPoint'].T                          # 24 x Ntest
        
        Load_Train_Val = Data_Train_Val['Load'].T                   # 24 x Ntrain
        Temperature_Train_Val = Data_Train_Val['Temperature'].T     # 24 x Ntrain
        Dew_Train_Val = Data_Train_Val['DewPoint'].T                # 24 x Ntrain

    if features =='load':
        Wtrain_load = ReducedData['Wload_opt']                      # 24 x K
        Load_reduced_train = ReducedData['Hload_opt'].T             # Ntrain x K
    if features =='all':   
        
        Wtrain_load = ReducedData['Wload_opt']                      # 24 x K
        Wtrain_temperature = ReducedData['Wtemperature_opt']        # 24 x K
        Wtrain_dew = ReducedData['Wdew_opt']                        # 24 x K

        Load_reduced_train = ReducedData['Hload_opt'].T             # Ntrain x K
        Temperature_reduced_train = ReducedData['Htemperature_opt'] # Ntrain x K
        Dew_reduced_train = ReducedData['Hdew_opt']                 # Ntrain x K
    
    ### REDUCTION OF THE NUMBER OF FEATURES THROUGH NMF MODEL GENERATED IN MATLAB
    if features =='load':
        Load_reduced_test =  ReducedData['Hload_test']
        Data_Test['Load_Red'] = Load_reduced_test 
        
    if features =='all':   

        Load_reduced_test =  ReducedData['Hload_test']
        Temperature_reduced_test =  ReducedData['Htemperature_test']
        Dew_reduced_test = ReducedData['Hdew_test']
    if features =='load':
        Load_reduced_Train_Val = ReducedData['Hload_opt']   # K x N
    if features =='all': 
        Load_reduced_Train_Val = ReducedData['Hload_opt']    # K x N
        Temperature_reduced_Train_Val = ReducedData['Htemperature_opt'] 
        Dew_reduced_Train_Val = ReducedData['Hdew_opt'] 

    
    ### CONCATENATE REDUCED VERSIONS OF THE THREE TYPES OF FEATURES
    if features =='all':
        Data_test_aux_red  = np.concatenate((Load_reduced_test,Temperature_reduced_test,Dew_reduced_test),axis = 0)
        Data_trainGP_aux_red = np.concatenate((Load_reduced_Train_Val,Temperature_reduced_Train_Val,Dew_reduced_Train_Val),axis = 0)
    if features =='load':
        Data_test_aux_red  = Load_reduced_test   # K x N
        Data_trainGP_aux_red = Load_reduced_Train_Val
    
    ### CONCATENATE NON-REDUCED VERSIONS OF THE THREE TYPES OF FEATURES
    if features =='all':
        Data_test_aux  = np.concatenate((Load_test,Temperature_test,Dew_test),axis = 0)
        Data_trainGP_aux = np.concatenate((Load_Train_Val,Temperature_Train_Val,Dew_Train_Val),axis = 0)
    if features =='load':
        Data_test_aux  = Load_test
        Data_trainGP_aux = Load_Train_Val
    ### BUFFER DATA TO CREATE AN AUTOREGRESSIVE MODEL WITH DELAY == "Delay"
    ### FOR THE REDUCED AND NON-REDUCED FEATURES
    X_Test_red = BufferData(Data_test_aux_red,Delay)
    X_Test = BufferData(Data_test_aux,Delay)
    Y_Test = Data_Test['Load'][Delay:]
    
    X_Train_Val_red = BufferData(Data_trainGP_aux_red,Delay)
    X_Train_Val = BufferData(Data_trainGP_aux,Delay)
    Y_Train_Val = Data_Train_Val['Load'][Delay:]
    #========================================================
    
    # SAVE ==================================================
    DATA = {'X_Test': X_Test, 'X_Test_red':X_Test_red,'Y_Test':Y_Test,'X_Train_Val':X_Train_Val,'X_Train_Val_red':X_Train_Val_red,'Y_Train_Val':Y_Train_Val,'Wtrain_load':Wtrain_load}

    save_obj(DATA,project_path+"Data/Exp_"+ str(EXPERIMENTO) + "/"+str(method)+"/"+str(methodGP)+"_"+str(method)+"_reduced_dataset_"+str(location)+"_std_"+str(stdnmf)+'_feat_'+str(features) + '_EXP_'+ str(EXPERIMENTO))








