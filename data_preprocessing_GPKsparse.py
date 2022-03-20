# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:09:33 2020

@author: mahom
"""

import sys
import pathlib as Path
import numpy as np
import scipy.io
from os import listdir
import pandas as pd
import os
print(os.getcwd())
ProjectPath = Path.Path.cwd()
DataPath = Path.Path.joinpath(ProjectPath,'Data')
PreprocessedDataPath = Path.Path.joinpath(DataPath,'Preprocessed')
UtilsPath = Path.Path.joinpath(ProjectPath,"utils")
sys.path.append(
    str(UtilsPath)
)
sys.path.append(
    str(DataPath)
)


from BufferData import BufferData
from save_obj import save_obj

EXPERIMENT = 2
exp_name = 'Exp_interp_1_'+str(EXPERIMENT)
folderpath = Path.Path.joinpath(DataPath ,exp_name,'NMF')
Delay = 7


onlyfiles = [f for f in listdir(folderpath)]
features = 'load'


for archivo in range(len(onlyfiles)):

    
    file_name = onlyfiles[archivo]
    filepath = Path.Path.joinpath(folderpath,file_name)
    ReducedData = scipy.io.loadmat(filepath)
    RD= ReducedData['RESULTS']  # GUARDAR COMO UN ARRAY DE CELULLAS DE ESTRUCTURAS EN MATLAB
    mdtype = RD.dtype
    ReducedData = {n: RD[n][0, 0] for n in mdtype.names}
    location = str(ReducedData['location'][0])
    method = str(ReducedData['method'][0])
    stdnmf = str(ReducedData['stdnmf'][0])
    Ninit = str(ReducedData['Ninit'][0])
    normW = str(ReducedData['normW'][0])
    P = str(ReducedData['P'][0])
    name_file_data_prep = "DATA_Train_Val_"+str(location)+".mat"
    Data_Train_Val = scipy.io.loadmat(str(Path.Path.joinpath(PreprocessedDataPath,name_file_data_prep)))   # DATA USED FOR TRAINING AND VALIDATING GPs
    Data_Test = scipy.io.loadmat(str(Path.Path.joinpath(PreprocessedDataPath,name_file_data_prep)))   # DATA USED FOR TRAINING AND VALIDATING GPs
    
       
       
    if features == 'load':
    #==================================================================
    LabelClass = ReducedData['LabelClass'] 
    LabelClass_test = ReducedData['LabelClass_test'] 
    NamesLabelsClass = ReducedData['NamesLabelsClass'] 
    NamesLabelsClass_test = ReducedData['NamesLabelsClass_test'] 
    
    #==================================================================
        Load_test = Data_Test['Load'].T                         # 24 x Ntest
        Load_Train_Val = Data_Train_Val['Load'].T               # 24 x Ntrain
        
        Load_reduced_Train_Val = ReducedData['Hload_opt']       # K x Ntrain
           
           
        Wtrain_load = ReducedData['Wload_opt']                  # 24 x K
        
        #Load_reduced_test =  ReducedData['Hload_test']          # K x Ntest
        
        #Data_test_aux_red  = Load_reduced_test                  # K x Ntest 
        Data_trainGP_aux_red = Load_reduced_Train_Val           # K x Ntrain
        
        Data_test_aux  = Load_test                              # 24 x Ntest 
        Data_trainGP_aux = Load_Train_Val                       # 24 x Ntrain 
       
    ### BUFFER DATA TO CREATE AN AUTOREGRESSIVE MODEL WITH DELAY == "Delay"
    ### FOR THE REDUCED AND NON-REDUCED FEATURES
    
    X_Test = BufferData(Data_test_aux,Delay)                    # Delay*24 x Ntest
    #Y_Test = Load_reduced_test.T[Delay:]                    # Ntest x K
    Y_Test_24 = Data_Test['Load'][Delay:]                       # Ntest x 24
    
    #X_Train_Val_red = BufferData(Data_trainGP_aux_red,Delay)
    X_Train_Val = BufferData(Data_trainGP_aux,Delay)           # Delay*24 x Ntrain
    Y_Train_Val = Load_reduced_Train_Val.T[Delay:]         # Ntrain x K
    Y_Train_Val_24 = Data_Train_Val['Load'][Delay:]            # Ntrain x 24
    #==================================================================
    
    Stds_train_load = ReducedData['Stds_train_load']            # 24 x 1
    
    # SAVE ==================================================
    Info = {'location':location,'method':method,'stdnmf':stdnmf,'Ninit':Ninit,'normW':normW,'P':P,'features':features,'Exp':EXPERIMENT,'Delay':Delay}
    
    DATA = {'X_Train_Val':X_Train_Val,'Y_Train_Val':Y_Train_Val,'X_Test': X_Test,'Y_Test_24':Y_Test_24,'Y_Train_Val_24':Y_Train_Val_24,'Wtrain_load':Wtrain_load,'Stds_train_load':Stds_train_load,'Info':Info,'RD':ReducedData}
    name_file = "GPK_"+str(method)+"_reduced_dataset_"+str(location)+'_std_'+str(stdnmf)+'_feat_'+str(features) + '_EXP_'+ str(EXPERIMENT)+ '_normW_'+ str(normW)+ '_P_'+ str(P)+ '_Ninit'+ str(Ninit)
    filepath = Path.Path.joinpath(folderpath,name_file)
    save_obj(DATA,str(filepath))
       
       
       
       
       
       
       
       
