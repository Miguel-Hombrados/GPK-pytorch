#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:05:03 2021

@author: Miguel A Hombrados
"""
import pandas as pd
import torch
from GPind_ori import GPind_ori
from GPind import GPind
from GPMT import GPMT
from to_torch import to_torch
from sklearn.model_selection import train_test_split
from gp_single import gp_single
def GPK_sparse_torch(x,y,x_test,W,n_tasks,kernel_type,option_lv,opt_parameters,metaNMFsparse,metaNMFsparse_test):
    
    trainsize = opt_parameters['trainsize']  
    x = to_torch(x)
    y = to_torch(y)
    x_test = to_torch(x_test)
    IDRegressorTypes = torch.tensor(metaNMFsparse['RegressorsIds'])
    NbOfRegressors = torch.unique(torch.tensor(metaNMFsparse['RegressorsIds']).reshape(-1,1))
    IDRegressorTypes_test = torch.tensor(metaNMFsparse_test['RegressorsIds_test'])
    
    LabelClass = torch.tensor(metaNMFsparse['LabelClass'])
    LabelClass_test = torch.tensor(metaNMFsparse_test['LabelClass_test'])
        
    #NamesLabelClass = torch.tensor(metaNMFsparse['NamesLabelsClass'])
    #NamesLabelClass_test = torch.tensor(metaNMFsparse_test['NamesLabelsClass_test'])
    
    X = {}
    Y = {}
    X_test = {}
    Y_test = {}
    Labels = {}
    Labels_test = {}
    Indices = {}
    Indices_test = {}
    OptModel = {}
    OptLikelihood = {}
    Results = {}
    Validation_Errors = {}
    Validation_Predictive_Errors = {}
    Ind_Val = {}
    
    for r in range(0, len(NbOfRegressors)+1):
        
        if r<len(NbOfRegressors):
            select_mask = IDRegressorTypes  == NbOfRegressors[r]
               
            select_mask_test = IDRegressorTypes_test == NbOfRegressors[r]
            indices_r = select_mask.nonzero()
            indices_test_r = select_mask_test.nonzero()
            indices_samples_r = indices_r[:,0].ravel()
            indices_test_samples_r = indices_test_r[:,0].ravel()
            
            if NbOfRegressors[r] ==10:
                mask10 = LabelClass[:,3] !=0
                mask10_test = LabelClass_test[:,3] !=0
                indices_samples_r = mask10.nonzero()[:,0]
                indices_samples_test_r = mask10_test.nonzero()[:,0]
            
            
            if 1<=NbOfRegressors[r]<=7:
                index_latent_var = NbOfRegressors[r]-1
            if  NbOfRegressors[r]==8:
                index_latent_var = list(range(7,7+365))
            if  NbOfRegressors[r]==9:
                index_latent_var = list(range(7+365,7+365+8))
            if  NbOfRegressors[r]==10:
                index_latent_var = list(range(7+365+8,7+365+8+10))
            else:
                index_latent_var = torch.tensor([390])
            if  NbOfRegressors[r]==10:
                indlatent = LabelClass[:,3]!=0
                LabelClass_prune = LabelClass[indlatent.nonzero(),3].squeeze()
                yaux = y[:,index_latent_var]
                yaux2 = yaux[indlatent,:]
                yaux3 = []
                for i in range(0,yaux2.size(0)):
                    yaux3.append(yaux2[i,LabelClass_prune[i]-1])
                y_r = torch.FloatTensor(yaux3)
                
            else:
          
                yaux = y[indices_samples_r,:]
                y_r = yaux[:,index_latent_var].reshape(-1,1).ravel()
            x_r = x[indices_samples_r,:]  
            x_test_r = x_test[indices_test_samples_r,:]
            ### Create input indices-----
            if 1<=NbOfRegressors[r]<=7:
                print("Processing weekdays")
            if  NbOfRegressors[r]==8:    
                ind_doy = LabelClass[select_mask].reshape(-1,1)
                ind_doy_test = LabelClass_test[select_mask_test].reshape(-1,1)
                x_r = torch.cat((x_r,ind_doy),dim=1)
                x_test_r = torch.cat((x_test_r,ind_doy_test),dim=1)
            if  NbOfRegressors[r]==9:
                print("Processing year")
            if  NbOfRegressors[r]==10:
                ind_h = LabelClass[indices_samples_r,3].reshape(-1,1)
                ind_h_test = LabelClass_test[indices_test_samples_r,3].reshape(-1,1)
                x_r = torch.cat((x_r,ind_h),dim=1)
                x_test_r = torch.cat((x_test_r,ind_h_test),dim=1)
            
        if r== len(NbOfRegressors):
            y_r = y[:,-1]
            x_r = x
            x_test_r = x_test
        
        LabelClass_r = LabelClass[indices_samples_r,:]
        LabelClass_test_r = LabelClass_test[indices_test_samples_r,:]
        
        X['task{}'.format(r+1)]  = x_r
        Y['task{}'.format(r+1)]  = y_r
        X_test['task{}'.format(r+1)]  = x_test_r
        Labels['task{}'.format(r+1)]  = LabelClass_r
        Labels['task{}'.format(r+1)]  = LabelClass_r
        Labels_test['task{}'.format(r+1)]  = LabelClass_test_r
        Indices['task{}'.format(r+1)]  = indices_samples_r
        Indices_test['task{}'.format(r+1)]  = indices_test_samples_r
        
        [MODELS_r,LIKELIHOODS_r,Results_r,Opt_model_r,Opt_likelihood_r,Validation_Errors_r,Validation_Predictive_Errors_r,ind_val_t] = gp_single(x_r,y_r,kernel_type,opt_parameters)

        Ind_Val['task{}'.format(r+1)] = ind_val_t
        OptModel['task{}'.format(r+1)] = Opt_model_r
        OptLikelihood['task{}'.format(r+1)] = Opt_likelihood_r
        Results['task{}'.format(r+1)] = Results_r
        Validation_Errors['task{}'.format(r+1)]  = Validation_Errors_r
        Validation_Predictive_Errors['task{}'.format(r+1)] = Validation_Predictive_Errors_r
    Results['ValidationErrors']  =  Validation_Errors   
    Results['ValidationPredictiveErrors'] = Validation_Predictive_Errors
    return OptModel,OptLikelihood, Results,IDRegressorTypes, IDRegressorTypes_test,X_test,Labels,Labels_test,Indices_test,Ind_Val
