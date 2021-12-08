#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:55:25 2020

@author: Miguel 
"""
import sys
sys.path.append('C:/Users/mahom/Documents/Python Scripts/UNM/utils')
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import gaussian_process
from sklearn.model_selection import train_test_split
from EvaluateConfidenceIntervals import EvaluateConfidenceIntervals
from sklearn.metrics import r2_score
from MAPE import MAPE 
"""
Xtrain : samples X features
    
Ytrain :samples X tasks
    
Xtest : samples X features

Ypredicted : samples X tasks
    
Vpredicted : (samples x tasks) It includes the noise estimated. 

Alphas : Array or list of the alpha values evaluated. 


"""
def GP24I(Xtrain_all,Ytrain_all,Xtest,kernel, TaskNumber,Alphas):
    RateOfValSamples = 0.2
    NbOfTrainSamples = np.size(Xtrain_all,0)
    NbOfValSamples = int(np.ceil(RateOfValSamples*NbOfTrainSamples))
    NbOfTestSample = np.size(Xtest,0)
    Nfold = 5
    Errors2 = np.zeros((len(Alphas),Nfold,TaskNumber))
    ErrorsVal = np.zeros((NbOfValSamples,len(Alphas),Nfold,TaskNumber))
    ErrorsValOpt = np.zeros((NbOfValSamples,Nfold,TaskNumber))
    Errors2_train = np.zeros((len(Alphas),Nfold,TaskNumber))
    R2s = np.zeros((len(Alphas),Nfold,TaskNumber))
    R2strain = np.zeros((len(Alphas),Nfold,TaskNumber))
    MAPEs =  np.zeros((len(Alphas),Nfold,TaskNumber))
    VarsALL= np.zeros((len(Alphas),Nfold,NbOfValSamples ,TaskNumber))
    IC1 = np.zeros((len(Alphas),Nfold,NbOfValSamples,TaskNumber))
    IC2 = np.zeros((len(Alphas),Nfold,NbOfValSamples,TaskNumber))
    NoiseParameters = np.zeros((TaskNumber))
    Val_All = np.zeros((len(Alphas),Nfold,NbOfValSamples,TaskNumber))
    Val_pred_All = np.zeros((len(Alphas),Nfold,NbOfValSamples,TaskNumber))
    
    start_val_time = time.time()
    for  ind_alpha, alpha in enumerate(Alphas):
        for  fold in range(0,Nfold):
            Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain_all, Ytrain_all, test_size=RateOfValSamples,shuffle=False)
            
     
            Ypredicted_24 = np.zeros((np.size(Yval,0),np.size(Yval,1)))
            Vpredicted_24 = np.zeros((np.size(Yval,0),np.size(Yval,1)))
            
            start_train_time = time.time()
            for task in range(0,TaskNumber):   
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,alpha = alpha)
    
                # Fit to data using Maximum Likelihood Estimation of the parameters
                gp.fit(Xtrain, Ytrain[:,task])
          
                # Make the prediction on the meshed x-axis (ask for MSE as well)
                y_pred, sigma = gp.predict(Xval, return_std=True)
                y_pred_train, sigma_train = gp.predict(Xtrain, return_std=True)
                
                Ypredicted_24[:,task] = y_pred
                Vpredicted_24[:,task] = np.power(sigma,2) #+ alpha + gp.kernel_.k2.noise_level
                R2s[ind_alpha,fold,task] = gp.score(Xval, Yval[:,task])
                R2strain[ind_alpha,fold,task] = gp.score(Xtrain, Ytrain[:,task])
                
                Errors2[ind_alpha,fold,task] = np.linalg.norm(y_pred-Yval[:,task])/len(y_pred)
                ErrorsVal[:,ind_alpha,fold,task] = y_pred-Yval[:,task]
                Errors2_train[ind_alpha,fold,task] = np.linalg.norm(y_pred_train-Ytrain[:,task])/len(y_pred_train)
                MAPEs[ind_alpha,fold,task] =  MAPE(Yval[:,task],y_pred)
                VarsALL[ind_alpha,fold,:,task] = np.power(sigma,2) #+ alpha
                
                #Errors[ind_alpha,fold,task] = np.linalg.norm(y_pred-Yval[:,task])

                print('Noise parameter: '+  str(alpha)+    '//Task: ' + str(task+1) + '//Error cuad Val:  ' +str(Errors2[ind_alpha,fold,task]))
                [ic1,ic2] = EvaluateConfidenceIntervals(Yval[:,task],y_pred,Vpredicted_24[:,task])
                IC1[ind_alpha,fold,:,task] = ic1.ravel()
                IC2[ind_alpha,fold,:,task] = ic2.ravel()
                Val_All[ind_alpha,fold,:,task] = Yval[:,task]
                Val_pred_All[ind_alpha,fold,:,task] = y_pred
            end_train_time = time.time()
            training_time = end_train_time-start_train_time
            print('task:',task)           
    end_val_time = time.time()
    val_time = end_val_time - start_val_time                       
        
    Error2Validation = np.zeros((TaskNumber,1))
    AverageErrors2 = np.mean(Errors2,1)
    Alphas_Best  = np.zeros((TaskNumber))
    YPred_Best = np.zeros((NbOfTestSample,TaskNumber))
    Vpredicted_Best = np.zeros((NbOfTestSample,TaskNumber))
    Covpredicted_Best = np.zeros((TaskNumber,NbOfTestSample,NbOfTestSample))
    BestModels = [None]*TaskNumber
    
    
    YvalsOpt = np.zeros((NbOfValSamples*Nfold,TaskNumber))
    Y_predValsOpt = np.zeros((NbOfValSamples*Nfold,TaskNumber))
    
    start_test_time = time.time()
    for task in range(0,TaskNumber):

        idxmin = np.argmin(AverageErrors2[:,task])
        Error2Validation[task] = AverageErrors2[idxmin,task]
        Alphas_Best[task] = Alphas[idxmin]
        ErrorsValOpt[:,:,task] = np.squeeze(ErrorsVal[:,idxmin,:,task])

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha = Alphas_Best[task])
        gp.fit(Xtrain_all, Ytrain_all[:,task])
        NoiseParameters[task] = gp.kernel_.k2.noise_level
        y_pred, Ct = gp.predict(Xtest, return_cov=True)
        Covpredicted_Best[task,:,:] = Ct
        sigma = np.sqrt(np.diag(Ct))
        YPred_Best[:,task] = y_pred
        Vpredicted_Best[:,task] = np.power(sigma,2)# +  Alphas[idxmin] # + gp.kernel_.k2.noise_level
        BestModels[task] = gp
        YvalsOpt[:,task] =    np.ndarray.flatten(np.squeeze(Val_All[idxmin,:,:,task]),'C')
        Y_predValsOpt[:,task] = np.ndarray.flatten(np.squeeze(Val_pred_All[idxmin,:,:,task]),'C')
    end_test_time = time.time()  
    test_time = end_test_time-start_test_time
        
    return YPred_Best, Vpredicted_Best, BestModels, Alphas_Best, IC1, IC2, Errors2, R2s, MAPEs,VarsALL,Errors2_train,R2strain,Error2Validation,ErrorsValOpt,NoiseParameters,YvalsOpt,Y_predValsOpt,Covpredicted_Best,training_time,val_time,test_time