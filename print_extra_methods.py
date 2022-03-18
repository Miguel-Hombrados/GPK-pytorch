# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:57:40 2022

@author: mahom
"""
import torch
import numpy as np
from correcting_factor_cov import correcting_factor_cov
from correcting_factor_cov_gamma import correcting_factor_cov_gamma
from predictive_variance_white import predictive_variance_white
from EvaluateConfidenceIntervals_Laplace import EvaluateConfidenceIntervals_Laplace
from norm2laplace import norm2laplace
from EvaluateConfidenceIntervals import EvaluateConfidenceIntervals
from print_results_ic import print_results_ic
from to_torch import to_torch
def print_extra_methods(Stds_train_load,Ntest,Ntrain,WTrain,YTrain_24,YTest_24,XTrain_S,YPredicted_24gp_K,VPredicted_24gp_K,option_lv,scalerY_K,RESULTS,model,DATA):
    
    # Stds_train_load: 24 x 1
    # WTrain: 24 x K
    # YTrain_24: N x 24
    # XTrain_S: N x F
    # YPredicted_24gp_K: N x 24
    # VPredicted_24gp_K: N x 24
    # option_lv: option of latent variable in gpk
    
#==============================================================================
    Nsamples_factor = 1000
    ind_a = np.random.permutation(range(0,Ntrain))[0:Nsamples_factor] 

    S2norm = torch.pow(Stds_train_load,2)
    Snorm = Stds_train_load.T.repeat(Ntest,1)
    Snorm_tr  = Stds_train_load.T.repeat(Ntrain,1)
 
    YPredicted_24gp = (YPredicted_24gp_K@WTrain.T)*Snorm
    stdgp = torch.sqrt(torch.tensor(scalerY_K.var_)).float()
    # NMF validation variance error============================================
    if 'RD' in DATA:
        OptValVar_F_std = to_torch(DATA['RD']['OptValVar_F'])
        OptValVar_F_aux = OptValVar_F_std*stdgp
        OptValVar_F = torch.tensor([a*b for a,b in zip(OptValVar_F_aux,S2norm)])
        OptTrainVar_F_std = to_torch(DATA['RD']['OptTrainVar_F'])
        OptTrainVar_F_aux = OptTrainVar_F_std*stdgp
        OptTrainVar_F = torch.tensor([a*b for a,b in zip(OptTrainVar_F_aux,S2norm)])
        
    # GP predictive validation variance error============================================
    if 'ValidationPredictiveErrors' in RESULTS:
        ErrorValidation_P_std = torch.stack(RESULTS['ValidationPredictiveErrors'],dim =1)
        Nval = ErrorValidation_P_std.size(0)
        ErrorValidation_P_aux = ErrorValidation_P_std*stdgp
        Snorm_val = Stds_train_load.T.repeat(Nval,1)
        NoiseEstimation_Variance_GPpred  = torch.var((ErrorValidation_P_aux@WTrain.T)*Snorm_val,axis=0) 
    # GP  validation variance error============================================
    if 'ValidationErrors' in RESULTS:
        ErrorValidation_std = torch.stack(RESULTS['ValidationErrors'],dim =1)
        ErrorValidation_aux = ErrorValidation_std*stdgp
        NoiseEstimation_Variance_GPval  = torch.var((ErrorValidation_aux@WTrain.T)*Snorm_val,axis=0) 
    #==========================================================================
    #==========================================================================
    #==========================================================================
    #==========================================================================
    #WHITENED COV with GPpred=================================================================================
    if 'ValidationPredictiveErrors' in RESULTS:
        VPredicted_24gp_white = predictive_variance_white(VPredicted_24gp_K,WTrain,NoiseEstimation_Variance_GPpred,S2norm)
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp_white,"1-Whitened Covariance with GP pred")
    #Correcting factor with GPpred=======================================================
    if 'ValidationPredictiveErrors' in RESULTS:
        a = correcting_factor_cov(model,WTrain,YTrain_24[ind_a,:],XTrain_S[ind_a,:],option_lv,scalerY_K,NoiseEstimation_Variance_GPpred,Stds_train_load)
        VPredicted_24gp = torch.zeros((Ntest,24))
        for ss in range(0,Ntest):
            VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  NoiseEstimation_Variance_GPpred)*a
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp,"2- Correcting factor with GP pred")
        print("1/a: ", 1/a)
    #WHITENED COV with GPval=================================================================================
    if 'ValidationErrors' in RESULTS:
        VPredicted_24gp_white2 = predictive_variance_white(VPredicted_24gp_K,WTrain,NoiseEstimation_Variance_GPval,S2norm)
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp_white2,"3- Whitened Covariance with GP validation marginal likelihood")
    #Correcting factor with GPval=======================================================
    if 'ValidationErrors' in RESULTS:
        a = correcting_factor_cov(model,WTrain,YTrain_24[ind_a,:],XTrain_S[ind_a,:],option_lv,scalerY_K,NoiseEstimation_Variance_GPval,Stds_train_load)
        VPredicted_24gp = torch.zeros((Ntest,24))
        for ss in range(0,Ntest):
            VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  NoiseEstimation_Variance_GPval)*a
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp,"4- Correcting factor with GP val")
        print("1/a: ", 1/a)
    
    if 'RD' in DATA:
        #WHITENED COV with NMF validation error variance===========================================================
        VPredicted_24gp_white3 = predictive_variance_white(VPredicted_24gp_K,WTrain,OptValVar_F,S2norm)
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp_white3,"5-Whitened Covariance with NMF validation error variance")
        #Correcting factor with NMF validation error variance=======================================================
        a = correcting_factor_cov(model,WTrain,YTrain_24[ind_a,:],XTrain_S[ind_a,:],option_lv,scalerY_K,OptValVar_F,Stds_train_load)
        VPredicted_24gp = torch.zeros((Ntest,24))
        for ss in range(0,Ntest):
            VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  OptValVar_F)*a
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp,"6-Correcting factor with NMF validation error variance")
        print("1/a: ", 1/a)
    if 'RD' in DATA: 
        #WHITENED COV with NMF train error variance===========================================================
        VPredicted_24gp_white4 = predictive_variance_white(VPredicted_24gp_K,WTrain,OptTrainVar_F,S2norm)
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp_white4,"7-Whitened Covariance with NMF training error variance")
        #Correcting factor with NMF train error variance=======================================================
        a = correcting_factor_cov(model,WTrain,YTrain_24[ind_a,:],XTrain_S[ind_a,:],option_lv,scalerY_K,OptValVar_F,Stds_train_load)
        VPredicted_24gp = torch.zeros((Ntest,24))
        for ss in range(0,Ntest):
            VPredicted_24gp[ss,:] = (torch.diag(WTrain@torch.diag(VPredicted_24gp_K[ss,:])@WTrain.T)*(S2norm.ravel())  +  OptTrainVar_F)*a
        _,_,_  = print_results_ic(YPredicted_24gp,YTest_24, VPredicted_24gp,"8-Correcting factor with NMF training error variance")
        print("1/a: ", 1/a)
        
