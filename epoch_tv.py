# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:49:14 2022

@author: mahom
"""
import torch
import gpytorch
from predGPind_ori import predGPind_ori
def train_epoch(model,data,mll,optimizer):
    train_loss=0.0
    model.train()

    (train_x,train_y) = data 
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

    train_loss = loss.item() * train_x.size(0)

    return train_loss,output

def valid_epoch(model,likelihood,output,data,mll):
    
   (val_x,val_y) = data 
   f_val_est = model.forward(val_x)
   #y_val_est = likelihood(f_val_est)
   loss = -mll(f_val_est,val_y)
   valid_loss=loss.item()*val_x.size(0)
   
   y_val_est  = likelihood(f_val_est)
   valid_error = val_y-y_val_est.mean
   #===predictive distribution=================================
   model.eval()
   likelihood.eval()
   with torch.no_grad(): #, gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(val_x))
        ypred_val = predictions.mean
        val_pred_error = ypred_val - val_y

   return valid_loss,valid_error,val_pred_error