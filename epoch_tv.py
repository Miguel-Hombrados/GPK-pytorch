# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:49:14 2022

@author: mahom
"""
import torch
import gpytorch
def train_epoch(model,data,mll,optimizer):
    train_loss=0.0
    model.train()
#    for images, labels in dataloader:

    #train_x,train_y = images.to(device),labels.to(device)
    (train_x,train_y) = data 
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * train_x.size(0)

    return train_loss,output

def valid_epoch(model,likelihood,output,data,mll):
   valid_loss = 0.0
   (val_x,val_y) = data 
   likelihood(val_y)
   #valid_loss+=loss.item()*val_x.size(0)

   return valid_loss