#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:29:35 2022

@author: Miguel

Description: Function that plots histograms of the standarized noise to for each 
tasks and global one.

true: N x T
predicted:  N x T
variance:  N x T

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
def histogram_errors(name,true,predicted,variance):
    n_tasks = np.size(true,1)
    
    true = true.detach().numpy()
    predicted = predicted.detach().numpy()
    variance = variance.detach().numpy()
    
    stddev = np.sqrt(variance)
    e = true-predicted
    enorm = np.divide(e,stddev)
    
    e0 = np.matrix.flatten(enorm)
    plt.hist(e0,density = True ,bins = 100,range=[-2,2])
    plt.title(name)
    plt.grid(True)
    x_pos = 1
    y_pos = 1
    values_k = "kurt:" + str("{:10.2f}".format(kurtosis(e0)))
    values_s = "Skew:" + str("{:10.2f}".format(skew(e0)))
    plt.text(x_pos,y_pos,values_k)
    plt.text(x_pos,y_pos-0.1,values_s)
    plt.savefig("histo"+".png",dpi=1000)
   
    #fig, ax = plt.subplots(nrows=4, ncols=n_tasks//4)
    # t = 4
    
    # for t in range(t,t+1):   
    #     e0 = torch.flatten(e[:,t])
    #     plt.hist(e0, density=True,bins = 30)
    #     namet = name + str(t)
    #     plt.savefig(namet+"_tasks.png")  
    # for row in ax:
    #     for col in row:
    #         col.hist(e0, density=True)
    #         col.set_title('Task '+str(t))
    #         #plt.grid(True)
    #         t = t + 1
    #         namet = name + str(t)
    #         plt.savefig(namet+"_tasks.png")
    # plt.tight_layout()
    
    # plt.savefig(name+"_tasks.png")
            
        
    