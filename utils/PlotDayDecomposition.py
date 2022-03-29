# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 06:28:06 2021

@author: mahom
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import matplotlib.colors as mcolors
def PlotDayDecomposition(LoadTrue,Ws,H,Std):
    """

    Parameters
    ----------
    LoadTrue : Real Matrix (24 x D).
    Vector of 24 times D  real values representing the power load of 
    each hour of the day. D is the number of days plotted.


    Returns
    -------
    BufferData : plot

    """
    
    LoadTrue = LoadTrue.detach().numpy()
    Ws = Ws.detach().numpy()
    H = H.detach().numpy()
    Std = Std.detach().numpy()
    
    sample = 1
    W = Std*Ws

    if np.ndim(LoadTrue)>1:
        D = np.size(LoadTrue,0)
    else:
        D = 1
    load_d = LoadTrue[sample,:].reshape(-1,1)
    Hours = np.array(range(0,24)).reshape(-1,1) 
    K = np.size(W,1)
    F = np.size(W,0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure()
    fig.set_size_inches(6., 4.)
    plt.ylim(0,1.2*np.max(load_d))
    y = np.array(range(0,2200,200))
    ticks = [str(x) for x in y]
    plt.yticks(y, ticks)
    #plt.plot(Hours, load_d, 'ro-', label=u'Observations')
    Blues = ['cadetblue','darkviolet','cornflowerblue','ghostwhite','mediumpurple','darkblue','slategrey','midnightblue']
    hn = H[sample,:]
    wk_ = np.zeros(F).reshape(-1,1)
    base_w = np.zeros(F).reshape(-1,1)
    base_w_ = np.zeros(F).reshape(-1,1)
    for k in range(0,K):
        wk = hn[k]*W[:,k].reshape(-1,1)
        base_w = base_w + wk
        if k==K-1:
            plt.plot(Hours, base_w, 'navy',marker='o',linewidth = 4, markersize=10 ,label=u'Observations')
            #plt.plot(Hours, base_w, 'navy','o-', label=u'Observations')
        else:
            
            plt.plot(Hours, base_w, 'navy','o-', label=u'Observations')
        plt.fill(np.concatenate([Hours, Hours[::-1]]),
            np.concatenate([base_w_,base_w[::-1]]),
            alpha=.5, fc=Blues[k])
        base_w_ = base_w 
    for h in range(0,24):
        plt.vlines(x = h,ymin = 0,ymax = load_d[h],ls='--',colors ='k',linewidth = 1)
        
    plt.xlabel('$Hours$', fontsize=18)
    plt.ylabel('$Power (MW)$', fontsize=18)    
    
    plt.xticks([0,4,8,12,16,20,23], ["00:00", "04:00", "08:00", "12:00", "16:00","20:00","23:00"],fontsize = 12)
    plt.savefig("24power_nmf_"+str(sample)+".png",bbox_inches='tight',dpp = 1500)
    
    baseZ = np.zeros(F).reshape(-1,1)
    fig = plt.figure()
    fig.set_size_inches(6., 4.)
    plt.ylim(0,1.2*np.max(load_d))
    y = np.array(range(0,2200,200))
    ticks = [str(x) for x in y]
    plt.yticks(y, ticks)
    plt.plot(Hours, load_d, 'navy',marker='o',linewidth = 4, markersize=10 ,label=u'Observations')
    plt.fill(np.concatenate([Hours, Hours[::-1]]),
            np.concatenate([baseZ,load_d[::-1]]),
            alpha=.5, fc=Blues[k])
    for h in range(0,24):
        plt.vlines(x = h,ymin = 0,ymax = load_d[h],ls='--',colors ='k',linewidth = 1)
    plt.xlabel('$Hours$', fontsize=18)
    plt.ylabel('$Power (MW)$', fontsize=18)  
    
    plt.xticks([0,4,8,12,16,20,23], ["00:00", "04:00", "08:00", "12:00", "16:00","20:00","23:00"],fontsize = 12)
    plt.savefig("24power_"+str(sample)+".png",bbox_inches='tight',dpp = 1500)