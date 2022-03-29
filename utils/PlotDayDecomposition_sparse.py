# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 06:28:06 2021

@author: mahom
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import matplotlib.colors as mcolors
def PlotDayDecomposition_sparse(LoadTrue,Ws,H,Std):
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
    
    sample = 0
    W = Std*Ws
    #W = Ws
    if np.ndim(LoadTrue)>1:
        D = np.size(LoadTrue,0)
    else:
        D = 1
    load_d = LoadTrue[sample,:].reshape(-1,1)
    Blues = ['cadetblue','darkviolet','cornflowerblue','ghostwhite','mediumpurple','darkblue','slategrey','midnightblue']
    hn = H[sample,:]
    hn_dense_index = hn.nonzero()[0][:]
    hn_dense = hn[hn_dense_index]
    nzK = np.count_nonzero(hn)
    Hours = np.array(range(0,24)).reshape(-1,1) 
    K = np.size(W,1)
    F = np.size(W,0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure()
    fig.set_size_inches(6., 4.)
    plt.ylim(0,1*np.max(load_d))
    y = np.array(range(0,int(np.max(load_d)),200))
    ticks = [str(x) for x in y]
    plt.yticks(y, ticks)
    #plt.plot(Hours, load_d, 'ro-', label=u'Observations')
    wk_ = np.zeros(F).reshape(-1,1)
    base_w = np.zeros(F).reshape(-1,1)
    base_w_ = np.zeros(F).reshape(-1,1)
    it = 0
    legend4 = ['seasonality','weekday','year','latent']
    legend5 = ['seasonality','weekday','year','holiday','latent']
    handles = []
    if nzK ==4:
        for k in range(0,5):
            wk = hn_dense[it]*W[:,it].reshape(-1,1)
            base_w = base_w + wk
            if k==4:
                plt.plot(Hours, base_w, 'navy',marker='o',linewidth = 4, markersize=10 ,label=u'Observations')
                plt.plot(Hours, base_w_, 'navy',label=u'Observations')
                plt.fill(np.concatenate([Hours, Hours[::-1]]),
                np.concatenate([base_w_,base_w[::-1]]),
                alpha=.5, fc=Blues[k])
                h = plt.scatter(1.,base_w_[1]+5.,c = Blues[k])
                handles.append(h)
                #plt.plot(Hours, base_w, 'navy','o-', label=u'Observations')
            else:
                if k!=3:
                    it = it + 1
                    plt.plot(Hours, base_w, 'navy','o-', label=u'Observations')
                    plt.fill(np.concatenate([Hours, Hours[::-1]]),
                    np.concatenate([base_w_,base_w[::-1]]),
                    alpha=.5, fc=Blues[k])
                    h = plt.scatter(1.,base_w_[1]+5.,c = Blues[k])
                    handles.append(h)
                    base_w_ = base_w 
        plt.legend(handles,legend4)           
    if nzK ==5:
        for k in range(0,5):
            wk = hn_dense[k]*W[:,k].reshape(-1,1)
            base_w = base_w + wk
            if k==4:
                plt.plot(Hours, base_w, 'navy',marker='o',linewidth = 4, markersize=10 ,label=u'Observations')
                #plt.plot(Hours, base_w, 'navy','o-', label=u'Observations')
            else:
                plt.plot(Hours, base_w, 'navy','o-', label=u'Observations')
                plt.fill(np.concatenate([Hours, Hours[::-1]]),
                np.concatenate([base_w_,base_w[::-1]]),
                alpha=.5, fc=Blues[k])
            base_w_ = base_w 
    for h in range(0,24):
        plt.vlines(x = h,ymin = 0,ymax = base_w[h],ls='--',colors ='k',linewidth = 1)
    
    plt.xlabel('$Hours$', fontsize=18)
    plt.ylabel('$Power (MW)$', fontsize=18)    
    
    plt.xticks([0,4,8,12,16,20,23], ["00:00", "04:00", "08:00", "12:00", "16:00","20:00","23:00"],fontsize = 12)
    plt.savefig("24power_nmf_"+str(sample)+".png",bbox_inches='tight',dpp = 1500)
    
    baseZ = np.zeros(F).reshape(-1,1)
    fig = plt.figure()
    fig.set_size_inches(6., 4.)
    plt.ylim(0,np.max(load_d))
    y = np.array(range(0,int(np.max(load_d)),200))
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