# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 06:28:06 2021

@author: mahom
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
def PlotDayLoad(LoadTrue,LoadPred,LoadStd):
    """

    Parameters
    ----------
    LoadTrue : Real Matrix (24 x D).
    Vector of 24 times D  real values representing the power load of 
    each hour of the day. D is the number of days plotted.
    
    LoadPred : Real Matrix (24 x D).
    Vector of 24 times D  real values representing the power load of 
    each hour of the day.
    
    LoadStd : Real Matrix (24 x D).
    Vector of 24 times D values representing the standard deviation of
    each hour of the day.


    Returns
    -------
    BufferData : plot

    """
    fig = plt.figure()
    if np.ndim(LoadTrue)>1:
        D = np.size(LoadTrue,0)
    else:
        D = 1
    Hours = np.array(range(0,24*D)).reshape(-1,1) 
    #plt.plot(Load, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    #plt.plot(Hours, LoadTrue.reshape(-1,1), 'r.-', markersize=10, label=u'Observations')
    plt.plot(Hours, LoadTrue.reshape(-1,1), 'r-', label=u'Observations')
    plt.plot(Hours, LoadPred.reshape(-1,1), 'b-', label=u'Prediction')
    plt.fill(np.concatenate([Hours, Hours[::-1]]),
             np.concatenate([LoadPred.reshape(-1,1) - 1.9600 * LoadStd.reshape(-1,1),
                           (LoadPred.reshape(-1,1) + 1.9600 * LoadStd.reshape(-1,1))[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% C.I.')
    plt.xlabel('$Hours$')
    plt.ylabel('$Power (MW)$')
    plt.legend(loc='upper left', framealpha=0.3)
    plt.grid(True)
    
    return fig