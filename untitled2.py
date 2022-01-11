# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:03:05 2022

@author: mahom
"""
samp = 30
plt.plot(YPredicted_24gp_ind[samp,:])
plt.plot(YPredicted_24gp_ind[samp,:]+torch.sqrt(VPredicted_24gp_ind[samp,:]))
plt.plot(YPredicted_24gp_ind[samp,:]-torch.sqrt(VPredicted_24gp_ind[samp,:]))
plt.plot(YTest_24[samp,:],'r')

samp = 100
plt.plot(VPredicted_24gp_ind[0:100,:].T)