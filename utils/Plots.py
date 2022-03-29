# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:44:02 2020

@author: mahom

Description: Tools for plotting the results of the experiments 
"""
import numpy as np
import sys
import matplotlib.pyplot as plt


project_path = 'C:/Users/mahom/Desktop/Paper_Load_Profiling_NMF_V4/'
path_save = project_path  + '/Results/'
sys.path.append(project_path+'/utils/')


#ALL_MAPE = AllMAPE[:,0:-1]
#ALL_R2 = AllR2[:,0:-1]
from load_obj import load_obj


LOCATIONS = ['ME','NH','VT','CT','RI','SEMASS','WCMASS','NEMASSBOST']
location = 2
#method = 'NMF_fixed_SigmaF'
methodGP = 'GPK' # GPt24  //GPK 
methodNMF = 'NMF' # NMF /Full/
method = methodGP+"_"+methodNMF
stdnmf = 'y'
EXPERIMENTO = 3
#RESULTS = load_obj(project_path+"Data/"+str(method)+'_'+str(methodGP)+'_'+'allK'+'allLocations')
RESULTS = load_obj(project_path +"Data/Exp_"+ str(EXPERIMENTO) + "/"+methodNMF+"/"+method+"_std_"+str(stdnmf)+"_allLocations")

ALL_R2 = RESULTS['ALL_R2']
ALL_MAPE = RESULTS['ALL_MAPE']

#ALL_R2 = ALL_R2[:,0:-1]
#ALL_MAPE = ALL_MAPE[:,0:-1]

Kmax = np.size(ALL_R2,1) +2
Ks = np.array(range(2,Kmax))

plt.figure()
plt.plot(Ks,ALL_MAPE[location,:].ravel(),'o-')
plt.xlabel('Number of latent components')
plt.ylabel('MAPE')
plt.grid()
plt.show()
plt.savefig(path_save + 'Validation_plot_'+str(method)+'_'+str(methodGP)+'_'+str(LOCATIONS[location]))

plt.figure()
plt.plot(Ks,ALL_R2[location,:].ravel(),'o-')
plt.xlabel('Number of latent components')
plt.ylabel('Rsquared')
plt.grid()


# PLOT ALL LOCATIONS ===========
Kopt = np.zeros((len(LOCATIONS)))
plt.figure()
for location in range(0,len(LOCATIONS)):
    plt.plot(Ks,ALL_MAPE[location,:].ravel(),'+-',label = LOCATIONS[location])
    plt.xlabel('Number of latent components')
    plt.ylabel('MAPE')
    minval = np.argmin(ALL_MAPE[location,:]) + 2
    Kopt[location] = minval
    val = np.min(ALL_MAPE[location,:])
    plt.plot(minval,val,'ro')
plt.grid()
plt.legend(loc="upper left")
plt.savefig(path_save + 'Validation_plot_MAPEs'+str(method)+'_'+str(methodGP)+'_allLocations.png')
plt.show()

    
plt.figure()
for location in range(0,len(LOCATIONS)):
    plt.plot(Ks,ALL_R2[location,:].ravel(),'+-',label = LOCATIONS[location])
    plt.xlabel('Number of latent components')
    plt.ylabel('Rsquare')
    maxval = np.argmax(ALL_R2[location,:]) +2
    val = np.max(ALL_R2[location,:])
    plt.plot(maxval,val,'ro')
plt.grid()
plt.legend(loc="upper left")
plt.savefig(path_save + 'Validation_plot_R2s'+str(method)+'_'+str(methodGP)+'_allLocations.png')
plt.show()
    
plt.savefig(path_save + 'Validation_plot_R2s'+str(method)+'_'+str(methodGP)+'_allLocations.png')
plt.show()




# PLOT TEST RESULTS =================================
from load_obj import load_obj
RESULTS = load_obj(project_path+"Data/"+str(method)+'_'+str(methodGP)+'_'+'allK'+'allLocations_test')
YTEST_ALL_test = RESULTS['YTEST_ALL_test']
YPREDICTED_ALL_test = RESULTS['YPREDICTED_ALL_test']
ALL_MAPE_test = RESULTS['ALL_MAPE_test']
ALL_R2_test = RESULTS['ALL_R2_test']
plt.figure()   
for location in range(0,len(LOCATIONS)):
    
    plt.subplot(811)
    plt.plot(X.T,'r', alpha=0.5)
    plt.subplot(821)



for location in range(0,len(LOCATIONS)):
    plt.plot(Ks,ALL_R2[location,:].ravel(),'+-',label = LOCATIONS[location])
    plt.xlabel('Number of latent components')
    plt.ylabel('Rsquare')
    maxval = np.argmax(ALL_R2[location,:]) +2
    val = np.max(ALL_R2[location,:])
    plt.plot(maxval,val,'ro')
plt.grid()
plt.legend(loc="upper left")
#plt.savefig(path_save + 'Validation_plot_R2s'+str(method)+'_'+str(methodGP)+'_allLocations.png')
plt.show()
    
plt.savefig(path_save + 'Validation_plot_R2s'+str(method)+'_'+str(methodGP)+'_allLocations.png')
plt.show()
