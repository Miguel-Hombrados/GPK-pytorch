# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 01:28:42 2021

@author: mahom

Script for plotting examples of 24 day ahead load prediction
"""
import numpy as np
import sys
import matplotlib.pyplot as plt


project_path = 'C:/Users/mahom/Desktop/Paper_Load_Profiling_NMF_V4/'
path_save = project_path  + '/Results/'
sys.path.append(project_path+'/utils/')
sys.path.append('C:/Users/mahom/Documents/Python Scripts/UNM/utils')
from load_obj import load_obj
from PlotDayLoad import PlotDayLoad

LOCATIONS = ['ME','NH','VT','CT','RI','SEMASS','WCMASS','NEMASSBOST']
location = 2
methodGP = 'GPK' # GPt24  //GPK 
methodNMF = 'NMF' # NMF /Full/
method = methodGP+"_"+methodNMF
stdnmf = 'y'
EXPERIMENTO = 3

#RESULTS = load_obj(project_path+"Data/"+str(method)+'_'+str(methodGP)+'_'+'allK'+'allLocations')
RESULTS = load_obj(project_path +"Data/Exp_"+ str(EXPERIMENTO) + "/"+methodNMF+"/"+method+"_std_"+str(stdnmf)+"_allLocations")

VP = RESULTS['VPREDICTED_ALL'][location]
YP = RESULTS['YPREDICTED_ALL'][location]
YT = RESULTS['YTEST_ALL'][location]

TestSample = 7
NumberOfDaysToPlot = 7

LoadTrue = YT[TestSample:TestSample+NumberOfDaysToPlot,:]
LoadPred = YP[TestSample:TestSample+NumberOfDaysToPlot,:]
LoadStd = np.sqrt(VP[TestSample:TestSample+NumberOfDaysToPlot,:])

fig = PlotDayLoad(LoadTrue,LoadPred,LoadStd)

plt.savefig(path_save + 'GPKExample.png', dpi=300)
plt.show()