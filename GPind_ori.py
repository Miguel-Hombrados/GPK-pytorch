# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:05:02 2021

@author: mahom
"""


from GPind_ori_aux1 import GPind_ori_aux1
from GPind_ori_aux2 import GPind_ori_aux2
def GPind_ori(x,y,n_tasks,kernel_type,opt_parameters):

      trainsize = opt_parameters['trainsize']  
      
      if int(trainsize) == 1:
          [MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood] = GPind_ori_aux2(x,y,n_tasks,kernel_type,opt_parameters)
      else:
          [MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood] = GPind_ori_aux1(x,y,n_tasks,kernel_type,opt_parameters)

 
      return MODELS,LIKELIHOODS,Results,Opt_model,Opt_likelihood