# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:43:37 2022

@author: mahom
Script plot time methods
"""

import matplotlib.pyplot as plt
method_names = ['GP24','GPK1','GPK']
average_test_times = [1,3,4]
average_train_times = [4,5,6]



plt.scatter(average_train_times,average_test_times,s=100,color="red")
plt.xlabel("Train times (s)")
plt.ylabel("Test times (s)")
for i, label in enumerate(method_names):
    plt.annotate(label, (average_train_times[i], average_test_times[i]))
plt.grid()
plt.savefig("computing_times.png",bbox_inches='tight',dpp = 1500)

