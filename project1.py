# Meri Khurshudyan 
# Project1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import random as r
import statistics as s

def OneDWalk(M, section):
     M = M+1
     if section == 1:
          x_ave = [s.mean([sum(r.choices([-1,1],k=N)) for i in range(int(np.sqrt(N)))]) for N in range(1,M,1)]
          x2_ave = [np.average(np.square([sum(r.choices([-1,1],k=N)) for i in range(int(np.sqrt(N)))])) for N in range(1,M,1)]     
          t = "Integer -1 or 1"
     if section == 2:
          x_ave = [s.mean([sum(np.random.uniform(-1,1,N)) for i in range(int(np.sqrt(N)))]) for N in range(1,M,1)]
          x2_ave = [np.average(np.square([sum(np.random.uniform(-1,1,N)) for i in range(int(np.sqrt(N)))])) for N in range(1,M,1)]
          t = "Float between -1 and 1"
     if section == 3:
          x_ave = [s.mean([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(N)))]) for N in range(1,M,1)]
          x2_ave = [np.average(np.square([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(N)))])) for N in range(1,M,1)]
          t = "Normal Distribution"
     
     plt.plot(range(M),range(M), label = "Theoretical <x>^2")
     plt.plot(range(M),[0]*M,label = "Theoretical <x>")
     plt.plot(range(len(x_ave)),x_ave,label = "Experimental <x>")
     plt.plot(range(len(x2_ave)),x2_ave, label = "Experimental <x>^2")
     plt.title(t)
     plt.legend()
     plt.show()
          
          
          
          
          
     
