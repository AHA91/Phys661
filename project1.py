# Meri Khurshudyan 
# Project1
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random as r
import statistics as s
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def KramerKronig():
     b = sp.Symbol("b")
     df = np.array(pd.read_csv("extinctionSpectrum.txt",delimiter = "\t"))
     n = [1+(2/np.pi)*sp.integrate((b*df[i,1])/((b**2)-df[i,0]**2),(b,0,i)) for i in range(0,1000,4)]
     plt.plot(range(0,1000,4),n)
     plt.show()
     
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

def twoDUniform(M,section):
     if section == 1:
          x = [np.cumsum(r.choices([-1,1],k=M)) for i in range(5)]
          y = [np.cumsum(r.choices([-1,1],k=M)) for i in range(5)]
          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(5):
               ax.plot(range(M),x[i],y[i])
          ax.set_ylabel("x")
          ax.set_zlabel("y")
          ax.set_xlabel("Number of Steps")
          plt.show()

     if section == 2:
          x = []
          y = []
          for i in range(5):
               theta = np.cumsum(r.choices([0,360],k=M))
               print(theta)
               x.append(5*np.cos(theta))
               y.append(5*np.sin(theta))

          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(5):
               ax.plot(range(M),x[i],y[i])
               ax.set_label(str(i))
          ax.set_ylabel("x")
          ax.set_zlabel("y")
          ax.set_xlabel("Number of Steps")
          plt.legend()
          plt.show()
          
          
          
     
