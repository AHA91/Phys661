import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import dblquad as di
import sympy as sp
import random as r
import statistics as s
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera



def KramerKronig():
     df = np.array(pd.read_csv("extinctionSpectrum.txt",delimiter = "\t",header = None))
     f = df[:,0]/(3*10**8)
     n_ = []
     o_ = []
     n = 0
     for m in range(1000):
          for i in range(999):
               if f[i] != f[m]:
                    n = n + ((f[i]*df[i,1])/((f[i]**2)-(f[m]**2)))*(f[i+1]-f[i])
               else:
                    n = n + 0
          o = 1 + (2/np.pi)*n
          o_.append(o)
          n = 0
     plt.plot(df[:,0],o_, label = "Real", color = "crimson")
     plt.plot(df[:,0], df[:,1], label = "Imaginary", color = "turquoise")
     plt.xlabel("Wavelength")
     plt.ylabel("Spectral Amplitude")
     plt.legend()
     plt.show()

def gauss(x_,y_,z_,s):
     eo = 8.854*10**(-12)
     q = 1.60217*10**(-19)
     
     df = pd.DataFrame(columns = ["Side","Flux"])
     #top
     Er = lambda y, x: ((s/2-z_))/((s/2 - z_)**2 + (y-y_)**2 +(x-x_)**2)**(3/2)
     flux = (q/(4*np.pi*eo))*di(Er,-s/2,s/2, lambda x: -s/2, lambda x: s/2)[0]
     df = df.append({"Side":"Top","Flux":flux},ignore_index = True)
     #bottom
     Er = lambda y, x: (abs(-s/2-z_))/((-s/2 - z_)**2 + (y-y_)**2 +(x-x_)**2)**(3/2)
     flux = (q/(4*np.pi*eo))*di(Er,-s/2,s/2, lambda x: -s/2, lambda x: s/2)[0]
     df = df.append({"Side":"Bottom","Flux":flux},ignore_index = True)
     #left
     Er = lambda y, z: ((s/2-x_))/((z-z_)**2 + (y-y_)**2 +(s/2-x_)**2)**(3/2)
     flux = (q/(4*np.pi*eo))*di(Er,-s/2,s/2, lambda z: -s/2, lambda z: s/2)[0]
     df = df.append({"Side":"Left","Flux":flux},ignore_index = True)
     #right
     Er = lambda y, z: (abs(-s/2-x_))/((-z-z_)**2 + (y-y_)**2 +(-s/2-x_)**2)**(3/2)
     flux = (q/(4*np.pi*eo))*di(Er,-s/2,s/2, lambda z: -s/2, lambda z: s/2)[0]
     df = df.append({"Side":"Right","Flux":flux},ignore_index = True)
     #front
     Er = lambda x, z: ((s/2-y_))/((z-z_)**2 + (s/2 - y_)**2 + (x-x_)**2)**(3/2)
     flux = (q/(4*np.pi*eo))*di(Er,-s/2,s/2, lambda z: -s/2, lambda z: s/2)[0]
     df = df.append({"Side":"Front","Flux":flux},ignore_index = True)
     #back
     Er = lambda x, z: (abs(-s/2-y_))/((z-z_)**2 + (-s/2 - y_)**2 + (x-x_)**2)**(3/2)
     flux = (q/(4*np.pi*eo))*di(Er,-s/2,s/2, lambda z: -s/2, lambda z: s/2)[0]
     df = df.append({"Side":"Back","Flux":flux},ignore_index = True)
     print("Expected: ", q/eo)
     print("Total Flux: ", sum(df["Flux"]))
     print(df)
     #cube
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1, projection='3d')
     ax.plot([s/2]*s,np.linspace(-s/2,s/2,s),[s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot(np.linspace(-s/2,s/2,s),[s/2]*s,[s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot([s/2]*s,[s/2]*s,np.linspace(-s/2,s/2,s),linewidth = 3, color = "cornflowerblue")
     ax.plot([-s/2]*s,np.linspace(-s/2,s/2,s),[-s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot(np.linspace(-s/2,s/2,s),[-s/2]*s,[-s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot([-s/2]*s,[-s/2]*s,np.linspace(-s/2,s/2,s),linewidth = 3, color = "cornflowerblue")
     ax.plot([s/2]*s,np.linspace(-s/2,s/2,s),[-s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot(np.linspace(-s/2,s/2,s),[s/2]*s,[-s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot([-s/2]*s,[s/2]*s,np.linspace(-s/2,s/2,s),linewidth = 3, color = "cornflowerblue")
     ax.plot([-s/2]*s,np.linspace(-s/2,s/2,s),[s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot(np.linspace(-s/2,s/2,s),[-s/2]*s,[s/2]*s,linewidth = 3, color = "cornflowerblue")
     ax.plot([s/2]*s,[-s/2]*s,np.linspace(-s/2,s/2,s),linewidth = 3, color = "cornflowerblue")
     ax.plot([x_,(s+1)/2],[y_,s/2],[z_,-s/7],linewidth = 3, color = "magenta")
     ax.plot([x_,(s+1)/2],[y_,-s/3],[z_,s/4],linewidth = 3, color = "magenta")
     ax.plot([x_,(s+1)/2],[y_,s/7],[z_,-s/2.5],linewidth = 3, color = "magenta")
     ax.plot(x_,y_,z_,"o", markersize = 30, color = "magenta")
     ax.plot([x_,0],[y_,(s+1)/2],[z_,-s/4],linewidth = 3, color = "magenta")
     ax.plot([x_,-s/3],[y_,(s+1)/2],[z_,s/4],linewidth = 3, color = "magenta")
     ax.plot([x_,s/7],[y_,(s+1)/2],[z_,s/2.5],linewidth = 3, color = "magenta")
     ax.plot([x_,0],[y_,-s/4],[z_,(s+1)/2],linewidth = 3, color = "magenta")
     ax.plot([x_,-s/3],[y_,s/2],[z_,(s+1)/2],linewidth = 3, color = "magenta")
     ax.plot([x_,s/7],[y_,0],[z_,(s+1)/2],linewidth = 3, color = "magenta")
     ax.plot([x_,-s/2],[y_,-s/4],[z_,-s/3],linewidth = 3, color = "magenta")
     ax.plot([x_,s/2],[y_,-s/4],[z_,-s/3],linewidth = 3, color = "magenta")
     ax.plot([x_,0],[y_,s/6],[z_,-(s+1)/2],linewidth = 3, color = "magenta")
     ax.plot([x_,0],[y_,-(s+1)/2],[z_,s/3],linewidth = 3, color = "magenta")
     ax.plot([x_,0],[y_,-(s+1)/2],[z_,-s/3],linewidth = 3, color = "magenta")
     ax.set_xlabel("x")
     ax.set_ylabel("y")
     ax.set_zlabel("z")
     plt.show()
               

     
def OneDWalk(M, H, section):
     M = M+1
     
     if section == 1:
          x_ave = [s.mean([sum(r.choices([-1,1],k=N)) for i in range(int(np.sqrt(N)))]) for N in range(1,M,H)]      
          x2_ave = [s.mean([(sum(r.choices([-1,1],k=N)))**2 for i in range(int(np.sqrt(N)))]) for N in range(1,M,H)]     
          t = "Integer -1 or 1"
     if section == 2:
          x_ave = [s.mean([sum(np.random.uniform(-1,1,N)) for i in range(int(np.sqrt(N)))]) for N in range(1,M,H)]
          x2_ave = [np.average(np.square([sum(np.random.uniform(-1,1,N)) for i in range(int(np.sqrt(N)))])) for N in range(1,M,H)]
          t = "Float between -1 and 1"
     if section == 3:
          x_ave = [s.mean([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(N)))]) for N in range(1,M,H)]
          x2_ave = [np.average(np.square([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(N)))])) for N in range(1,M,H)]
          t = "Normal Distribution"
     
     #plt.plot(range(1,M,H),range(1,M,H), label = "Theoretical <x>^2")
     #plt.plot(range(1,M,H),[0]*(int(M/H)-1),label = "Theoretical <x>")
     plt.plot(range(1,M,H),x_ave,label = "Experimental <x>")
     plt.plot(range(1,M,H),x2_ave, label = "Experimental <x>^2")
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
          for i in range(1):
               theta = np.random.uniform(0,2*np.pi,M)
               x.append(5*np.cos(theta))
               y.append(5*np.sin(theta))
          

          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(1):
               ax.plot(range(M),x[i],y[i], label = str(i+1))
          ax.set_ylabel("x")
          ax.set_zlabel("y")
          ax.set_xlabel("Number of Steps")
          plt.legend()
          plt.show()

     if section == 3:
          x1 = [np.cumsum(np.random.uniform([-1,1],k=M)) for i in range(5)]
          y1 = [np.cumsum(r.choices([-1,1],k=M)) for i in range(5)]
                
          x = []
          y = []
          for i in range(5):
               theta = np.cumsum(r.choices([0,360],k=M))
               x.append(5*np.cos(theta))
               y.append(5*np.sin(theta))
          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(5):
               ax.plot(range(M),x[i],y[i], label = "Polar " +str(i+1))
               ax.plot(range(M), x1[i], y1[i], label = "Cartesian " + str(i+1))
          ax.set_ylabel("x")
          ax.set_zlabel("y")
          ax.set_xlabel("Number of Steps")
          plt.legend()
          plt.show()
          
     if section == 4:
          x1 = np.array([s.mean([sum(r.choices([-1,1],k=M)) for i in range(int(np.sqrt(M)))]) for N in range(5)])
          y1 = np.array([s.mean([sum(r.choices([-1,1],k=M)) for i in range(int(np.sqrt(M)))]) for N in range(5)])

          x2_avecar = [np.average(np.square([sum(r.choices([-1,1],k=M)) for i in range(int(np.sqrt(M)))])) for N in range(5)]
          y2_avecar = [np.average(np.square([sum(r.choices([-1,1],k=M)) for i in range(5)])) for N in range(5)]
     
          #ok
          f = [np.sqrt(x1[i]**2 + y1[i]**2) for i in range(len(x1))]
          f1 = [np.sqrt(x2_avecar[i]**2 + y2_avecar[i]**2) for i in range(len(x2_avecar))]
          plt.plot(range(len(x1)),f, label = "<r>")
          plt.plot(range(len(x2_avecar)),f1,label = "<r^2>")
          plt.legend()
          plt.title("2D Cartesian")
          plt.xlabel("N")
          
          plt.show()
          
     if section == 5:
          x1 = np.array([s.mean([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(M)))]) for N in range(5)])
          y1 = np.array([s.mean([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(M)))]) for N in range(5)])

          x2_avecar = [np.average(np.square([sum(np.random.normal(size=N)) for i in range(int(np.sqrt(M)))])) for N in range(5)]
          y2_avecar = [np.average(np.square([sum(np.random.normal(size=N)) for i in range(5)])) for N in range(5)]
     
          #ok
          f = [np.sqrt(x1[i]**2 + y1[i]**2) for i in range(len(x1))]
          f1 = [np.sqrt(x2_avecar[i]**2 + y2_avecar[i]**2) for i in range(len(x2_avecar))]
          plt.plot(range(len(x1)),f, label = "<r>")
          plt.plot(range(len(x2_avecar)),f1,label = "<r^2>")
          plt.legend()
          plt.title("Normally Distributed")
          plt.xlabel("N")
          
          plt.show()


def brownian(Time,dt,k,section):
     if section == 1:
          x2_ave = np.array([(Time*k_/dt)*s.mean([((1/np.sqrt(2))*(sum(r.choices([-1,1],k=int(Time*k_/dt)))))**2 \
                                                  for i in range(int(np.sqrt(Time*k_/dt)))]) for k_ in np.linspace(1,k,100)])     
          #It is proportional to n**2
          o = np.array([(Time*k_/dt)**2 for k_ in np.linspace(1,k,100)])
     
          plt.plot(o,x2_ave)
          plt.xlabel("N = T*k/dt")
          plt.ylabel("N * <x^2>")
          plt.title("Binary")
          plt.plot(o,o, label = "Theoretical")
          plt.show()
          
     if section == 2:
          x2_ave = np.array([(Time*k_/dt)*s.mean([((sum(np.random.normal(size=int(Time*k_/dt)))))**2 \
                                                  for i in range(int(np.sqrt(Time*k_/dt)))]) for k_ in np.linspace(1,k,100)])     
          #It is proportional to n**2
          o = np.array([(Time*k_/dt) for k_ in np.linspace(1,k,100)])
     
          plt.plot(o,x2_ave)
          plt.xlabel("N = T*k/dt")
          plt.ylabel("N * <x^2>")
          plt.title("Binary")
          plt.plot(o,o, label = "Theoretical")
          plt.show()
          
     if section == 3:
          al = []
          r2_ave = []
          N_ = int(Time*k/dt)
          for k_ in np.linspace(1,k,10):# number of points 
               N = int(Time*k_/dt)
               gg = np.zeros((N,3),dtype = float)
               for h in range(int(np.sqrt(N))):# number of trails
                    for i in range(1,N): #number of steps
                         ran = r.choices([1,2,3])
                         if ran == [1]:
                              x_ = np.random.normal(size = 1)
                              gg[i,0] = x_
                         if ran == [2]:
                              y_ = np.random.normal(size = 1)
                              gg[i,1] = y_
                         if ran == [3]:
                              z_ = np.random.normal(size = 1)
                              gg[i,2] = z_
                    r_ = sum(np.sqrt(gg[:,0]**2 + gg[:,1]**2 + gg[:,2]**2))
                    al.append(r_)
                    gg = np.zeros((N,3),dtype = float)
               al = np.square(np.array(al))
               b = N*s.mean(al)
               r2_ave.append(b)
               al = [] 
          plt.plot(np.linspace(0,N,len(r2_ave)),r2_ave)
          plt.ylabel("N*<r^2>")
          plt.xlabel("N")
          plt.show()
          
     if section == 4:
          gg = np.zeros((100,3),dtype = float)
          al = []
          for j in range(50):
               for i in range(100):
                    ran = r.choices([1,2,3])
                    if ran == [1]:
                         x_ = r.choices([-1/np.sqrt(2),1/np.sqrt(2)])
                         gg[i,0] = x_[0]
                    if ran == [2]:
                         y_ = r.choices([-1/np.sqrt(2),1/np.sqrt(2)])
                         gg[i,1] = y_[0]
                    if ran == [3]:
                         z_ = r.choices([-1/np.sqrt(2),1/np.sqrt(2)])
                         gg[i,2] = z_[0]
               a = np.cumsum(gg,axis = 0)
               al.append(a)
               gg = np.zeros((100,3),dtype = float)
          fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
          camera = Camera(fig)
          
          for i in range(len(a[:,0])):
               for j in range(len(al)):
                    an = al[j]
                    ax.plot(an[i,0],an[i,1],an[i,2],"o")
               camera.snap()
          anim = camera.animate()
          plt.show()
          

                    
          
     
     
          
     
     
     

   
       
