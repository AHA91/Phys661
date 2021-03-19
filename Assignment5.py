#### Meri Khurshudayn
#### Assignment 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####################### Part I and II ##########################

def Differentiate(points, Type, h = 1, Input = True, f = 0):
     result = []
     if Input == True:
          f = 'lambda x:' + input('Enter a function: ')
          f = eval(f)
          
     if Type == "TPF":
          for i in points:
               y = (f(i+h)-f(i))/h
               result.append(y)

     if Type == "TPB":
          for i in points:
               y = (f(i)-f(i-h))/h
               result.append(y)

     if Type == "TPC":
          for i in points:
               y = (f(i+h)-f(i-h))/(2*h)
               result.append(y)

     return result

########################### Part III ##############################
def Part3():
     f_x = lambda x: np.sin(x) - np.cos(x)
     f_p_x = np.cos(0) + np.sin(0)

     h = np.linspace(1*10**(-1),1*10**(-12),25)

     table = np.zeros((len(h),3))
     table[:,0] = h
     z = []

     for i in h:
          n = Differentiate([0],"TPC", f = f_x, h = i, Input = False)
          z.append(n)
          
     z = np.array(z)
     table[:,1] = np.reshape(z,len(z))
     table[:,2] = abs(table[:,1] - f_p_x)

     table1 = pd.DataFrame(data = table, columns = ["Step Size","TPC","Error"])
     print(table1)
     
     plt.plot(h, table[:,2], "o-", color = "Pink")
     plt.xlim(1*10**(-1),0)
     plt.xlabel("Step Size")
     plt.ylabel("Error")
     plt.show()

######################### Part I and II ###########################
integralres = []

def Integral(low, up, num, Type, Input = True, f = 0):
     if Input == True:
          f = 'lambda x:' + input('Enter a function: ')
          f = eval(f)
          
     if Type == "LSM":
          x = (up-low)/num
          integral = 0
          integralres.append(integral)
          while low <= up:
               integral += (f(low)*x)
               integralres.append(integral)
               low += x
          
     if Type == "RSM":
          x = (up-low)/num
          right = low+x
          integral = 0
          integralres.append(integral)
          while right <= up:
               integral += (f(right)*x)
               integralres.append(integral)
               right += x
          integralres.append(integral)   

     if Type == "Trapezoid":
          x_del = (up-low)/num
          integral = f(low)*(x_del/2)
          integralres.append(integral)
          while low <= up-x_del:
               integral += 2*(f(low))*(x_del/2)
               integralres.append(integral)
               low += x_del
          integral += f(up)*(x_del/2)
          integralres.append(integral)
     
     if Type == "MPM":
          x = (up-low)/num
          mid = low+(x/2)
          integral = 0
          integralres.append(integral)
          while mid <= up:
               integral += (f(mid)*x)
               integralres.append(integral)
               mid += x

     return integral  

######################### Part III #################################

def Part33():
     table = np.zeros((26,4))
     f_x = lambda x: x**2
     lsm = Integral(0,10,25,"LSM", Input = False, f = f_x)
     table[:,0] = integralres
     integralres.clear()
     rsm = Integral(0,10,25,"RSM", Input = False, f = f_x)
     table[:,1] = integralres
     integralres.clear()
     tra = Integral(0,10,25,"Trapezoid", Input = False, f = f_x)
     table[:,2] = integralres
     integralres.clear()
     mpm = Integral(0,10,25,"MPM", Input = False, f = f_x)
     table[:,3] = integralres
     integralres.clear()
     table1 = pd.DataFrame(data = table, columns = ["LSM","RSM","Trapezoid","MPM"])
     return table1
