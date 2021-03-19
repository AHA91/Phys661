#### Meri Khurshudayn
#### Assignment 5

import numpy as np
import matplotlib.pyplot as plt

####################### Part I and II ##########################

def Differentiate(points, Type, h = 1, Input = False, f = 0):
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

     h = np.linspace(1*10**(-1),1*10**(-12),250)

     table = np.zeros((len(h),3))
     table[:,0] = h
     z = []

     for i in h:
          n = Differentiate([0],"TPC", f = f_x, h = i)
          z.append(n)
          
     z = np.array(z)
     table[:,1] = np.reshape(z,len(z))
     table[:,2] = abs(table[:,1] - f_p_x)

     print(table)
     
     plt.plot(h, table[:,2])
     plt.xlim(1*10**(-1),0)
     plt.show()

######################### Part I and II ###########################
def Integral(low, up, num, Type, Input = False, f = 0):
     if Input == True:
          f = 'lambda x:' + input('Enter a function: ')
          f = eval(f)
          
     if Type == "LSM":
          x = (up-low)/num
          integral = 0
          while low <= up:
               integral += (f(low)*x)
               low += x
          
     if Type == "RSM":
          x = (up-low)/num
          right = low+x
          integral = 0
          while right <= up:
               integral += (f(right)*x)
               right += x

     if Type == "Trapezoid":
          x_del = (up-low)/num
          integral = f(low) + f(up) #first and last are not 2x
          while low <= up-x_del:
               integral += 2*(f(low))
               low += x_del
          integral = (x_del/2*integral)

     
     if Type == "MPM":
          x = (up-low)/num
          mid = low+(x/2)
          integral = 0
          while mid <= up:
               integral += (f(mid)*x)
               mid += x
               
     return integral 
