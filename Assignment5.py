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
     f_x = lambda x: np.exp(x)
     f_p_x = 1
     #table
     table = pd.DataFrame(index = range(10), columns = ["H","TPF","Error"])

     table["H"] = [10**-10,10**-9,10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
     table["TPF"] = Differentiate([0],"TPF", np.array(table["H"]), Input = False, f = f_x)[0]
     table["Error"] = abs(np.array(table["TPF"])-1)
     
     h = np.linspace(10**-14,10**-7,200)
     j = np.linspace(10**-7,10**-1,200)

     n = abs(np.array(Differentiate([0],"TPF", h , Input = False, f = f_x)[0]) - 1)
     n3 =  abs(np.array(Differentiate([0],"TPF", j , Input = False, f = f_x)[0]) - 1)

     plt.loglog(h ,n , "-", color = "brown")
     plt.loglog(j,n3,"-", color = "brown")

     plt.xlabel("Step Size")
     plt.ylabel("Error")
     plt.show()
     
     return table
  

######################### Part I and II ###########################
inte = []
def Integral(low, up, num, Type, Input = True, f = 0):
     x = (up-low)/num
     if Input == True:
          f = 'lambda x:' + input('Enter a function: ')
          f = eval(f)
          
     if Type == "LSM":
          integral = 0
          while low <= up-x:
               integral += (f(low)*x)
               inte.append(integral)
               low += x
          
     if Type == "RSM":
          right = low+x
          integral = 0
          while right <= up:
               integral += (f(right)*x)
               inte.append(integral)
               right += x
 

     if Type == "Trapezoid":
          integral = 0
          while low <= up-x:
               integral += x*((f(low)+f(low+x))/2)
               inte.append(integral)
               low += x

     
     if Type == "MPM":
          mid = low+(x/2)
          integral = 0
          while mid <= up:
               integral += (f(mid)*x)
               inte.append(integral)
               mid += x

     return integral  

######################### Part III #################################

def Part33(num):
     f_x = lambda x: np.cos(x)
     table = pd.DataFrame(index = range(num), columns = ["LSM","RSM","Trapezoid","MPM"])
     Integral(0,1, num,"LSM",Input = False, f = f_x)
     table["LSM"] = inte
     inte.clear()
     Integral(0,1, num,"RSM",Input = False, f = f_x)
     table["RSM"] = inte
     inte.clear()
     Integral(0,1, num,"Trapezoid",Input = False, f = f_x)
     table["Trapezoid"] = inte
     inte.clear()
     Integral(0,1, num,"MPM",Input = False, f = f_x)
     table["MPM"] = inte
     inte.clear()
     return table
