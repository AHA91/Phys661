#### Meri Khurshudayn
#### Assignment 5

import numpy as np

####################### Part I and II ##########################

def Differentiate(points, Type, h = 1):
     f = 'lambda x:' + input('Enter a function: ')
     f = eval(f)
     result = []
     
     if Type == "TPF":
          for i in points:
               y = (f(i+h)-f(i))/h
               result.append(y)

     if Type == "TPB":
          for i in points:
               y = (f(i)-f(i-h))/h
               result.append(y)

     if Type = "TPC":
          for i in points:
               y = (f(i+h)-f(i-h))/(2*h)
               result.append(y)

     return result
