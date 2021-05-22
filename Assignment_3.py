###### Meri Khurshudyan 
###### Assignment 3

import numpy as np
############################# Part I ################################

# user must input a function in numpy format
y = eval("lambda x:" + input("Enter a function:"))

def bisector01(f,a,b, tol = 10**(-10)):
    if f(a)*f(b) > 0:
        print("Error")  
    while (b-a) > tol:
        m = (a+b)/2
        if (f(m)*f(a))>0:
            a = m
        else:
            b = m
    return (a+b)/2

############################# Part II ################################
def decimalP():
    _D_ = input("Enter Decimal Places or Def to use default:")
    if _D_ == "Def":
        _d_ = 6
    else:
        _d_ = int(_D_)
    return _d_

def bisector02(a,b,tol = 10**(-10), d = 6):
    zero = bisector01(y,a,b,tol)
    return round(zero,d)
############################ Part III #################################

def tolP():
    _T_ = input("Enter tolerance value or Def to use default:")
    if _T_ == "Def":
        t = 10**(-10)
    else:
        t = eval(_T_)
    return t


############################ Print Functions ###########################
#Part I
#print(bisector01(y,0,1))

#Part II
#print(bisector02(0,1))

#Part III
#print(bisector02(0,1,tolP(),decimalP()))
