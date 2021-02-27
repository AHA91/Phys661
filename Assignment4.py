##### Meri Khurshudyan
##### Assignment4
import sympy as sp



###################### Part I with display option ########################
def FPI(init = 1, ittr = 5, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter Fixed Point Function:"))
     n = f.subs(x,init).evalf()
     while ittr != 0:
          if display == True:
               print("X:  ", round(n,3), "F(X):  ", round(f.subs(x,n).evalf(), 3))
          n = f.subs(x,n)
          ittr = ittr-1
     return n



###################### Part II with display option #######################
def NPM(init = 1, ittr = 5, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter A Function:"))
     f_prime = sp.diff(f,x) 
     n = float(init)
     f_p = f_prime.subs(x,init).evalf()
     f_x = f.subs(x,init).evalf()
     while ittr != 0:
          if display == True:
               print("X:  ",round(n,3),"F(n):  ",round(f.subs(x,n).evalf(),3))
          n = n-f_x/f_p
          f_p = f_prime.subs(x,n).evalf()
          f_x = f.subs(x,n).evalf()
          ittr = ittr - 1
     return n


 
##################### Part III with display options #######################
def secant(x0, x1, ittr = 5, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter A Function:"))
     while ittr != 0:
          if display == True:
               print("X0:  ", round(x0,3), "X1:  "\
                     ,round(x1,3), "F(X1):  ", round(f.subs(x,x1).evalf(),3))
          temp = x1
          x1 = x1 - (f.subs(x,x1).evalf()*(x1-x0)/\
                     (f.subs(x,x1).evalf()-f.subs(x,x0).evalf()))
          x0 = temp
          ittr = ittr - 1
     return x1

############################### Print Functions ############################

#Part I
#print("X = ", FPI(display = True))

#Part II
#print("X = ", NPM(display = True))

#Part III
#print("X1 = ", secant(1,2, display = True))
          














     
