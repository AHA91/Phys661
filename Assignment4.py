##### Meri Khurshudyan
##### Assignment4
import sympy as sp



###################### Part I with display option ########################
def FPI(init = 1, ittr = 5, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter Fixed Point Function: x = "))
     n = f.subs(x,init).evalf()
     while ittr != 0:
          if display == True:
               print("X:".ljust(2), '{:.4f}'.format(n),"|", "G(X):", '{:.4f}'.format(f.subs(x,n).evalf()))
          n = f.subs(x,n)
          ittr = ittr-1
     return format(n,".4f")



###################### Part II with display option #######################
def NPM(init = 1, ittr = 5, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter A Function: 0 = "))
     f_prime = sp.diff(f,x) 
     n = float(init)
     f_p = f_prime.subs(x,init).evalf()
     f_x = f.subs(x,init).evalf()
     while ittr != 0:
          if display == True:
               print("X:".ljust(2), '{:.4f}'.format(n) , "|".ljust(2), "F(n): ",'{:.4f}'.format(f.subs(x,n).evalf()))
          n = n-f_x/f_p
          f_p = f_prime.subs(x,n).evalf()
          f_x = f.subs(x,n).evalf()
          ittr = ittr - 1
     return format(n,".4f")


 
##################### Part III with display options #######################
def secant(x0, x1, ittr = 5, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter A Function: 0 = "))
     while ittr != 0:
          if display == True:
               print("X0:".ljust(2), '{:.4f}'.format(x0), "|", "X1:".ljust(2)\
                     ,'{:.4f}.'.format(x1), "|","F(X1):".ljust(2), '{:.4f}'.format(f.subs(x,x1).evalf()))
          temp = x1
          x1 = x1 - (f.subs(x,x1).evalf()*(x1-x0)/\
                     (f.subs(x,x1).evalf()-f.subs(x,x0).evalf()))
          x0 = temp
          ittr = ittr - 1
     return format(x1,".4f")

############################### Print Functions ############################

#Part I
#print("X = ", FPI(display = True))

#Part II
#print("X = ", NPM(display = True))

#Part III
#print("X1 = ", secant(1,2, display = True))







     
