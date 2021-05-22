##### Meri Khurshudyan
##### Assignment4
import sympy as sp



###################### Part I with display option ########################
def FPI(init = 1, ittr = 30, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter Fixed Point Function: x = "))
     n = f.subs(x,init).evalf()
     while ittr != 0:
          if round(float(f.subs(x,n).evalf()-n),3) == 0.000:
               break
          n = f.subs(x,n)
          ittr = ittr-1
          if display == True:
               print("X:".ljust(2), '{:.4f}'.format(n),"|", "G(X):", '{:.4f}'.format(float(f.subs(x,n).evalf())),"|", "F(X):",\
                     '{:.4f}'.format(float(f.subs(x,n).evalf()-n)))
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
          if f_p == 0:
              print("Divide by zero encountered")
              break
          n = n-f_x/f_p
          f_p = f_prime.subs(x,n).evalf()
          f_x = float(f.subs(x,n).evalf())
          ittr = ittr - 1
          if display == True:
               print("X:".ljust(2), '{:.4f}'.format(n) , "|".ljust(2), "F(n): ",'{:.4f}'.format(f_x))
          if f_x == 0.000:
               break
     return format(n,".4f")


 
##################### Part III with display options #######################
def secant(x0, x1, ittr = 10, display = False):
     x = sp.symbols('x')
     f = eval(input("Enter A Function: 0 = "))
     while ittr != 0:
          if (f.subs(x,x1).evalf()-f.subs(x,x0).evalf()) == 0:
               print("Divide by zero encountered")
               break
          temp = x1
          x1 = x1 - (f.subs(x,x1).evalf()*((x1-x0)/\
                     (f.subs(x,x1).evalf()-f.subs(x,x0).evalf())))
          if f.subs(x,x1).evalf() == 0.000:
               break
          x0 = temp
          ittr = ittr - 1
          if display == True:
               print("X0:".ljust(2), '{:.4f}'.format(x0), "|", "X1:".ljust(2)\
                     ,'{:.4f}'.format(x1), "|","F(X1):".ljust(2), '{:.4f}'.format(float(f.subs(x,x1).evalf())))
          
     return format(x1,".4f")

############################### Print Statements ############################

#Part I
#print("X = ", FPI(display = False))

#Part II
#print("X = ", NPM(display = True))

#Part III
#print("X = ", secant(0,30, display = True))
