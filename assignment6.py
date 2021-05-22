import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def Part1(N):
     x = [random.uniform(-1,1) for x in range(N)] #x
     y = [random.uniform(-1,1) for x in range(N)] #y
     in_count = 0

     for i in range(len(x)):
          if np.sqrt(x[i]**2 + y[i]**2) > 1:
               plt.plot(x[i],y[i], "o", color = "magenta")
          else:
               plt.plot(x[i],y[i], "o", color = "GREEN")
               in_count += 1
     print("Calculated value of pi: ", 4*in_count / len(x))
     print("Error: ", abs((np.pi-(4*in_count / len(x))))/np.pi * 100)

     x_ = np.linspace(-1,1,1000)
     unit = np.sqrt(1 - x_**2)
     unit2 = -unit
     plt.plot(x_,unit, color = "blue")
     plt.plot(x_, unit2, color = "blue")
     plt.show()
     
def Part2(N):
     x = [random.uniform(0,1) for x in range(N)] #x
     y = [random.uniform(0,1) for x in range(N)] #y
     z = [random.uniform(0,1) for x in range(N)] #z
     in_count = 0
     ax = plt.axes(projection = '3d')
     u,v = np.mgrid[0:2*np.pi/4:20j, 0:np.pi/2:10j]
     ax.plot_surface(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), alpha = 0.2, color = "salmon")
     for i in range(len(x)):
               if np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) > 1:
                    ax.scatter(x[i],y[i],z[i], "o", color = "magenta")
               else:
                    plt.plot(x[i],y[i],z[i], "o", color = "GREEN")
                    in_count += 1
     print("Calculated value of pi: ", 4*in_count / len(x))
     print("Error: ", abs((np.pi-(4*in_count / len(x))))/np.pi * 100)
     plt.show()

def Part3(N0, time, lam, dt):
     copyT = time
     Ndecay = 0
     ndec = []
     N = []
     N.append(N0)

     # let 1 be decay and 0 be no decay where probability favors decay
     p_decay = lam*dt
     p_not = 1 - p_decay

     while time != 0:
          for i in range(N0): #will check all 100 nuclei per dt 
               w = random.choices([0,1], weights = (p_not,p_decay))
               if w == [1]: #there is decay
                    N0 -= 1
                    Ndecay += 1
                    ndec.append(Ndecay)
                    N.append(N0)
               else:
                    ndec.append(Ndecay)
                    N.append(N0)
          time -= dt

     t = np.linspace(0,copyT, len(N))          
     plt.plot(t, N, "o", color = "red")
     plt.show()





