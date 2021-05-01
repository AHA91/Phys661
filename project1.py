import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import random as r

def OneDWalk(N, section):
     displ = None
     sqrtNDispl = []
     if section == 1:
          for i in range(int(np.sqrt(N))):
               displ = np.cumsum(np.array(r.choices([-1,1],k=N)))#cumulitive displacement
               sqrtNDispl.append(displ[N-1])

     if section == 2:
          for i in range(int(np.sqrt(N))):
               displ = np.cumsum(np.array(np.random.uniform(-1,1.00000000000002,N)))
               sqrtNDispl.append(displ[N-1])
               
     if section == 3:
          for i in range(int(np.sqrt(N))):
               displ = np.cumsum(np.array(np.random.normal(size = N)))
               sqrtNDdispl.append(displ[N-1])
          

     x_aver = displ
     x2_aver = np.square(displ)
     plt.plot(x_aver,range(N), label = "X-Average", color = "Magenta")
     plt.plot(x2_aver,range(N),label = "X^2-Average", color = "Blue")
     plt.xlabel("N")
     plt.ylabel("Displacement")
     plt.legend()
     plt.show()
          
          
          
          
          
     
