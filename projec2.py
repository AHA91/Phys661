import numpy as np
import matplotlib.pyplot as plt
def airres(Vx0, Vy0, b0):
     air_time = (Vy0/9.81)*2 #time spent in air
     t = np.linspace(0,air_time,100)
     dx = Vx0*t
     dy = Vy0*t-(9.81*t**2)/2
     #exponential
     b_ = b0*np.exp(-dy*0.5)
     airX = b_ * dx
     airY = b_ * dy
     #constant
     b = 0.1
     air_resX = b*dx
     air_resY = b*dy
     #graphs
     plt.plot(dx,dy, label = "No Resistance")
     plt.plot(dx-air_resX, dy-air_resY, label = "Constant Resistance")
     plt.plot(dx-airX, dy-airY, label = "Exponential")
     plt.legend()
     plt.show()
     
