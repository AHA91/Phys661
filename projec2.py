import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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


def gravity(r0, v0, a_r0, dt, t):
     
     r1 = r0+v0*dt+(a_r0*dt**2)/2
     r_min1 = 0
     r_pl1 = 2*r1 - r_min1 + a_r0*dt**2
     for i in range(0,t,dt):
          a_r = (r_pl1 - 2*r1 + r_min1)/dt
          v_pl1 = v0+((a_r0+a_r)/2)*dt
          r_pl1 = 2*r1 - r_min1 + a_r*dt**2
          r_min1 = r1
          r1 = r_pl1
          print(a_r)

def singleOscillator(k, v0, m, dt, t):
     x_ = []
     f = v0/dt
     x = -f/k
     x_.append(x)
     for i in range(0,t,dt):
          v_pl1 = v0 + f*dt
          a = (v_pl1 - v0)/dt
          x = -m*a/k
          v0 = v_pl1
          x_.append(x)
     plt.plot(range(0,t+dt,dt),x_)
     plt.show()

def vectorfield(w, t, p):
    x1, y1, x2, y2 = w
    m1, m2, k1, k2, L1, L2, b1, b2 = p

    f = [y1,
         (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1,
         y2,
         (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2]
    return f
     
def coupledOscillation():
     m1 = 1.0
     m2 = 1.5
     k1 = 8.0
     k2 = 40.0
     L1 = 0.5
     L2 = 1.0
     b1 = 0
     b2 = 0

     
     x1 = 0.5
     y1 = 0.0
     x2 = 2.25
     y2 = 0.0

     abserr = 1.0e-8
     relerr = 1.0e-6
     stoptime = 10.0
     numpoints = 250

     t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]


     p = [m1, m2, k1, k2, L1, L2, b1, b2]
     w0 = [x1, y1, x2, y2]

     wsol = odeint(vectorfield, w0, t, args=(p,),atol=abserr, rtol=relerr)
     plt.plot(t,wsol[2])
     plt.plot(t,wsol[2])
     plt.show()

