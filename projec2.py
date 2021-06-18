import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as tr
import matplotlib.animation as an
from celluloid import Camera
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.lines import Line2D

def airres(Vx0, Vy0, b0, beta):
     air_time = (Vy0/9.81)*2
     t = np.linspace(0,air_time,100000)
     dx = Vx0*t
     dy = Vy0*t-(9.81*t**2)/2
     
     #exp
     y_= 0
     x_ = 0
     Vx = Vx0
     Vy = Vy0
     yy = []
     xx = []
     
     for t_ in t:
          b_ = b0*np.exp(-y_*beta) 
          A = 9.81+b_
          Vx = Vx - b_*t_     
          x_ += Vx*t_
          Vy = Vy - A*t_
          y_ += Vy*t_-(A*t_**2)/2
          if y_ >= 0:
               yy.append(y_)
               xx.append(x_)
          else:
               break
       
     #constant

     b = .5
     air_resA = 9.81+b # new acceleration
     air_t = (Vy0/air_resA)*2 
     t = np.linspace(0,air_t,100)
     x = Vx0*t
     y = Vy0*t-(air_resA*t**2)/2
     
     #graphs

     plt.plot(dx,dy, label = "No Resistance")
     plt.plot(x,y, label = "Constant Resistance")
     plt.plot(xx, yy, label = "Exponential")
     plt.legend()
     plt.show()

def Gravity(x0,y0,vx0,vy0,m1,m2,G,t_,dt):
    t = np.arange(0,t_,dt)

    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    ax = np.zeros(len(t))
    ay = np.zeros(len(t))
    L = np.zeros(len(t))
    E = np.zeros(len(t))

    x[0] = x0
    y[0] = y0
    vx[0] = vx0
    vy[0] = vy0
    ax[0] = -x0/(np.sqrt(x0**2+y0**2))**3
    ay[0] = -y0/(np.sqrt(x0**2+y0**2))**3

    for i in range(len(t)-1):
        x[i+1] = x[i] + vx[i]*dt + (0.5)*ax[i]
        y[i+1] = y[i] + vy[i]*dt + (0.5)*ay[i]

        ax[i+1] = -m1*G*(x[i+1])/(np.sqrt(x0**2+y0**2))**3
        ay[i+1] = -m1*G*(y[i+1])/(np.sqrt(x0**2+y0**2))**3

        vx[i+1] = vx[i] + (0.5)*(ax[i]+ax[i+1])*dt
        vy[i+1] = vy[i] + (0.5)*(ay[i]+ay[i+1])*dt

    L = vx*np.sqrt(x**2+y**2)
    E = (1/2)*m1*(np.sqrt(vx**2+vy**2))**2-G*m1*m2/(np.sqrt(x**2+y**2))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(x,y)
    ax1.set_title('Orbit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.plot(t,L)
    ax2.set_xlabel('time')
    ax2.set_ylabel('Angular Momentum')

    ax3.plot(t,E)
    ax3.set_xlabel('time')
    ax3.set_ylabel('Total Energy')


    plt.show()
               
     

def singleOscillator(k, v0, x0, m, dt, t, part):
     nm = x0
     v0_ = v0
     x0_ = x0
     x = np.array([])
     v = np.array([])
     x_ = np.array([])
     v_ = np.array([])
     ran = int(t/dt)
     for i in range(ran):
          #semi-implicit
          a_ = -(k*x0_)/m
          vi_ = v0_ + a_*dt
          xi_ = x0_ + vi_*dt
          #Euler
          a = -(k*x0)/m
          xi = x0 + v0*dt
          #print(xi)
          vi = v0 + a*dt
          v = np.append(v,v0)
          x = np.append(x,x0)
          v_ = np.append(v_, v0_)
          x_ = np.append(x_, x0_)
          x0_ = xi_
          v0_ = vi_
          x0 = xi
          v0 = vi
     if part == 1:
          t = np.arange(0,t,dt)
          fig , (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
          ax1.plot(t,x, "-", color = "blue", label = "Euler Method")
          ax1.set_ylabel("Position")
          ax1.legend()
          ax2.plot(t, v, "-", color = "blue")
          ax2.set_ylabel("Velocity")
          ax3.plot(t, (m*v**2)/2, "-",color = "blue")
          ax3.set_ylabel("Kinetic Energy")
          ax4.plot(t, (m*x**2)/2,"-",color = "blue")
          ax4.set_ylabel("Potential Energy")
          ax4.set_xlabel("Time")
          ax5.plot(t, (((m*x**2)/2) + ((m*v**2)/2)),"-",  color = "blue")
          ax5.set_ylabel("Total Energy")
          ax6.plot(x,v,"-", color = "blue")
          ax6.set_xlabel("Velocity")
          ax6.set_ylabel("Position")
          plt.show()
     if part == 2:
          t = np.arange(0,t,dt)
          fig , (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
          ax1.plot(t,x_, "-", color = "orange", label = "Semi Implicit")
          ax1.set_ylabel("Position")
          ax1.legend()
          ax2.plot(t, v_, "-", color = "orange")
          ax2.set_ylabel("Velocity")
          ax3.plot(t, (m*v_**2)/2, "-",color = "orange")
          ax3.set_ylabel("Kinetic Energy")
          ax4.plot(t, (m*x_**2)/2,"-",color = "orange")
          ax4.set_ylabel("Potential Energy")
          ax4.set_xlabel("Time")
          ax5.plot(t, (((m*x_**2)/2) + ((m*v_**2)/2)),"-",  color = "orange")
          ax5.set_ylabel("Total Energy")
          ax6.plot(x_,v_,"-", color = "orange")
          ax6.set_xlabel("Velocity")
          ax6.set_ylabel("Position")
          plt.show()
     if part == 3:
          t = np.arange(0,t,dt)
          fig = plt.figure()
          camera = Camera(fig)
          base = plt.gca().transData
          rot = tr.Affine2D().rotate_deg(90)
          for i in range(len(t)):
               plt.plot(-3,x[i],"o", markersize = 15, color = "teal",transform= rot + base)
               plt.plot(3,x_[i],"o", markersize = 15, color = "red",transform= rot + base)
               plt.vlines([-3]*len(x),[nm]*len(x),x[i],transform= rot + base)
               plt.vlines([3]*len(x_),[nm]*len(x_),x_[i],transform= rot + base)
               camera.snap()
          animation = camera.animate()
          plt.xlabel("Position")
          labels = ['Euler',"Semi-Implicit"]
          colors = ['teal', 'red']
          handles = []
          for c, l in zip(colors, labels):
              handles.append(Line2D([0], [0], color = c, label = l))

          plt.legend(handles = handles, loc = 'upper left')
          plt.show()

def coupledOscillator(k1, k2, m, x01, x02, v01, v02, dt, t, part):
     x1 = []
     x2 = []
     v1 = []
     v2 = []
     ran = int(t/dt)
     for i in range(ran):
          a1 = -((k1+k2)*x01 + k2*x02)/m
          a2 = -((k1+k2)*x02 + k2*x01)/m
          vi1 = v01 + a1*dt
          xi1 = x01 + vi1*dt
          vi2 = v02 + a2*dt
          xi2 = x02 + vi2*dt
          x1.append(x01)
          x2.append(x02)
          v1.append(v01)
          v2.append(v02)
          v01 = vi1
          v02 = vi2
          x01 = xi1
          x02 = xi2
     t = np.arange(0,t,dt)
     if part == 1:
          fig, (ax1, ax2) = plt.subplots(2)
          ax1.plot(t,x1, linewidth = 2, color = "aqua")
          ax1.set_ylabel("Mass 1 Position")
          ax2.plot(t,x2, linewidth = 2, color = "plum")
          ax2.set_ylabel("Mass 2 Position")
          ax2.set_xlabel("Time")
          plt.show()
     if part == 2:
          fig = plt.figure()
          camera = Camera(fig)
          base = plt.gca().transData
          rot = tr.Affine2D().rotate_deg(90)
          for i in range(len(t)):
               plt.plot(x1[i],"o", markersize = 15, color = "red",transform= rot + base)
               plt.plot(x2[i],"o", markersize = 15, color = "red",transform= rot + base)
               plt.vlines([0]*len(x1),x1[i],x2[i],transform= rot + base)
               camera.snap()
         
          animation = camera.animate()
          plt.xlabel("Position")
          plt.show()
     if part == 3:
          x1 = np.array(x1)
          x2 = np.array(x2)
          v1 = np.array(v1)
          v2 = np.array(v2)
          fig, (ax1, ax2, ax3) = plt.subplots(3)
          ax1.plot(t,((k1*x1**2) + k2*(x2-x1)**2 + (k1*x2**2))/2)
          ax1.set_ylabel("PE 2 mass system")
          ax2. plot(t,((m*v1**2) + (m*v2**2))/2)
          ax2.set_ylabel("KE 2 mass system")
          ax3.plot(t,((k1*x1**2) + k2*(x2-x1)**2 + (k1*x2**2) + ((m*v1**2) + (m*v2**2)))/2)
          ax3.set_ylabel("Total Energy")
          ax3.set_xlabel("Time")
          plt.show()


def tripleOscillator(k1, k2, m, x01, x02, x03, v01, v02, v03, dt, t, part):
     x1 = np.array([])
     x2 = np.array([])
     x3 = np.array([])
     v1 = np.array([])
     v2 = np.array([])
     v3 = np.array([])
     
     ran = int(t//dt)
     for i in range(ran+1):
          a1 = (-k1*x01)+(k2*(x02-x01))/m
          a2 = ((-k2*(x02-x01))+(k2*(x03-x02)))/m
          a3 = ((-k2*(x03-x02))+(-k1*x03))/m
          vi1 = v01 + a1*dt
          xi1 = x01 + vi1*dt
          vi2 = v02 + a2*dt
          xi2 = x02 + vi2*dt
          vi3 = v03 + a3*dt
          xi3 = x03 + vi3*dt
          x1 = np.append(x1,x01)
          x2 = np.append(x2,x02)
          x3 = np.append(x3,x03)
          v1 = np.append(v1,v01)
          v2 = np.append(v2,v02)
          v3 = np.append(v3,v03)
          v01 = vi1
          v02 = vi2
          v03 = vi3
          x01 = xi1
          x02 = xi2
          x03 = xi3     
     t = np.arange(0,t,dt)
     if part == 1:
          fig, (ax1,ax2,ax3) = plt.subplots(3)
          ax1.plot(t,x1,color = "darkorange")
          ax1.set_ylabel("Position mass 1")
          ax2.plot(t,x2,color = "red")
          ax2.set_ylabel("Position mass 2")
          ax3.plot(t,x3,label = "Position x3", color = "deeppink")
          ax3.set_ylabel("Position mass 3")
          ax3.set_xlabel("Time")

     if part == 2:
          fig = plt.figure()
          camera = Camera(fig)
          base = plt.gca().transData
          rot = tr.Affine2D().rotate_deg(90)
          for i in range(len(t)):
               plt.plot(x1[i],"o", markersize = 15, color = "red",transform= rot + base)
               plt.plot(x2[i],"o", markersize = 15, color = "red",transform= rot + base)
               plt.plot(x3[i],"o", markersize = 15, color = "red",transform= rot + base)
               plt.vlines([0]*len(x1),x1[i],x2[i],transform= rot + base)
               plt.vlines([0]*len(x1),x2[i],x3[i],transform= rot + base)
               camera.snap()
          animation = camera.animate()
          plt.xlabel("Position")
          plt.show()

     if part == 3:
          fig, (ax1,ax2,ax3) = plt.subplots(3)
          ax1.plot(t,((m*v1**2) + (m*v2**2) + (m*v3**2))/2, color = "darkorange")
          ax1.set_ylabel("KE 3 mass system")
          ax2.plot(t,((k1*x1**2)+(k2*(x2-x1))+(k2*(x2-x3))+(k1*x3))/2, color = "orange")
          ax2.set_ylabel("PE 3 mass system")
          ax3.plot(t,((m*v1**2) + (m*v2**2) + (m*v3**2)+(k1*x1**2)+(k2*(x2-x1))+(k2*(x2-x3))+(k1*x3))/2, color = "deeppink")
          ax3.set_ylabel("Total Energy")
          plt.xlabel("Time")
             
     plt.show()


def heatEQ(uo,uL,L,k,h,t,T0):
     L = L+1
     r = k/h**2
     time = np.arange(0,t,k)
     x = np.arange(0,L,h)
     u = np.ones(len(x))*T0
     u[0] = uo
     u[-1] = uL
     u_ = []
     u_.append(np.copy(u))
     c = np.copy(u)
     for j in range(len(time)):
          for i in range(len(u)-2):
               c[i+1] = (1-2*r)*u[i+1] + r*u[i] + r*u[i+2]
          u_.append(np.copy(c))
          u = np.copy(c)
     fig = plt.figure()
     camera = Camera(fig)
     for i in range(len(u_)):
         plt.plot(u_[i], color = "deeppink")
         camera.snap()
     animation = camera.animate()
     plt.xlabel("Position")
     plt.ylabel("Temperature")
     plt.show()


def heatEQ2D(uo, uL, L, h, k, T0, t):
     p = int(L/h)
     #L = L +1
     r = k/h**2
     time = np.arange(0,t,k)
     c = np.array([[T0]*p]*p)
     c[0,0] = uo
     c[-1,-1] = uL
     
     u_ = []
     #print(c)
     for j in range(len(time)):
          for i in range(L-2):
               for k in range(L-2):
                    c[i+1,k+1] = (((1-2*r)*c[k,i+1]) + (r*c[k,i]) + (r*c[k,i+2]))+\
                                 (((1-2*r)*c[i+1,k]) + (r*c[i,k]) + (r*c[i+2,k]))
          u_.append(c)
          print(u_[0][1])
    



     #print(u_)
     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
     x = c[0,:]
     y = c[:,0]
     x,y = np.meshgrid(x,y)

     camera = Camera(fig)
     mycmap = plt.get_cmap('jet')
     for i in range(L):
          z = u_[i]
          ax.plot_surface(x,y,z, cmap=mycmap,linewidth=0, antialiased=False)
          camera.snap()
     anim = camera.animate(blit=False, interval=10)


     plt.show()





          
          
     
     
     
          
     
