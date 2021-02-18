#### Meri Khurshudyan
#### Assignment 2

from itertools import product
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random

#@@@@@@@@@@@@@@@@@@@@@@@ Needed Functions @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
######################## Ideal Case Coin ###############################
def ideal_coin(_N_):
    # 3 coin tosses each with 1/2 prop make probability of dice 1/8
    #probability of each number is N/8 and percentage is N/8 
    return [_N_/1.5]*6
    
    
######################## Ideal Case 2 Dice #############################
def ideal(_N_, k):
    t = [list(range(1,k)) for x in range(2)] #creates two dice
    g = list(map(sum,list(product(*t)))) #gives all possible combinations
    z = Counter(g) #counts the occurances of each combination
    h = np.array(list(z.items())).astype(np.float) # convert the dict to an array
    total = sum(h[:,1]) # gives the total probabilities
    h[:,1] = (h[:,1]/total)*_N_ # where N is the number of times rolled the dice
    return h

####################### Experimental Case 2 Dice #########################
def experimental(_N_): 
    rand1 = [random.randint(1,6) for x in range(_N_)]
    rand2 = [random.randint(1,6) for x in range(_N_)]
    h = [rand1[i]+rand2[i] for i in range(_N_)]
    g = Counter(h)
    return g

###################### Biased Experimental Case 2 Dice ###################
def biased(_N_):
    rand1 = [int(random.triangular(1,6,5)) for x in range(_N_)]
    rand2 = [int(random.triangular(1,6,6)) for x in range(_N_)]
    h = [rand1[i]+rand2[i] for i in range(_N_)]
    g = Counter(h)
    return g
#################### Coin Simulation for Dice ############################
def coin_dice(_N_):
    _N_ = int(_N_/3)
    d = {'000':1, '001':2,'010':3,'011':4,'100':5,'101':6}
    result = []
    c1b = []
    c2b = []
    c3b = []
    C = []
    for i in range(_N_):
        a = str(random.randint(0,1)) 
        c1b.append(a)

        if a == '1':
            b = '0'
            c2b.append(b)
        else:  
            b = str(random.randint(0,1)) 
            c2b.append(b)              
        c_ = str(random.randint(0,1))
        c3b.append(c_)
    for i in range(_N_):
        C.append(c1b[i]+c2b[i]+c3b[i])
        C[i] = d.get(C[i])
    _C_ = Counter(C)
    h = np.array(list(_C_.items())).astype(np.float)
    h[:,1] = h[:,1]*100/_N_
    return h

###########################  Graphs ######################################
############## Remove Quotes for Coin Representation of Dice #############

m_ = ideal_coin(25)
plt.plot(range(1,7),m_,'o-', color = 'red')

C = coin_dice(25)
x_ = C[:,0]
y_ = C[:,1]


plt.bar(x_,y_,color = 'green')
plt.show()

####################### Remove Quotes for Part I #########################
'''
h = ideal(1000,7)
x = h[:,0]
y = h[:,1]

b = experimental(1000)
m = b.keys()
n = b.values()
plt.bar(m,n, color = 'pink')
plt.plot(x,y,'o', color = 'purple')
plt.show()
'''


###################### Remove Quotes for Part II #########################
'''
h = ideal(1000,7)
x = h[:,0]
y = h[:,1]

_h_ = biased(1000)
_x_ = _h_.keys()
_y_ = _h_.values()
plt.bar(_x_,_y_, color = 'red')
plt.plot(x,y,'o', color = 'purple')
plt.show()
'''
