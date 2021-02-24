#### Meri Khurshudyan
#### Assignment 2
##### Up to date version of matplotlib is required 

from itertools import product
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random

#@@@@@@@@@@@@@@@@@@@@@@@ Needed Functions @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
######################## Ideal Case 2 Dice #############################
def ideal(_N_, k):
    t = [list(range(1,k)) for x in range(2)] #creates two dice
    g = list(map(sum,list(product(*t)))) #gives all possible combinations
    z = Counter(g) #counts the occurances of each combination
    h = np.array(list(z.items())).astype(np.float) # convert the dict to an array
    total = sum(h[:,1]) # gives the total probabilities
    h[:,1] = (h[:,1]/total)*_N_ # where N is the number of times rolled the dice
    return h

####################### Single Dice #######################################
def experimental1(_N_):
    rand1 = [random.randint(1,6) for x in range(_N_)]
    m = Counter(rand1)
    return m
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

###########################  Graphs ######################################
######################## Remove Quotes for Single Dice ###################
N_ = 1000
probab = [float((N_/6))]*6
print(probab)
range_ = range(1,7)
d = experimental1(N_)

plt.plot(range_, probab,"o", color = "red")
plt.bar(d.keys(),d.values())
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
