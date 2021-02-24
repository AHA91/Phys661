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
    h = np.array(m.items()).astype(np.float)
    total = sum(h[:,1])
    h[:,1] = h[:,1]*(100/total)
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
    sides = [1,2,3,4,5,6]
    rand1 = [random.choices(sides,(10,10,20,40,70,5)) for x in range(_N_)]
    rand2 = [random.choices(sides,(10,10,10,80,40,30)) for x in range(_N_)]
    h = [rand1[i]+rand2[i] for i in range(_N_)]
    g = Counter(h)
    return g

###########################  Graphs ######################################
######################## Remove Quotes for Single Dice ###################
'''
N_ = 1000
probab = [float((100/6))]*6
range_ = range(1,7)
d = experimental1(N_)

plt.plot(range_, probab,"o", color = "red")
plt.bar(d[:,0],d[:,1])
plt.xlabel("Side")
plt.ylabel("Outcome (%)")
plt.title("Single Dice")
plt.show()
'''
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
plt.ylabel("Frequency")
plt.xlabel("Sum For Roll")
plt.title("Fair Dice")
plt.show()
'''
###################### Remove Quotes for Part II #########################

h = ideal(1000,7)
x = h[:,0]
y = h[:,1]

_h_ = biased(1000)
_x_ = _h_.keys()
_y_ = _h_.values()
plt.bar(_x_,_y_, color = 'red')
plt.plot(x,y,'o', color = 'purple')
plt.ylabel("Frequency")
plt.xlabel("Sum For Rolls")
plt.title("Unfair Dice")
plt.show()

