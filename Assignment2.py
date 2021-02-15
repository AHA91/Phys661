#### Meri Khurshudyan
#### Assignment 2

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
    
h = ideal(1000,7)
x = h[:,0]
y = h[:,1]

_h_ = biased(1000)
_x_ = _h_.keys()
_y_ = _h_.values()


b = experimental(1000)
m = b.keys()
n = b.values()



'''
####################### Remove Quotes for Part I #########################
plt.bar(m,n, color = 'pink')
plt.plot(x,y,'o', color = 'purple')
plt.show()
'''

'''
###################### Remove Quotes for Part II #########################
plt.bar(_x_,_y_, color = 'red')
plt.plot(x,y,'o', color = 'purple')
plt.show()
'''
