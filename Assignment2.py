#### Meri Khurshudyan
#### Assignment 2

from itertools import product
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random



################### Ideal Case 2 Dice ##################################
def ideal(_N_, k):
    t = [list(range(1,k)) for x in range(2)] #creates two dice
    g = list(map(sum,list(product(*t)))) #gives all possible combinations
    z = Counter(g) #counts the occurances of each combination
    h = np.array(list(z.items())).astype(np.float) # convert the dict to an array
    total = sum(h[:,1]) # gives the total probabilities
    h[:,1] = (h[:,1]/total)*_N_ # where N is the number of times rolled the dice
    return h

#################### Experimental Case 2 Dice ###########################
def experimental(_N_): #where _N_ is still the number of coin tosses
    rand1 = [random.randint(1,6) for x in range(_N_)]
    rand2 = [random.randint(1,6) for x in range(_N_)]
    h = [rand1[i]+rand2[i] for i in range(_N_)]
    g = Counter(h)
    return g


h = ideal(1000,7)
x = h[:,0]
y = h[:,1]

b = experimental(1000)
m = b.keys()
n = b.values()

'''
plt.bar(m,n, color = 'pink')
plt.plot(x,y,'o', color = 'purple')
plt.show()
'''
