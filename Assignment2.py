#### Meri Khurshudyan
#### Assignment 2

from itertools import product
from itertools import permutations
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt



################### Ideal Case 2 Dice ##################################
def ideal(_N_):
    t = [list(range(1,7)) for x in range(2)] #creates two dice
    g = list(map(sum,list(product(*t)))) #gives all possible combinations
    z = Counter(g) #counts the occurances of each combination
    h = np.array(list(z.items())).astype(np.float) # convert the dict to an array
    total = sum(h[:,1]) # gives the total probabilities
    h[:,1] = h[:,1]/total*_N_ # where N is the number of times rolled the dice
    return h


h = ideal(1000)
x = h[:,0]
y = h[:,1]

plt.plot(x,y,'o')
plt.show()

#################### Experimental Case 2 Dice ###########################
