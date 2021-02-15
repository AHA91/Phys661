#### Meri Khurshudyan
#### Assignment 2

from itertools import product
from itertools import permutations
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

_N_ = 1000

################### Ideal Case 2 Dice ##################################
t = [list(range(1,7)) for x in range(2)] #creates two dice
g = list(map(sum,list(product(*t)))) #gives all possible combinations
z = Counter(g) #counts the occurances of each combination
h = np.array(list(z.items())) # convert the dict to an array
h = h.astype(np.float) # just convert it to float 
total = sum(h[:,1]) # gives the total probabilities
h[:,1] = h[:,1]/total*_N_ # where N is the number of times rolled the dice

x = h[:,0]
y = h[:,1]

plt.plot(x,y)
plt.show()
