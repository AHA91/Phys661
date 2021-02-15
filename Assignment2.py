from itertools import product
from itertools import permutations
from collections import Counter

t = [list(range(1,7)) for x in range(2)]
print(t)
g = list(map(sum,list(product(*t))))
z = Counter(g)
