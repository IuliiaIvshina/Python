x=5
print(x)

import numpy as np
a = np.arange(10).reshape(2,5)
print(a)
print(a.shape)
print(a.ndim)
print(a.size)
print(a.dtype.name)
print(a.itemsize)
print(np.array([range(5),range(5)]))

ch = np.array( [a for a in range(10) if a/2 == 0])
print(ch)

from numpy.random import randint
#print(np.array([a/randint(10) for a in range(10)]))

M_test = np.zeros((10,10))
M_test[2,2] = 10
print(M_test)
np.load()