import numpy as np

data1 = np.load('test.npy')

data2 = np.load('one.npy')

dist = np.sqrt(np.sum(np.square(data1 - data2)))

print(dist)
