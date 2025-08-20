import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from functools import partial
import multiprocessing as mp
import math
from collections import Counter

@njit
def pasos(x0,dt):
    x1 = x0 + np.sqrt(2*dt)*np.random.normal(0.0,1.0,1)[0]
    return x1




if __name__ == '__main__':
    # mp.freeze_support()


    dt = 0.01
    N = 100_000
    M = 10000
    x0 = [0.0]*M
    momento_2 = []
    for t in range(N): # N total
        print(t)
        x_t = []
        
        for i in range(M): # particulas
            x_t.append(pasos(x0[i],dt))
        momento_2.append(np.var(x_t) + np.mean(x_t)**2)
        x0 = x_t


    plt.plot(momento_2)
    plt.xlabel('t')
    plt.ylabel('momento 2')
    plt.show()