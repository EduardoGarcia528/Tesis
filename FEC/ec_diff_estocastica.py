import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def pasos(x0,dt):
    x1 = x0 + np.sqrt(2*dt)*np.random.normal(loc=0.0, scale=1.0, size=None)

    return x1

@njit
def segundo_momento(x_t):
    n = len(x_t)
    suma = 0.0
    for i in range(n):
        suma += x_t[i]*x_t[i]
    momento2 = suma / n
    return momento2



@njit
def main_estocastico(M,N,dt):
    x0 = [0.0]*M
    momento_2 = []
    for t in range(N): # N total
        print(t)
        x_t = []
        
        for i in range(M): # particulas
            x_t.append(pasos(x0[i],dt))
        momento_2.append(segundo_momento(x_t))
        x0 = x_t
    return momento_2


if __name__ == '__main__':
    # mp.freeze_support()


    dt = 0.01
    N = 100_000
    M = 10000
    T = np.linspace(0.0,N*dt,N)
    momento_2 = np.load("momento_2.npy")
    # momento_2 = main_estocastico(M,N,dt)


    plt.plot(T[1:],momento_2[:-1])
    plt.plot(T,2*T)
    plt.xlabel('t')
    plt.ylabel('momento 2')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()