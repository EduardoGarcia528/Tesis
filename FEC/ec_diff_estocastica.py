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

@njit
def segundo_momento(x_t):
    n = len(x_t)
    suma = 0.0
    suma2 = 0.0
    for i in range(n):
        suma += x_t[i]
        suma2 += x_t[i] * x_t[i]
    mean = suma / n
    var = suma2 / n - mean * mean
    return var + mean * mean


@njit
def main_estocastico(M,N,dt):
    x0 = [0.0]*M
    momento_2 = []
    T = np.linspace(0.0,N*dt,N)
    for t in range(N): # N total
        print(t)
        x_t = []
        
        for i in range(M): # particulas
            x_t.append(pasos(x0[i],dt))
        momento_2.append(segundo_momento(x_t))
        x0 = x_t
    return T,momento_2

@njit
def pasos_dicotomico(x0, v0, dt, v, lam):
    # con probabilidad lam*dt se invierte la velocidad
    if np.random.rand() < lam*dt:
        v0 = -v0
    x1 = x0 + v0*dt
    return x1, v0


if __name__ == '__main__':
    # mp.freeze_support()


    dt = 0.01
    N = 100_000
    M = 10000
    T = np.linspace(0.0,N*dt,N)
    momento_2 = np.load("momento_2.npy")
    # T, momento_2 = main_estocastico(M,N,dt)
    # np.save("momento_2.npy",momento_2)
    # v = 1.0       # velocidad fija
    # lam = 1.0     # tasa de cambio
    # x0 = np.zeros(M)
    # v0 = np.random.choice([-v, v], size=M)  # condiciones iniciales ±v
    # momento2_dic = []

    # for t in range(N):
    #     print(t)
    #     x_t = []
    #     for i in range(M):
    #         x0[i], v0[i] = pasos_dicotomico(x0[i], v0[i], dt, v, lam)
    #         x_t.append(x0[i])
    #     momento2_dic.append(np.var(x_t) + np.mean(x_t)**2)

    plt.plot(T,momento_2)
    plt.plot(T,2*T)
    # plt.plot(T, )
    plt.xlabel('t')
    plt.ylabel('momento 2')
    plt.show()