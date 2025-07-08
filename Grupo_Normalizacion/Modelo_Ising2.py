import numpy as np
import pandas as pd
from numba import njit

# Parámetros
N = 128
T = 2.269
n = 1_000_000
equilibrar = 1000

# Funciones numba
@njit
def energia_vecino(spins, i, j):
    N = spins.shape[0]
    return spins[i, j] * (
        spins[(i+1)%N, j] + spins[(i-1)%N, j] +
        spins[i, (j+1)%N] + spins[i, (j-1)%N]
    )

@njit
def paso_montecarlo(spins, T):
    N = spins.shape[0]
    for _ in range(N * N):
        i = np.random.randint(N)
        j = np.random.randint(N)
        dE = 2 * energia_vecino(spins, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1

@njit
def magnetizacion(spins):
    return np.sum(spins) / (spins.shape[0] * spins.shape[1])

@njit
def generar_serie_ising(spins, T, n, equilibrar):
    for _ in range(equilibrar):
        paso_montecarlo(spins, T)

    serie = np.empty(n, dtype=np.float32)
    for t in range(n):
        paso_montecarlo(spins, T)
        print(t)
        serie[t] = magnetizacion(spins)
    return serie

#Inicializar spins
spins = np.random.choice([-1, 1], size=(N, N))

#Correr y guardar
serie = generar_serie_ising(spins, T=T, n=n, equilibrar=equilibrar)

np.save('serie_ising_Tc.csv',serie)