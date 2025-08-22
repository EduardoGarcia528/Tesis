import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from functools import partial
import multiprocessing as mp
import math
from collections import Counter

def logistic_map(r, x):
    return r * x * (1 - x)

@njit
def distancia(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

@njit
def mejor_vector(p1, p2):
    # Precomputar diferencias en los 9 cuadrantes
    diffs = [
        [p2[0], p2[1]],
        [p2[0], p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1]],
        [p2[0] + 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0], p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1]],
        [p2[0] - 2 * np.pi, p2[1] + 2 * np.pi],
    ]
    # Encontrar el índice con menor distancia
    d_og = distancia(p1,p2)
    min_idx = 0
    for i in range(9):
        d = distancia(p1, diffs[i])
        if d < d_og:
            min_idx = i
            d_og = d
    p2 = diffs[min_idx]
    return [p2[0] - 2*p1[0], p2[1] - 2*p1[1]]

@njit
def calcular_angulos(vectores):
    n = len(vectores) - 1
    angulos = np.empty(n)
    for i in range(n):
        v1 = vectores[i]
        v2 = vectores[i + 1]
        norm_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if norm_v1 == 0 or norm_v2 == 0:
            angulo = 0.0
        else:
            v1n0 = v1[0] / norm_v1
            v1n1 = v1[1] / norm_v1
            v2n0 = v2[0] / norm_v2
            v2n1 = v2[1] / norm_v2
            dot = v1n0 * v2n0 + v1n1 * v2n1
            if dot > 1.0: dot = 1.0
            if dot < -1.0: dot = -1.0
            angulo = np.arccos(dot)
            cruz = v1[0] * v2[1] - v1[1] * v2[0]
            if cruz > 0:
                angulo = np.pi - angulo
            elif cruz == 0 and angulo < 0:
                angulo = np.pi
            elif cruz < 0:
                angulo += np.pi
        angulos[i] = angulo
    return angulos

def caminata_univariante(X, tau):
    x1 = X[tau:]
    y1 = X[:-tau]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    n = len(ff1) - 1
    vectores = np.empty((n, 2))
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def indice_J(angulos):
    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    return J

def entropia_shannon(x, bins=150):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    p = hist / hist.sum()
    H = -np.sum(p * np.log2(p))
    return H / np.log2(bins)

def entropia_permutacion(x, m=4, tau=1):
    n = len(x)
    if n < (m - 1) * tau + 1:
        return np.nan

    patrones = []
    for i in range(n - (m - 1) * tau):
        ventana = x[i:i + tau * m:tau]
        orden = tuple(np.argsort(ventana))
        patrones.append(orden)

    cuenta = Counter(patrones)
    total = sum(cuenta.values())
    p = np.array(list(cuenta.values())) / total
    H = -np.sum(p * np.log2(p))
    H_norm = H / np.log2(math.factorial(m))  # normalización
    return H_norm

def diff_S(d, angulos):
    if d == 0:
        return entropia_shannon(angulos)
    for _ in range(d):
        angulos = np.diff(angulos)

    return entropia_shannon(angulos)

def main(serie,d):
    vectores = caminata_univariante(serie,tau = 1)
    angulos = calcular_angulos(vectores)
    entropia = diff_S(d,angulos=angulos)
    print(d)
    J = indice_J(angulos)
    if not np.isfinite(entropia):
        entropia = 0
    return J, entropia


if __name__ == '__main__':
    mp.freeze_support()

    l = 1000
    d = range(1000)

    # main_con_serie = partial(main, serie) Si main tiene más parámetros

    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados = pool.map(main, d)


    J, S_vals = map(np.array, zip(*resultados))