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

def caminata_univariante(X, tau, bivariante):
    if bivariante is False:
        x1 = X[tau:]
        y1 = X[:-tau]
    else:
        x1 = X
        y1 = bivariante
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

def entropia_shannon(x, bins=100):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    try:
        hist, _ = np.histogram(x, bins=bins, density=True)
    except: 
        return np.nan
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

def main(angulos,d):
    # angulos = np.load('henon_C_1000diff.npy')
    entropia = diff_S(d,angulos=angulos)
    print(d)
    J = indice_J(angulos)
    if not np.isfinite(entropia):
        entropia = 0
    return J, entropia

def henon_map(a, b, n, trans=0):
    total = n + trans
    xs = np.zeros(total)
    ys = np.zeros(total)
    
    # Condiciones iniciales
    xs[0] = 0.1
    ys[0] = 0.1
    
    # Iteraciones
    for i in range(1, total):
        xs[i] = 1 - a * xs[i-1]**2 + ys[i-1]
        ys[i] = b * xs[i-1]
    
    return xs[trans:], ys[trans:]


def pendiente_loglog_optima(d, H, min_len=10):
    """
    Encuentra la pendiente óptima de log(H) vs log(d) y el rango donde el ajuste es mejor.

    Parámetros:
        d : array_like
            Valores de la variable independiente (debe ser >0).
        H : array_like
            Valores de la variable dependiente (debe ser >0).
        min_len : int
            Longitud mínima de la ventana para considerar el ajuste.
    
    Retorna:
        dict con:
            'pendiente': pendiente óptima beta
            'intercepto': log-constante
            'indices_opt': índices de la ventana óptima
            'd_fit': valores de d de la ventana óptima
            'H_fit': valores ajustados de H en la ventana óptima
            'error': error cuadrático medio del ajuste
    """
    d = np.array(d)
    H = np.array(H)
    
    # Verificar que todos los valores sean positivos para el log
    if np.any(d <= 0) or np.any(H <= 0):
        raise ValueError("Todos los valores de d y H deben ser positivos para log.")
    
    x = np.log(d)
    y = np.log(H)
    
    n = len(y)
    mejor_error = np.inf
    resultado = {}

    # Probar todas las ventanas posibles mayores a min_len
    for start in range(n):
        for end in range(start + min_len, n+1):
            x_window = x[start:end]
            y_window = y[start:end]

            # Ajuste lineal (mínimos cuadrados)
            A = np.vstack([x_window, np.ones_like(x_window)]).T
            m, b = np.linalg.lstsq(A, y_window, rcond=None)[0]

            # Error cuadrático medio
            y_fit = m*x_window + b
            error = np.mean((y_window - y_fit)**2)

            if error < mejor_error:
                mejor_error = error
                resultado = {
                    'pendiente': m,
                    'intercepto': b,
                    'indices_opt': np.arange(start, end),
                    'd_fit': d[start:end],
                    'H_fit': np.exp(y_fit),  # volver a escala original
                    'error': error
                }
    
    return resultado


if __name__ == '__main__':
    # mp.freeze_support()
    
    l = 1000
    d = np.arange(1,l+1,1)

    x = 0.6
    r = 3.5699431086217244
    # r = 3.52
    # r = 4.0
    serie = []
    for _ in range(100_000):
        x = logistic_map(r,x)

    for _ in range(100_000):
        x = logistic_map(r,x)
        serie.append(x)
    serie = np.array(serie)
    print("ja")
    # serie = np.load("kuramoto_Rc.npy")[90000:] # 1.401155189
    a = 1.0576244479 
    a = 1.057730803809001
    # a = 1.06
    # a = 1.057730791108901
    seriex,seriey = henon_map(a, b = 0.3, n = 300_000,trans=100_000)
    print("ja")
    # serie = np.random.uniform(0, 1, 200_000)
    vectores = caminata_univariante(seriex,tau = 1,bivariante=seriey)
    angulos = calcular_angulos(vectores)
    S_vals= []
    for i in np.arange(1,l+1,1):
        print(i)
        angulos = np.diff(angulos)
        S_vals.append(entropia_shannon(angulos,bins=100))
    # main_con_serie = partial(main, angulos)
    # print("jo")
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     resultados = pool.map(main_con_serie, d)
    

    # J, S_vals = map(np.array, zip(*resultados))

    # S_vals = []
    # for d in range(2000):
    #     J, S = main(seriex,d, bivariante=seriey)
    #     S_vals.append(S)
    # np.save('S_vals_henon_Rc.npy')
    S_vals = np.array(S_vals, dtype=np.float64)[np.isfinite(S_vals)]
    d = d[np.isfinite(S_vals)]

    print(len(S_vals[np.isfinite(S_vals)]))
    # Ejemplo
    res = pendiente_loglog_optima(d, S_vals, min_len=100)
    print("Pendiente:", res['pendiente'])
    print("Intercepto:", res['indices_opt'])
    print("Error óptimo:", res['error'])
    # Graficar
    plt.loglog(S_vals[np.isfinite(S_vals)], marker='o')
    plt.xlabel("Diferenciación (d)")
    plt.title(f'Henon Map (bivariante), a = {a}, b = 0.3')
    plt.ylabel("Entropía de Shannon normalizada")
    plt.grid()
    plt.tight_layout()
    plt.show()