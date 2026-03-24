import numpy as np
from numba import njit

@njit
def circ_dist(a, b):
    """
    Distancia angular mínima entre dos ángulos en [0, 2*pi).
    Regresa un valor en [0, pi].
    """
    d = abs(a - b)
    if d > np.pi:
        d = 2.0 * np.pi - d
    return d


@njit
def correlation_integrals_circular(x, max_d, eps):
    """
    Calcula C[d] para una serie angular x usando distancia circular.
    
    x debe estar en [0, 2*pi).
    """
    N = len(x)

    # Índices iniciales válidos para comparar bloques de longitud max_d
    M = N - max_d + 1
    C = np.zeros(max_d + 1, dtype=np.float64)

    if M < 2:
        C[:] = np.nan
        return C

    Ci = np.zeros(max_d + 1, dtype=np.int64)

    # Comparación directa entre todos los pares válidos
    for ia in range(M):
        for ib in range(ia + 1, M):

            k = 0
            while k < max_d and circ_dist(x[ia + k], x[ib + k]) <= eps:
                k += 1
                Ci[k] += 1

    norm = 2.0 / (M * (M - 1))
    C[0] = 1.0
    for d in range(1, max_d + 1):
        C[d] = Ci[d] * norm

    return C


def gamma(C, j):
    return 1.0 - (C[j] ** 2) / (C[j - 1] * C[j + 1])


def gamma_index_jacobs_circular(data, max_gamma, nu=5.0):
    """
    Índice gamma para series angulares.
    
    Usa:
        eps = pi / nu
    y distancia circular mínima.
    """
    max_d = max_gamma + 1
    eps = np.pi / nu

    data = np.asarray(data, dtype=np.float64)
    C = correlation_integrals_circular(data, max_d, eps)

    g = np.zeros(max_gamma, dtype=np.float64)
    for j in range(1, max_gamma + 1):
        if np.isnan(C[j - 1]) or np.isnan(C[j + 1]) or C[j - 1] * C[j + 1] == 0.0:
            g[j - 1] = np.nan
        else:
            g[j - 1] = gamma(C, j)

    return C, g