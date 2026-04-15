import numpy as np
from numba import njit

# C[d] devuelve la integral de correlación rank-based en dimensión d
# C[0] = 1.0 por convención
# Si quieres gamma_1, ..., gamma_max_gamma, necesitas C[0] hasta C[max_gamma+1]
# Por eso max_d = max_gamma + 1

@njit
def correlation_integrals_rank(x, max_d, K):
    N = len(x)

    # Número de vectores iniciales válidos para poder comparar hasta dimensión max_d
    # Un vector que empieza en i usa x[i], x[i+1], ..., x[i+max_d-1]
    Nv = N - max_d + 1

    C = np.zeros(max_d + 1, dtype=np.float64)
    C[0] = 1.0

    if Nv < 2:
        for d in range(1, max_d + 1):
            C[d] = np.nan
        return C

    # Construcción de rangos ord_[t]:
    # ord_[t] = posición de x[t] en el ordenamiento global de la serie
    idx = np.argsort(x)
    ord_ = np.empty(N, dtype=np.int64)
    for rank in range(N):
        ord_[idx[rank]] = rank

    # Ci[d] contará cuántos pares cumplen cercanía hasta dimensión d
    Ci = np.zeros(max_d + 1, dtype=np.int64)

    # Recorremos pares de vectores iniciales válidos
    for i in range(Nv):
        for j in range(i + 1, Nv):
            k = 0
            while k < max_d and abs(ord_[i + k] - ord_[j + k]) <= K:
                k += 1
                Ci[k] += 1

    # Número total de pares de vectores
    M = Nv * (Nv - 1) / 2.0

    for d in range(1, max_d + 1):
        C[d] = Ci[d] / M

    return C


def gamma(C, j):
    denom = C[j - 1] * C[j + 1]
    if denom == 0.0 or np.isnan(denom):
        return np.nan
    return 1.0 - (C[j] ** 2) / denom


def gamma_index_jacobs_rank(data, max_gamma, mu=5.0):
    """
    Calcula la versión rank-based de las integrales de correlación y del índice gamma.

    Parámetros
    ----------
    data : array_like
        Serie de tiempo.
    max_gamma : int
        Calcula gamma_1, ..., gamma_max_gamma.
    mu : float
        Parámetro de resolución. La ventana en rangos es K = floor(N/(2*mu)).

    Regresa
    -------
    C : ndarray
        Arreglo con C[0], C[1], ..., C[max_gamma+1].
    g : ndarray
        Arreglo con gamma_1, ..., gamma_max_gamma.
    K : int
        Ventana de rangos usada.
    """
    data = np.asarray(data, dtype=np.float64)
    N = len(data)

    max_d = max_gamma + 1
    K = int(np.floor(N / (2.0 * mu)))

    # Evitar K negativo o degenerado
    if K < 0:
        K = 0

    C = correlation_integrals_rank(data, max_d, K)

    g = np.empty(max_gamma, dtype=np.float64)
    for j in range(1, max_gamma + 1):
        g[j - 1] = gamma(C, j)

    return C, g