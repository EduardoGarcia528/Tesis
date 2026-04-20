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

###################

@njit
def ranks_with_ties(x):
    """
    Asigna el mismo rango a valores iguales.
    Ejemplo:
        x = [60, 62, 60, 64, 62] -> ord = [0, 1, 0, 2, 1]
    """
    N = len(x)
    idx = np.argsort(x)
    ord_ = np.empty(N, dtype=np.int64)

    if N == 0:
        return ord_

    current_rank = 0
    ord_[idx[0]] = current_rank

    for p in range(1, N):
        i_prev = idx[p - 1]
        i_curr = idx[p]

        if x[i_curr] != x[i_prev]:
            current_rank += 1

        ord_[i_curr] = current_rank

    return ord_


@njit
def correlation_integrals_rank_ties(x, max_d, K):
    """
    Integrales de correlación por rangos C_d^(R), tratando empates
    con el mismo rango.

    Parámetros
    ----------
    x : array 1D
        Serie de tiempo.
    max_d : int
        Máxima dimensión d requerida en C[d].
    K : int
        Ventana de cercanía en rangos.

    Regresa
    -------
    C : ndarray
        C[0], C[1], ..., C[max_d], con C[0] = 1.
    """
    N = len(x)
    C = np.zeros(max_d + 1, dtype=np.float64)
    C[0] = 1.0

    # Número de vectores válidos de longitud max_d
    Nv = N - max_d + 1
    if Nv < 2:
        for d in range(1, max_d + 1):
            C[d] = np.nan
        return C

    ord_ = ranks_with_ties(x)

    Ci = np.zeros(max_d + 1, dtype=np.int64)

    # Pares de vectores iniciales válidos
    for i in range(Nv):
        for j in range(i + 1, Nv):
            k = 0
            while k < max_d and abs(ord_[i + k] - ord_[j + k]) <= K:
                k += 1
                Ci[k] += 1

    M = Nv * (Nv - 1) / 2.0

    for d in range(1, max_d + 1):
        C[d] = Ci[d] / M

    return C

def gamma_index_jacobs_rank_ties(data, max_gamma, mu=5.0):
    data = np.asarray(data, dtype=np.float64)
    ord_ = ranks_with_ties(data)
    n_unique = int(ord_.max()) + 1 if len(ord_) > 0 else 0

    max_d = max_gamma + 1
    K = int(np.floor(n_unique / (2.0 * mu)))
    if K < 0:
        K = 0

    C = correlation_integrals_rank_ties(data, max_d, K)
    g = np.empty(max_gamma, dtype=np.float64)

    for j in range(1, max_gamma + 1):
        g[j - 1] = gamma(C, j)

    return C, g