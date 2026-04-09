import numpy as np
from numba import njit

# maxd te da maxd+2 valores de C, el primero es C[0](d=0) = 1.0, el ultimo es C[maxd+1]
# max_gamma te da max_gamma valores de g, el primero es g[0] = gamma(C,1), el ultimo es g[max_gamma-1] = gamma(C,max_gamma)
# maxd = max_gamma + 1

# max_gamma te da desde gamma_1 hasta gamma_{max_gamma}, lo que significa que 
# te da desde C(d=0) hasta C(d=max_gamma+1) (max_gamma+1 integrales de correlacion en total)

@njit
def correlation_integrals(x, max_d, eps):
    N = len(x)
    idx = np.argsort(x)

    Ci = np.zeros(max_d + 1, dtype=np.int64)

    for a in range(N):
        ia = idx[a]
        for b in range(a + 1, N):
            ib = idx[b]

            if abs(x[ib] - x[ia]) > eps:
                break

            k = 0
            while k < max_d and abs(x[ia + k] - x[ib + k]) <= eps:
                k += 1
                Ci[k] += 1

    C = np.zeros(max_d + 1)
    norm = 2.0 / (N * (N - 1))
    C[0] = 1.0
    for d in range(1, max_d + 1):
        C[d] = Ci[d] * norm

    return C


def gamma(C, j):
    return 1-(C[j] ** 2) / (C[j - 1] * C[j + 1])

def gamma_index_jacobs(data, max_gamma, mu = 5.0):
    max_d = max_gamma + 1
    C = correlation_integrals(data, max_d, np.std(data) / mu)
    g = np.zeros(max_gamma)
    for j in range(1, max_gamma + 1):
        if C[j - 1] * C[j + 1] == 0.0:
            g[j - 1] = np.nan
        else:
            g[j - 1] = gamma(C, j)
    return C, g