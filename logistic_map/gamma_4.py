import numpy as np
from numba import njit


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
    return 1.0 - (C[j] ** 2) / (C[j - 1] * C[j + 1])

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