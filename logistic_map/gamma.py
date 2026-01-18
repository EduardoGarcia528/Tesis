import numpy as np


def gamma_index_jacobs(data, MAXD=7, MAXDEL=6, mu=5):
    """
    Traducción directa del código C de Laurence Jacobs
    para el cálculo del índice gamma.

    Parámetros
    ----------
    data : array-like
        Serie temporal 1D
    MAXD : int
        Dimensión máxima de embedding (default = 7)
    MAXDEL : int
        Máximo gamma reportado (default = 6)
    mu : int
        Resolución de datos: eps = sd / mu

    Retorna
    -------
    C : ndarray
        Integrales de correlación C_d
    gamma : ndarray
        Índices gamma
    """

    # --- Copia fiel de la lógica del C ---
    data = np.asarray(data, dtype=float)

    # El código C ignora data[0] y trabaja desde data[1]
    data = np.concatenate([[0.0], data])
    N = len(data) - 1

    # Media y desviación estándar
    mean = np.mean(data[1:])
    sd = np.sqrt(np.sum((data[1:] - mean) ** 2) / (N - 1))

    eps = sd / mu

    # Valor sentinela
    maxdat = np.max(data[1:])
    data = np.append(data, maxdat + 100 * eps)

    # --- label(): ordenar índices según data ---
    tseq = np.argsort(data)[::-1]   # orden descendente
    tseq = tseq.astype(int)

    # --- Conteo de coincidencias ---
    Ci = np.zeros(MAXD + 1, dtype=int)

    for j in range(1, N + 1):
        for i in range(j + 1, N + 1):
            k = 0
            while (
                k < MAXD
                and abs(data[tseq[i] + k] - data[tseq[j] + k]) <= eps
            ):
                k += 1
                Ci[k] += 1

    # Normalización
    norm = 2.0 / (N * (N - 1))
    C = np.zeros(MAXD + 1)
    C[0] = 1.0

    for i in range(1, MAXD + 1):
        C[i] = Ci[i] * norm

    # --- Cálculo de gamma ---
    gamma = np.zeros(MAXD + 1)

    for i in range(1, MAXDEL + 1):
        denom = C[i - 1] * C[i + 1]
        if denom != 0:
            gamma[i] = 1.0 - (C[i] ** 2) / denom

    return C, gamma
