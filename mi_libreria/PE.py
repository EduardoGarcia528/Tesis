import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
from tqdm import tqdm


@njit
def lehmer_code(perm):
    """Codifica una permutación en un índice único usando Lehmer code"""
    m = len(perm)
    code = 0
    factor = 1
    for i in range(m-1, -1, -1):
        c = 0
        for j in range(i+1, m):
            if perm[j] < perm[i]:
                c += 1
        code += c * factor
        factor *= (m - i)
    return code

@njit
def stable_argsort_by_value_then_index(x):
    m = x.shape[0]
    idx = np.arange(m)
    # insertion sort por clave (valor, índice)
    for i in range(1, m):
        key = idx[i]
        j = i - 1
        while j >= 0:
            a = x[idx[j]]
            b = x[key]
            if (a > b) or (a == b and idx[j] > key):  # (valor) y luego (índice)
                idx[j+1] = idx[j]
                j -= 1
            else:
                break
        idx[j+1] = key
    return idx

@njit
def permutation_entropy(arr, m=3, tau=1, norm=True):
    n = len(arr)
    if n < m:
        return np.nan
    # m!:
    fact = 1
    for k in range(2, m+1):
        fact *= k
    counts = np.zeros(fact, dtype=np.int64)
    denom = n - (m-1)*tau
    for i in range(denom):
        subseq = np.empty(m, np.float64)
        for j in range(m):
            subseq[j] = arr[i + j*tau]
        idx = stable_argsort_by_value_then_index(subseq)
        code = lehmer_code(idx)      # tu misma función
        counts[code] += 1
    # entropía normalizada (independiente de base)
    probs = counts[counts > 0] / denom
    n_prohibidos = fact - len(probs)
    H = -np.sum(probs * np.log(probs))
    if norm:
        Hnorm = H / np.log(fact)
        return Hnorm
    else:
        return H



@njit
def tie_pattern(subseq):
    """
    Devuelve el patrón ordinal con empates.
    Ejemplos:
    (60, 60, 62, 64) -> (0, 0, 1, 2)
    (62, 62, 60, 64) -> (1, 1, 0, 2)
    """
    m = subseq.shape[0]
    idx = stable_argsort_by_value_then_index(subseq)

    pattern = np.empty(m, dtype=np.int64)

    rank = 0
    pattern[idx[0]] = rank

    for k in range(1, m):
        if subseq[idx[k]] > subseq[idx[k - 1]]:
            rank += 1
        pattern[idx[k]] = rank

    return pattern


@njit
def tie_pattern_code(pattern, m):
    """
    Codifica el patrón en un índice único usando base m.
    No todos los códigos corresponden a patrones válidos, pero eso no importa:
    los inválidos simplemente nunca aparecen.
    """
    code = 0
    for i in range(pattern.shape[0]):
        code = code * m + pattern[i]
    return code


@njit
def count_tie_patterns(m):
    """
    Número de patrones ordinales admisibles con empates:
    sum_{k=1}^m k! * S(m,k)
    donde S(m,k) son números de Stirling de segunda especie.
    """
    S = np.zeros((m + 1, m + 1), dtype=np.int64)
    S[0, 0] = 1

    for n in range(1, m + 1):
        for k in range(1, n + 1):
            S[n, k] = S[n - 1, k - 1] + k * S[n - 1, k]

    total = 0
    fact = 1
    for k in range(1, m + 1):
        fact *= k
        total += fact * S[m, k]

    return total


@njit
def modified_permutation_entropy(arr, m=3, tau=1,norm=True):
    """
    Entropía permutacional modificada para incluir empates.
    Normaliza por el número de patrones admisibles con empates.
    """
    n = len(arr)
    denom = n - (m - 1) * tau

    if denom <= 0:
        return np.nan

    # espacio de códigos en base m
    n_codes = 1
    for _ in range(m):
        n_codes *= m

    counts = np.zeros(n_codes, dtype=np.int64)

    for i in range(denom):
        subseq = np.empty(m, dtype=np.float64)
        for j in range(m):
            subseq[j] = arr[i + j * tau]

        pattern = tie_pattern(subseq)
        code = tie_pattern_code(pattern, m)
        counts[code] += 1

    probs = counts[counts > 0] / denom

    H = -np.sum(probs * np.log(probs))

    # máximo teórico correcto para patrones con empates
    n_states = count_tie_patterns(m)
    if norm:
        Hnorm = H / np.log(n_states)
        return Hnorm
    else:
        return H