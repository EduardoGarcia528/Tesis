import numpy as np
import math
from itertools import permutations

def ordinal_pattern_index(vec):
    """
    Devuelve el índice del patrón ordinal de vec.
    Usa mergesort para desempates estables.
    """
    ranks = tuple(np.argsort(vec, kind="mergesort"))
    return ranks

def ordinal_counts_nonoverlap(x, D, tau=1):
    """
    Conteos p_i usando ventanas NO traslapadas, como en el artículo.
    """
    x = np.asarray(x)
    patterns = list(permutations(range(D)))
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    counts = np.zeros(math.factorial(D), dtype=int)

    step = D * tau
    max_start = len(x) - (D - 1) * tau

    for s in range(0, max_start, step):
        vec = x[s : s + D * tau : tau]
        if len(vec) == D:
            pat = ordinal_pattern_index(vec)
            counts[pattern_to_idx[pat]] += 1

    return counts

def N_required_eq25_from_counts(counts, sigma=0.01):
    """
    Implementa Eq. (25) TAL COMO ESTÁ escrita.
    counts = array con los p_i.
    """
    counts = np.asarray(counts, dtype=float)
    D_fact = len(counts)              # esto es D!
    logDf = np.log(D_fact)

    term1 = np.sum(counts**2)
    term2 = np.sum(np.outer(counts, counts)) / (D_fact - 1)

    rhs = ((D_fact - 1) / (sigma**2 * logDf**2)) * (term1 - term2)
    N_req = rhs ** (1/3)

    return N_req

def N_required_eq25_from_series(x, D, tau=1, sigma=0.01):
    counts = ordinal_counts_nonoverlap(x, D=D, tau=tau)
    N_obs = counts.sum()
    N_req = N_required_eq25_from_counts(counts, sigma=sigma)
    return {
        "counts": counts,
        "N_obs": int(N_obs),
        "N_req": float(N_req),
        "enough_data": N_obs >= N_req
    }

# ejemplo
# rng = np.random.default_rng(123)
# x = rng.normal(size=2460)
# D = 5
# res = N_required_eq25_from_series(x, D=D, tau=1, sigma=0.01)

# print("N observado =", res["N_obs"])
# print("N requerido =", res["N_req"])
# print(f"L requerido = {res['N_req']*D}")
# print("¿alcanza la precisión sigma=0.01?:", res["enough_data"])