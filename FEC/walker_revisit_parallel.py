import numpy as np
import math
import time
from numba import njit
from multiprocessing import Pool, cpu_count
from functools import partial

@njit
def beta_binomial_probs(n, alpha, beta):
    probs = np.empty(n, dtype=np.float64)
    N = n

    log_B0 = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_N_fact = math.lgamma(N + 1)

    for k in range(n):
        log_binom = log_N_fact - math.lgamma(k + 1) - math.lgamma(N - k + 1)
        log_B1 = (math.lgamma(k + alpha)
                  + math.lgamma(N - k + beta)
                  - math.lgamma(N + alpha + beta))
        probs[k] = math.exp(log_binom + log_B1 - log_B0)

    s = 0.0
    for i in range(n):
        s += probs[i]
    for i in range(n):
        probs[i] /= s

    return probs


@njit
def choice_numba(probs):
    cdf = np.empty_like(probs)
    cdf[0] = probs[0]
    for i in range(1, probs.size):
        cdf[i] = cdf[i - 1] + probs[i]

    r = np.random.random()
    for i in range(cdf.size):
        if r < cdf[i]:
            return i
    return probs.size - 1

@njit
def simulate_trajectory(T, q, alpha, beta, p=0.5, x0=0):
    positions = np.empty(T + 1)
    positions[0] = x0

    for t in range(1, T + 1):
        if np.random.random() < q:
            probs = beta_binomial_probs(t, alpha, beta)
            t_prime = choice_numba(probs)
            positions[t] = positions[t_prime]
        else:
            step = 1 if np.random.random() < p else -1
            positions[t] = positions[t - 1] + step

    return positions

def worker_msd(n_traj_local, T, q, alpha, beta, p, x0, seed):
    np.random.seed(seed)

    sq_positions = np.zeros((n_traj_local, T + 1))
    for i in range(n_traj_local):
        traj = simulate_trajectory(T, q, alpha, beta, p, x0)
        sq_positions[i] = traj ** 2

    return np.mean(sq_positions, axis=0)

def simulate_ensemble_msd(
    T,
    n_traj,
    q,
    alpha,
    beta,
    p=0.5,
    x0=0,
    seed=123,
    n_proc=None,
):

    if n_proc is None:
        n_proc = cpu_count()

    traj_per_proc = n_traj // n_proc
    extras = n_traj % n_proc

    counts = [traj_per_proc + (i < extras) for i in range(n_proc)]
    seeds = [seed + i for i in range(n_proc)]

    with Pool(processes=n_proc) as pool:
        results = pool.starmap(
            worker_msd,
            [(counts[i], T, q, alpha, beta, p, x0, seeds[i])
             for i in range(n_proc)]
        )

    msd = np.mean(results, axis=0)
    t_values = np.arange(T + 1)
    return t_values, msd

if __name__ == "__main__":
    T = 10000
    n_traj = 15000
    q = 0.4
    beta = 1.0
    p = 0.5

    for alpha in [2.5]:
        # t0 = time.perf_counter()
        for beta in [2.0]:
            print(alpha,beta)
            t, msd = simulate_ensemble_msd(
                T=T,
                n_traj=n_traj,
                q=q,
                alpha=alpha,
                beta=beta,
                p=p,
                seed=123,
                n_proc=None,
            )
            # t1 = time.perf_counter()
            # print(f"Tiempo transcurrido: {t1 - t0:.6f} s")
            np.save(f"general_MSD/msd_beta_binomial_a_{str(alpha)[0]}_{str(alpha)[-1]}_b_{str(beta)[0]}_{str(beta)[-1]}.npy", msd)
