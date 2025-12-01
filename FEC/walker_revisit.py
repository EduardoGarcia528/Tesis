import numpy as np
from scipy.stats import betabinom
from numba import njit
import math

def plot_beta_binomial(n, alpha, beta):
    k = np.arange(0, n)                 # posibles valores 0,1,...,n
    pmf = betabinom.pmf(k, n, alpha, beta)  # función de masa de probabilidad
    probs = pmf / pmf.sum()

    return probs

@njit
def beta_binomial_probs(n, alpha, beta):
    """
    Aproxima betabinom.pmf(k, n, alpha, beta) para k = 0,1,...,n-1
    y normaliza para que sume 1 (truncada en k = n-1, igual que tu código).
    """
    probs = np.empty(n, dtype=np.float64)  # k = 0,...,n-1
    N = n                                  # parámetro 'n' de la beta-binomial

    # log B(alpha, beta)
    log_B0 = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_N_fact = math.lgamma(N + 1)

    for k in range(n):  # k = 0,...,n-1
        # log binom(N, k)
        log_binom = log_N_fact - math.lgamma(k + 1) - math.lgamma(N - k + 1)

        # log B(k+alpha, N-k+beta)
        log_B1 = (math.lgamma(k + alpha)
                  + math.lgamma(N - k + beta)
                  - math.lgamma(N + alpha + beta))

        log_p = log_binom + log_B1 - log_B0
        probs[k] = math.exp(log_p)

    # Normalizar por seguridad (igual que pmf / pmf.sum())
    s = 0.0
    for i in range(n):
        s += probs[i]
    for i in range(n):
        probs[i] /= s

    return probs

@njit
def choice_numba(probs):
    """
    Emula np.random.choice(np.arange(len(probs)), p=probs)
    """
    # Construir CDF
    cdf = np.empty_like(probs)
    cdf[0] = probs[0]
    for i in range(1, probs.size):
        cdf[i] = cdf[i - 1] + probs[i]
    
    # Número uniforme en [0,1)
    r = np.random.random()
    
    # Buscar el primer índice donde CDF >= r
    for i in range(cdf.size):
        if r < cdf[i]:
            return i
    
    # Por seguridad, devolver el último
    return probs.size - 1


@njit
def simulate_trajectory(
    T: int,
    q: float,
    alpha: float,
    beta: float,
    p: float = 0.5,
    x0: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:

    positions = np.empty(T + 1)
    positions[0] = x0

    for t in range(1, T + 1):
        x_prev = positions[t - 1]
        if np.random.random() < q:
            # Paso con memoria: elegir tiempo pasado t' con Beta-binomial
            # Usamos n = t, de modo que t' ∈ {0,...,t}
            probs = beta_binomial_probs(t, alpha, beta)
            # probs = plot_beta_binomial(t, alpha, beta)
            # t_prime = np.random.choice(np.arange(t), p = probs)
            t_prime = choice_numba(probs)
            positions[t] = positions[t_prime]
        else:
            # Paso local aleatorio
            step = 1 if np.random.random() < p else -1
            positions[t] = x_prev + step

    return positions


def simulate_ensemble_msd(
    T: int,
    n_traj: int,
    q: float,
    alpha: float,
    beta: float,
    p: float = 0.5,
    x0: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)

    # Para memoria: guardamos x^2 para cada trayectoria
    sq_positions = np.zeros((n_traj, T + 1))
    for i in range(n_traj):
        print(i)
        traj = simulate_trajectory(
            T=T,
            q=q,
            alpha=alpha,
            beta=beta,
            p=p,
            x0=x0,
            rng=rng,
        )
        sq_positions[i, :] = (traj)**2
        # sq_positions[0,:] = sq_positions[0,:] + sq_positions[1,:]

    msd = np.mean(sq_positions, axis=0)
    t_values = np.arange(T + 1)

    return t_values, msd


if __name__ == "__main__":
    # Ejemplo de uso: caso especial alpha=2, beta=1
    T = 10000
    n_traj = 5000
    q = 0.2
    alpha = 2.0
    beta = 1.0
    p = 0.5
    # t, msd = simulate_ensemble_msd(
        # T=T, n_traj=n_traj, q=q, alpha=alpha, beta=beta, p=p, seed=123)
    # np.save("msd_beta_binomial_1_2.npy", msd)
    msd = np.load("msd_beta_binomial_2.npy")
    t = np.arange(0, len(msd))
    # msd_theory = ((1 - q) / q) * (np.log(q*t) + 0.577215664902)  # Constante de Euler-Mascheroni
    msd_theory = np.log(q*t)*alpha*(1-q)/(q) + alpha*(alpha-1)*(1-q)/(t*q**2)
    import matplotlib.pyplot as plt
    plt.plot(t, msd, label=f"q={q}, α={alpha}, β={beta}")
    plt.plot(t, msd_theory, lw=2, ls='--', label=f"Teoría (α={alpha}, β=1)")
    plt.xlabel("Tiempo t")
    plt.ylabel("MSD ⟨x(t)²⟩")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MSD para caminante con memoria Beta-binomial")
    plt.legend()
    plt.tight_layout()
    plt.show()
