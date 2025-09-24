import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def pasos_browniano(x0, dt, D):
    # np.random.normal() soportado en numba, sin argumentos
    return x0 + np.sqrt(2*D*dt) * np.random.normal()

@njit
def simulate_history_return_rw_numba(N,dt, q, b, seed, alpha=1.0, x0=0.0):
    y = 0.0
    np.random.seed(seed)
    x = np.empty(N+1, dtype=np.float64)
    x[0] = x0
    for t in range(N):
        y = pasos_browniano(y, dt=dt, D=1.0)
        if y >= b:
            if np.random.random() < q:
                k = np.random.randint(0, t+1)  # uniforme en {0,...,t}
                x[t+1] = x[k]
            else:
                if np.random.random() < 0.5:
                    x[t+1] = x[t] + alpha
                else:
                    x[t+1] = x[t] - alpha
            y = 0.0
        else:
            x[t+1] = x[t]
    return x

@njit
def msd_ensemble_numba(N, M,dt, q, b, seed=12345, x0 = 0.0):
    """
    Devuelve:
      - msd[t]   = E[(x_t - x0)^2] promedio en M caminantes
      - var[t]   = Var[(x_t - x0)^2] en el ensamble (para barras de error)
    """
    acc1 = np.zeros(N+1, dtype=np.float64)  # suma de d^2
    acc2 = np.zeros(N+1, dtype=np.float64)  # suma de (d^2)^2

    for m in range(M):
        print(m)
        seed_m = seed + 17*m + 23
        x = simulate_history_return_rw_numba(N,dt=dt, q=q, b=b, seed = seed)
        for t in range(N+1):
            d2 = (x[t] - x0) * (x[t] - x0)
            acc1[t] += d2
            acc2[t] += d2 * d2

    msd = acc1 / M
    var = acc2 / M - msd * msd
    # corrección numérica por posibles negativas ~1e-16
    for t in range(N+1):
        if var[t] < 0.0:
            var[t] = 0.0
    return msd, var

if __name__ == '__main__':


    # Ejemplo de uso: MSD para M caminantes
    T, M, dt = 10000, 10000, 0.01
    b, q = 1.0, 0.5
    alpha = 1.0
    N = int(T / dt)
    


    # Calcula MSD (usando la función que hicimos antes)
    msd, _ = msd_ensemble_numba(N, M, dt = dt, q=q, b= b)
    t = np.arange(0, N) * dt
    np.save("msd_boyer_q_espina_8.npy", msd)
    # msd = np.load("msd_boyer_q_8.npy")
    m, b = np.polyfit(np.log(t[1:]), msd[1:], 1)
    # --- Graficar ---
    plt.figure(figsize=(8, 5))
    plt.plot(t, msd, lw=1.5, label="MSD(t) simulado")
    plt.plot(t, ((1-q)/q)*(alpha*alpha*np.log(q*t) + 0.5772156649), 'k--', lw=1.5, label="MSD teórico")
    plt.xlabel("t (pasos)")
    plt.ylabel("MSD(t)")
    plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"MSD vs t para {M} caminantes (q={q}, α={alpha})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



