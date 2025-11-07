import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def trayectoria_1D(N, dt, D):
    x = np.zeros(N)
    s = np.sqrt(2.0 * D * dt)
    for t in range(1, N):
        x[t] = x[t-1] + s * np.random.normal()
    return x

@njit
def msd_y_finales(M, N, dt, D):
    msd = np.zeros(N)
    finales = np.zeros(M)
    s = np.sqrt(2.0 * D * dt)
    for m in range(M):
        x = 0.0
        for t in range(N):
            x += s * np.random.normal()
            msd[t] += x * x
        finales[m] = x
    msd /= M
    return msd, finales

# =========================
# Main
# =========================
if __name__ == "__main__":
    D  = 1.0         # coeficiente de difusión
    dt = 0.01        
    N  = 100_000     # pasos por trayectoria
    M  = 5000        # número de caminantes 
    T_total = N * dt
    rng_seed = 12345 

    if rng_seed is not None:
        np.random.seed(rng_seed)

    # 1) Una trayectoria (desplazamiento vs tiempo)
    x = trayectoria_1D(N, dt, D)
    t = np.arange(N) * dt

    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Caminante Browniano: una trayectoria")
    plt.tight_layout()

    msd, finales = msd_y_finales(M, N, dt, D)
    distancias = np.abs(finales)  

    plt.figure()
    plt.hist(np.abs(np.diff(x)), bins=80, density=True)
    plt.xlabel(r"|x(T)|")
    plt.ylabel("Densidad")
    plt.title(f"Histograma de |x(T)|, M={M}, T={T_total:.2f}")

    # sigma^2 = 2 D T, p(r) = sqrt(2/(pi sigma^2)) * exp(-r^2/(2 sigma^2)), r>=0
    sigma = np.sqrt(2*D*dt)
    x = np.linspace(0, np.abs(np.diff(x)).max(), 200)
    pdf = np.sqrt(2/np.pi)/sigma * np.exp(-x**2/(2*sigma**2))
    plt.plot(x, pdf, 'r', lw=2, label="Teoría")
    plt.tight_layout()

    # 3) MSD vs t y comparación con 2 D t (log-log)
    plt.figure()
    plt.plot(t, msd, label="MSD (ensamble)")
    plt.plot(t, 2.0 * D * t, linestyle="--", label=r"$2Dt$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\langle x^2(t)\rangle$")
    plt.title("Desplazamiento medio cuadrático (MSD)")
    plt.legend()
    plt.tight_layout()

    plt.show()
