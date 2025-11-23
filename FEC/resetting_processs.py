import numpy as np
import matplotlib.pyplot as plt

def simulate_diffusion_with_reset(
    x0=1.0,
    D=1.0,
    r=0.5,
    dt=1e-3,
    t_max=50.0,
    random_state=None
):
    """
    Simula una trayectoria 1D de difusión con reseteo:
        dx = sqrt(2D) dW
        reset a x0 con tasa r (Poisson)
    """
    rng = np.random.default_rng(random_state)
    n_steps = int(t_max / dt)
    t = np.linspace(0.0, t_max, n_steps + 1)

    x = np.empty(n_steps + 1, dtype=float)
    x[0] = x0

    sqrt_2Ddt = np.sqrt(2 * D * dt)

    for i in range(n_steps):
        # Paso difusivo
        x_trial = x[i] + sqrt_2Ddt * rng.normal()

        # Checar reseteo (aprox. Poisson: prob ~ r dt)
        if rng.random() < r * dt:
            x[i + 1] = x0
        else:
            x[i + 1] = x_trial

    return t, x

def estimate_stationary_distribution(
    x0=1.0,
    D=1.0,
    r=0.5,
    dt=1e-3,
    t_max=200.0,
    n_bins=80,
    random_state=None
):
    t, x = simulate_diffusion_with_reset(
        x0=x0, D=D, r=r, dt=dt, t_max=t_max, random_state=random_state
    )

    # Descartar transitorio (ej. primera mitad)
    x_ss = x[len(x)//2:]

    # Histograma normalizado
    hist, edges = np.histogram(x_ss, bins=n_bins, density=True)
    centers = 0.5 * (edges[0:-1] + edges[1:])
    # Distribución teórica
    alpha0 = np.sqrt(r / D)
    p_theory = 0.5 * alpha0 * np.exp(-alpha0 * np.abs(centers - x0))

    return centers, hist, p_theory



if __name__ == "__main__":
    # Ejemplo de trayectoria
    t, x = simulate_diffusion_with_reset(
        x0=1.0, D=1.0, r=0.5, dt=1e-3, t_max=100.0, random_state=123
    )

    plt.figure()
    plt.plot( x, lw=0.8, ls = '--') 
    plt.axhline(1.0, color='k', lw=0.5, ls='--')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Difusión 1D con reseteo estocástico")
    plt.tight_layout()
    plt.show()
    plt.hist(x, bins=50, density=True)
    plt.show()


    x0 = 1.0
    D = 1.0
    r = 0.5

    centers, hist, p_theory = estimate_stationary_distribution(
        x0=x0, D=D, r=r, dt=1e-3, t_max=10000.0,
        n_bins=1000, random_state=123
    )

    plt.figure()
    plt.bar(centers, hist, width=centers[1]-centers[0], alpha=0.4,
            label="Simulación (histograma)")
    plt.plot(centers, p_theory, lw=2, label="Teoría estacionaria")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Distribución estacionaria con reseteo")
    plt.legend()
    plt.tight_layout()
    plt.show()

