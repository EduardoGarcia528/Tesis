import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def simulate_diffusion_with_reset(
    x0=1.0,
    D=1.0,
    r=0.5,
    dt=1e-3,
    t_max=50.0,
    random_state=None,
    first_passage=False
):
    rng = np.random.default_rng(random_state)
    n_steps = int(t_max / dt)
    t = np.linspace(0.0, t_max, n_steps + 1)

    x = np.empty(n_steps + 1, dtype=float)
    x[0] = x0

    sqrt_2Ddt = np.sqrt(2 * D * dt)

    for i in range(n_steps):
        # Paso difusivo
        x_trial = x[i] + sqrt_2Ddt * rng.normal()

        # Checar reseteo 
        if rng.random() < r * dt:
            x[i + 1] = x0
        else:
            x[i + 1] = x_trial
        if first_passage == True and x[i + 1] <= 0.0:
            return (i + 1)*dt            
    if first_passage == False:
        return x

def estimate_stationary_distribution(
    x0=1.0,
    D=1.0,
    r=0.5,
    dt=1e-3,
    t_max=200.0,
    n_bins=80,
    random_state=None
):
    x = simulate_diffusion_with_reset(
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

import numpy as np

def compute_msd(X, X0):
    disp = X - X0  
    msd = np.mean(disp**2, axis = 0)
    return msd



if __name__ == "__main__":
    # Ejemplo de trayectoria
    x = simulate_diffusion_with_reset(
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

    x0 = 1.0
    D = 1.0
    r = 0.5

    centers, hist, p_theory = estimate_stationary_distribution(
        x0=x0, D=D, r=r, dt=1e-3, t_max=100.0,
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

    x0 = [0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    MFPT = []
    # for x0_i in x0:
    #     print(x0_i)
    #     times = []
    #     for i in range(5000):
    #         T = simulate_diffusion_with_reset(
    #             x0=x0_i, D=D, r=r, dt=1e-3, t_max=10_000,
    #             first_passage=True)
    #         if T is not None:
    #             times.append(T)
    #     mfpt_num = np.mean(times)
    #     MFPT.append(mfpt_num)
    # np.save("mfpt_reset.npy", np.array([x0, MFPT]))
    x0, MFPT = np.load("mfpt_reset.npy")

    x0_theory = np.linspace(0.1, 3.0, 1000)
    alpha0 = np.sqrt(r / D)
    mfpt_theory = (np.exp(alpha0 * x0_theory) - 1) / r

    plt.figure()
    plt.plot(x0, MFPT, 'o', label="MFPT numérico")
    plt.plot(x0_theory, mfpt_theory, '-', label="MFPT teórico")
    plt.xlabel("Posición inicial $x_0$")
    plt.ylabel("MFPT")
    plt.title("Tiempo promedio de primer paso con reseteo")
    plt.legend()
    plt.tight_layout()
    plt.show()

    r_0, MFPT = np.load("mfpt_reset_r.npy")
    # r_0 = [0, 0.01,0.03, 0.05, 0.1, 0.2, 0.5,1.0, 2.5, 15.0]
    # for r_i in [20.0]:
    #     print(r_i)
    #     times = []
    #     for i in range(5000):
    #         T = simulate_diffusion_with_reset(
    #             x0=1.0, D=D, r=r_i, dt=1e-3, t_max=10_000,
    #             first_passage=True)
    #         if T is not None:
    #             times.append(T)
    #     if len(times) < 2000:
    #         mfpt_num = np.nan
    #     else:
    #         mfpt_num = np.mean(times)
    #     MFPT = np.concatenate( (MFPT, [mfpt_num]) )
    #     r_0 = np.concatenate( (r_0, [r_i]) )
    # np.save("mfpt_reset_r.npy", np.array([r_0, MFPT]))

    r0_theory = np.linspace(0.001, 100.0, 2000)
    alpha0 = np.sqrt(r0_theory / D)
    mfpt_theory = (np.exp(alpha0) - 1) / r0_theory

    plt.figure()
    plt.plot(r_0, MFPT, 'o', label="MFPT numérico")
    plt.plot(r0_theory, mfpt_theory, '-', label="MFPT teórico")
    plt.xlabel("Tasa $r$")
    plt.ylabel("MFPT")
    plt.title("Tiempo promedio de primer paso con reseteo")
    plt.legend()
    plt.tight_layout()
    plt.show()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    r0_theory = np.linspace(0.000000001, 100.0, 2000) 
    alpha0 = np.sqrt(r0_theory / D)
    mfpt_theory = (np.exp(alpha0) - 1) / r0_theory

    plt.figure(figsize=(7,4))
    ax = plt.gca()

    # --- Gráfica principal ---
    ax.plot(r_0, MFPT, 'o', label="MFPT numérico")
    ax.plot(r0_theory, mfpt_theory, '-', label="MFPT teórico")
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(0,15.0)
    ax.set_xlabel("Tasa $r$")
    ax.set_ylabel("MFPT")
    ax.set_title("Tiempo promedio de primer paso con reseteo")
    ax.legend()

    # --- INSET ---
    ax_inset = inset_axes(ax, width="35%", height="35%", loc="upper right")

    # mismos datos
    ax_inset.plot(r0_theory, mfpt_theory, '-', color="C1")
    ax_inset.plot(r_0, MFPT, '.', color="C0")

    # límites para el zoom
    ax_inset.set_xlim(-10.0, 100.0)

    # ajusta automáticamente el eje y según los datos en ese rango
    mask = (r0_theory >= 0) & (r0_theory <= 3.0)
    ax_inset.set_ylim(0, 220.0)
    # ax_inset.set_ylim(mfpt_theory[mask].min()-1, mfpt_theory[mask].max())

    # ticks más simples
    ax_inset.tick_params(axis='both', labelsize=8)

    # rectángulo que marca la zona del zoom (opcional)
    # mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.6")

    plt.tight_layout()
    plt.show()
