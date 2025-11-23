import numpy as np
import matplotlib.pyplot as plt

def msd_diffusion_with_reset(
    x0=1.0,
    D=1.0,
    r=0.5,
    dt=1e-3,
    t_max=10.0,
    n_trajs=1000,
    random_state=None
):
    """
    Calcula el MSD <(x(t) - x0)^2> para difusión 1D con reseteo a x0,
    usando muchas trayectorias pero sin guardar todas las posiciones.

    Ahorro de memoria:
        - NO guardamos una matriz [n_trajs, n_steps].
        - Solo:
            - acumulador msd_sum[t_idx]
            - posición actual x de cada trayectoria
    """
    rng = np.random.default_rng(random_state)

    n_steps = int(t_max / dt)
    t = np.linspace(0.0, t_max, n_steps + 1)

    # Acumulador para el MSD (suma sobre trayectorias)
    msd_sum = np.zeros(n_steps + 1, dtype=float)

    sqrt_2Ddt = np.sqrt(2 * D * dt)

    for traj in range(n_trajs):
        x = x0  # cada trayectoria empieza en x0

        # tiempo t=0: contribución al MSD
        msd_sum[0] += (x - x0) ** 2  # esto es 0, pero lo dejo por claridad

        for i in range(1, n_steps + 1):
            # Paso difusivo
            x_trial = x + sqrt_2Ddt * rng.normal()

            # Reseteo con prob ~ r dt (Poisson)
            if rng.random() < r * dt:
                x = x0
            else:
                x = x_trial

            # Acumular (x - x0)^2 para este tiempo
            msd_sum[i] += (x - x0) ** 2

    # Promedio sobre trayectorias
    msd = msd_sum / n_trajs

    return t, msd


if __name__ == "__main__":
    # Parámetros
    x0 = 1.0
    D = 1.0
    r = 0.5
    dt = 1e-3
    t_max = 50.0
    n_trajs = 100000

    t, msd_num = msd_diffusion_with_reset(
        x0=x0, D=D, r=r,
        dt=dt, t_max=t_max,
        n_trajs=n_trajs,
        random_state=123
    )

    # MSD teórico: <(x - x0)^2> = (2D/r)(1 - e^{-rt})
    msd_theory = (2 * D / r) * (1.0 - np.exp(-r * t))

    # Gráfica
    plt.figure()
    plt.plot(t, msd_num, label="MSD numérico", lw=1)
    plt.plot(t, msd_theory, "--", label="MSD teórico", lw=2)
    plt.xlabel("t")
    plt.ylabel(r"$\langle (x(t)-x_0)^2 \rangle$")
    plt.title("MSD en difusión con reseteo")
    plt.legend()
    plt.tight_layout()
    plt.show()
    np.save("msd.npy", np.array(msd_num))