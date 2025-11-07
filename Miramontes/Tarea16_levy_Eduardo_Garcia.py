import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import seaborn as sns

alphas   = [1.05, 1.5, 2.0, 2.5, 6.0]
N        = 10_000     # pasos por trayectoria
M        = 1000       # caminantes en el ensamble 
rng_seed = 12345  
x_range = (0, 1000)
y_range = (0, 1000)


def levy_walk(n_walkers, n_steps, alpha, x_range, y_range):
    """
    Simulate Lévy Walk for multiple walkers with boundary conditions.

    Args:
        n_walkers (int): Number of walkers.
        n_steps (int): Number of steps for each walker.
        alpha (float): Power-law exponent for the step length distribution.
        x_range (tuple): Range for x coordinates (min, max).
        y_range (tuple): Range for y coordinates (min, max).

    Returns:
        ndarray: Array of walker trajectories with shape (n_walkers, n_steps, 2).
    """
    positions = np.zeros((n_walkers, n_steps, 2))

    for i in range(n_walkers):
        x, y = np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])
        for j in range(n_steps):
            # Generate step length from power-law distribution
            step_length = np.random.pareto(alpha)

            # Generate random direction
            theta = np.random.uniform(0, 2 * np.pi)

            # Update position
            new_x = x + step_length * np.cos(theta)
            new_y = y + step_length * np.sin(theta)

            # Check boundary conditions
            if x_range[0] <= new_x <= x_range[1] and y_range[0] <= new_y <= y_range[1]:
                x, y = new_x, new_y

            positions[i, j] = [x, y]

    return positions

def msd_y_finales_levy(M, N, alpha, x_range, y_range):
    msd = np.zeros(N)
    positions = levy_walk(n_walkers = M, n_steps = N, alpha = alpha, x_range = x_range, y_range = y_range)
    x = positions[:,:,0]
    x_centered = x - x.mean(axis=1, keepdims=True)
    msd = np.mean(x_centered**2, axis = 0)
    # msd /= M
    return msd



if __name__ == "__main__":
    if rng_seed is not None:
        np.random.seed(rng_seed)

    t = np.arange(N, dtype=np.float64)  # 1 paso = 1 unidad de tiempo

    plt.figure(figsize=(10, 5))
    for alpha in alphas:
        x = levy_walk(1, N, alpha, x_range, y_range)[0, : , 0]
        plt.plot(t, x, label=f"α={alpha}")
    plt.xlabel("t (pasos)")
    plt.ylabel("x(t)")
    plt.title("Una trayectoria de Lévy flight por α (1D, truncado)")
    plt.legend()
    plt.tight_layout()


    fig_hist, axs_hist = plt.subplots(1, len(alphas), figsize=(14, 3.6), sharey=False)

    for j, alpha in enumerate(alphas):
        print(j)
        msd = msd_y_finales_levy(M, N, alpha, x_range, y_range)
        np.save(f't15/msd{j}.npy', msd)
        x = levy_walk(1, N, alpha, x_range, y_range)[0, : , 0]
        dist_abs = np.diff(x)

        # Histograma |x(T)|
        axh = axs_hist[j]
        axh.hist(dist_abs, bins=100, density=True)
        axh.set_yscale('log')
        axh.set_title(f"α={alpha}")
        axh.set_xlabel(r"|x(T)|")
        if j == 0:
            axh.set_ylabel("Densidad")


    fig_hist.suptitle(f"Histograma de |x(T)|, M={M}, N={N}")
    fig_hist.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    for j, alpha in enumerate(alphas):
        msd  = np.load(f't15/msd{j}.npy')
        plt.plot(t[1:], msd[1:], label = f"MSD α={alpha}")
    # plt.plot(t[1:], t[1:], label = "m = 1")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.legend()
    plt.ylabel(r"$\langle x^2(t)\rangle$")
    plt.show()