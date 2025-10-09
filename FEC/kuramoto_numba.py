import numpy as np
import matplotlib.pyplot as plt
from numba import njit  


@njit
def kuramoto_sim(theta0, omega, K, dt, nsteps):

    N = theta0.shape[0]
    theta = theta0.copy()
    R_values = np.empty(nsteps)

    for step in range(nsteps):
        # if step == nsteps//4 or step == nsteps//2:
        re = 0.0
        im = 0.0
        for i in range(N):
            re += np.cos(theta[i])
            im += np.sin(theta[i])
        re /= N
        im /= N

        R = (re*re + im*im) ** 0.5
        psi = np.arctan2(im, re)
        R_values[step] = R

        # Paso de Euler
        for i in range(N):
            theta[i] += dt * (omega[i] + K * R * np.sin(psi - theta[i]))

    return R_values, theta


if __name__ == "__main__":
    # Parámetros
    N = 500               # número de osciladores
    dt = 0.01             # paso de integración
    tmax = 100_000.0         # tiempo total
    sigma = 1.0           # desviación estándar de ω_i
    Kc = 2 * sigma * np.sqrt(2/np.pi)  # umbral teórico para g(ω) ~ N(0, σ^2)
    K = 2                # acoplamiento en el umbral


    # Inicialización
    omega = np.random.normal(0.0, sigma, N)
    theta0 = np.random.uniform(0.0, 2*np.pi, N)
    nsteps = int(tmax / dt)

    # R_values, theta_final = kuramoto_sim(theta0, omega, K, dt, nsteps)

    # np.save("kuramoto/kuramoto_2.npy", R_values[100_000:])

    R_values = np.load('kuramoto/kuramoto_1.npy')
    t = np.linspace(0.0, tmax, nsteps, endpoint=False)[100_000:]
    plt.figure(figsize=(8, 4))
    plt.plot(t, R_values - np.mean(R_values), lw=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Parámetro de orden R(t)")
    plt.title(f"Modelo de Kuramoto en K ≈ {K:.3f}, N={N}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
