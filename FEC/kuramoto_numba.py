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

@njit
def _order_param(theta):
    """Devuelve R y psi del parámetro de orden a partir de theta."""
    re = 0.0
    im = 0.0
    N = theta.shape[0]
    for i in range(N):
        re += np.cos(theta[i])
        im += np.sin(theta[i])
    re /= N
    im /= N
    R = (re*re + im*im) ** 0.5
    psi = np.arctan2(im, re)
    return R, psi

@njit
def _rhs(theta, omega, K, out):
    """out[:] = f(theta) = ω + K*R*sin(ψ - theta)"""
    R, psi = _order_param(theta)
    N = theta.shape[0]
    for i in range(N):
        out[i] = omega[i] + K * R * np.sin(psi - theta[i])

@njit
def kuramoto_sim_rk4(theta0, omega, K, dt, nsteps):
    """
    Integra Kuramoto (forma de campo medio) con RK4.
    Retorna R_values (tamaño nsteps) y theta final.
    """
    N = theta0.shape[0]
    theta = theta0.copy()

    R_values = np.empty(nsteps)

    # buffers para RK4 (evita asignaciones dentro del loop)
    k1 = np.empty(N)
    k2 = np.empty(N)
    k3 = np.empty(N)
    k4 = np.empty(N)
    th_tmp = np.empty(N)

    for step in range(nsteps):
        # Guarda R(t) del estado actual
        R, _ = _order_param(theta)
        R_values[step] = R

        # RK4
        _rhs(theta, omega, K, k1)

        for i in range(N):
            th_tmp[i] = theta[i] + 0.5 * dt * k1[i]
        _rhs(th_tmp, omega, K, k2)

        for i in range(N):
            th_tmp[i] = theta[i] + 0.5 * dt * k2[i]
        _rhs(th_tmp, omega, K, k3)

        for i in range(N):
            th_tmp[i] = theta[i] + dt * k3[i]
        _rhs(th_tmp, omega, K, k4)

        # actualización y envoltura de ángulos
        for i in range(N):
            theta[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
            # mantener ángulos acotados en (-pi, pi]
            theta[i] = (theta[i] + np.pi) % (2.0*np.pi) - np.pi

    return R_values, theta




def binder_cumulant(m_series):
    m2 = np.mean(m_series**2)
    m4 = np.mean(m_series**4)
    return 1.0 - m4 / (2.0 * m2 * m2)

if __name__ == "__main__":
    # Parámetros
    N = 500               # número de osciladores
    dt = 0.01             # paso de integración
    tmax = 10_000.0         # tiempo total
    sigma = 1.0           # desviación estándar de ω_i
    Kc = 2 * sigma * np.sqrt(2/np.pi)  # umbral teórico para g(ω) ~ N(0, σ^2)
    K = 2.0                # acoplamiento en el umbral


    # Inicialización
    omega = np.random.normal(0.0, sigma, N)
    theta0 = np.random.uniform(0.0, 2*np.pi, N)
    nsteps = int(tmax / dt)

    # R_values, theta_final = kuramoto_sim_rk4(theta0, omega, K, dt, nsteps)

    # np.save("kuramoto/kuramoto_2.npy", R_values[100_000:])


    K_coarse = np.linspace(0.5, 2.5, 20)
    K_fine = np.linspace(Kc - 0.05, Kc + 0.05, 25)
    K_list = np.unique(np.concatenate([K_coarse, K_fine]))

    # for K in np.linspace(2.5, 3.5,10):
    #     R_values, theta_final = kuramoto_sim_rk4(theta0, omega, K, dt, nsteps)
    #     np.save(f"kuramoto/R_values/R_{K}.npy", R_values[:])


    print("hola")
    for N, tamaño in zip([600, 500, 400, 300, 200, 100],['600', '500', '400', '300', '200', '100']):
        omega = np.random.normal(0.0, sigma, N)
        theta0 = np.random.uniform(0.0, 2*np.pi, N)
        nsteps = int(tmax / dt)
        U_4_N = np.empty(len(K_list))
        print(N)
        for i,K in enumerate(K_list):
            R_values, theta_final = kuramoto_sim_rk4(theta0, omega, K, dt, nsteps)
            U_4_N[i] = binder_cumulant(R_values[100_000:])
        np.save('kuramoto/U_4_'+tamaño+'.npy', U_4_N)



    omega = np.random.normal(0.0, sigma, N)
    theta0 = np.random.uniform(0.0, 2*np.pi, N)
    nsteps = int(tmax / dt)
    R_means = np.empty(len(K_list))
    R_stds = np.empty(len(K_list))
    # for i,K in enumerate(K_list):
    #     print(K)
    #     R_values, theta_final = kuramoto_sim_rk4(theta0, omega, K, dt, nsteps)
    #     R_means[i] = np.mean(R_values[100_000:])
    #     R_stds[i] = np.std(R_values[100_000:])
    # np.save('kuramoto/R_mean_N'+str(N)+'.npy', R_means)
    # np.save('kuramoto/R_std_N'+str(N)+'.npy', R_stds)


    # R_values = np.load('kuramoto/kuramoto_1.npy')
    t = np.linspace(0.0, tmax, nsteps, endpoint=False)[:]
    plt.figure(figsize=(8, 4))
    plt.plot(R_values[100_000:], lw=1)
    plt.xlabel("Tiempo")
    plt.ylim(0, 1)
    plt.ylabel("Parámetro de orden R(t)")
    plt.title(f"Modelo de Kuramoto en K ≈ {K:.3f}, N={N}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
