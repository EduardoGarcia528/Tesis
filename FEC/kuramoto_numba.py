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
    return 1.0 - m4 / (3.0 * m2 * m2)

if __name__ == "__main__":
    # ========= Parámetros globales =========
    sigma = 1.0                           # desviación estándar de ω_i
    Kc = 2 * sigma * np.sqrt(2/np.pi)     # umbral teórico para g(ω) ~ N(0, σ^2)

    # Tamaños de sistema (puedes agregar 6400, 12800 si tu compu aguanta)
    # N_list = [400, 1600, 3200]
    N_list= [4000,5000]

    # Rango de K alrededor de Kc
    K_values = np.linspace(1.2, 2.0, 21)  # incluye Kc ~ 1.596

    # Tiempo de integración
    dt = 0.02
    tmax = 800.0                         # como en el paper: Nt = 4e4 pasos aprox.
    nsteps = int(tmax / dt)

    # Transitorio a descartar (por ejemplo, mitad de la simulación)
    n_transient = 5_000

    # Número de realizaciones de desorden (muestras) por (N, K)
    n_realizations = 200                  # sube o baja según paciencia/cómputo

    # Diccionario para guardar B^{(2)}(K, N)
    B2_results = {}

    # ========= Loop principal sobre N y K =========
    rng = np.random.default_rng(seed=12345)  # RNG para reproducibilidad

    for N in N_list:
        print(f"Simulando N = {N}")
        B2_vs_K = np.zeros_like(K_values, dtype=float)

        for ik, K in enumerate(K_values):
            print(K)
            b_samples = np.zeros(n_realizations, dtype=float)

            for s in range(n_realizations):
                # Frecuencias naturales ~ N(0, sigma^2), desorden quenched por muestra
                omega = rng.normal(loc=0.0, scale=sigma, size=N)
                # Condiciones iniciales de las fases
                theta0 = rng.uniform(0.0, 2.0*np.pi, size=N)

                # Integra Kuramoto con RK4
                R_values, _ = kuramoto_sim_rk4(theta0, omega, K, dt, nsteps)

                # Descarta el transitorio
                R_ss = R_values[n_transient:]

                # Binder por muestra: b_s = 1 - <R^4> / (3 <R^2>^2)
                b_samples[s] = binder_cumulant(R_ss)

            # B^{(2)}(K, N) = promedio sobre muestras de b_s
            B2_vs_K[ik] = b_samples.mean()
        np.save(f'kuramoto/bindersN/B2_vs_K_{int(N)}.npy',B2_vs_K)

        B2_results[N] = B2_vs_K

    # ========= Gráfica tipo Fig. 5: B^{(2)} vs K =========
    plt.figure(figsize=(6, 4))
    for N in N_list:
        # B2_results = np.load(f'kuramoto/bindersN\B2_vs_K_{int(N)}.npy')
        plt.plot(K_values, B2_results[N], marker='o', ms=3, lw=1, label=f"N={N}")

    plt.axvline(Kc, linestyle='--', alpha=0.7, label=r"$K_c$")
    plt.xlabel(r"Acoplamiento $K$")
    plt.ylabel(r"$B_{\Delta}^{(2)}$")
    plt.title(r"Cumulante de Binder $B_{\Delta}^{(2)}$ vs $K$")
    plt.legend()
    plt.tight_layout()
    plt.show()

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
