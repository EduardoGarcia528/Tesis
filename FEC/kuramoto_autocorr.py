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


def acf_fft(x, max_lag=None, demean=True, unbiased=True):
    """
    ACF normalizada C(τ)=⟨δx_t δx_{t+τ}⟩/⟨δx_t^2⟩ por FFT.
    - unbiased: divide por (N-τ); si False, divide por N (biased).
    - max_lag: máximo retardo (índice), por defecto N-1.
    Devuelve C[0..max_lag] con C[0]=1.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 2:
        return np.array([1.0])
    if demean:
        x = x - x.mean()

    # FFT de longitud power-of-two suficiente
    nfft = 1
    while nfft < 2*N - 1:
        nfft <<= 1
    F = np.fft.rfft(x, n=nfft)
    S = F * np.conjugate(F)
    acf_full = np.fft.irfft(S, n=nfft)[:N].real

    if unbiased:
        norm = np.arange(N, 0, -1)  # N, N-1, ..., 1
    else:
        norm = np.full(N, N, dtype=float)

    acf = acf_full / norm
    var = acf[0] if acf[0] > 0 else 1.0
    acf /= var

    if max_lag is None or max_lag >= N:
        max_lag = N - 1
    return acf[:max_lag+1]

def integrated_autocorr_time(acf, dt=1.0, window='automatic'):
    """
    τ_int = dt * (1 + 2 * sum_{τ=1}^{W} ACF[τ])
    - window='automatic': suma hasta el primer cruce ACF<=0 (o todo si no hay cruce).
    - window=int: usa ese W fijo.
    """
    if len(acf) == 0:
        return 0.0, 0
    if window == 'automatic':
        pos = np.where(acf[1:] <= 0)[0]
        W = (pos[0] + 1) if len(pos) else len(acf) - 1
    elif isinstance(window, int):
        W = min(window, len(acf) - 1)
    else:
        W = len(acf) - 1
    tau_int = dt * (1.0 + 2.0 * float(np.sum(acf[1:W+1])))
    return tau_int, W

def analyze_R(R_t, dt=1.0, tau_max_time=None):
    """
    - R_t: serie R(t).
    - tau_max_time: τ_max en unidades de tiempo (no en índices).
    """
    R_ss = R_t
    if tau_max_time is None:
        max_lag = None
    else:
        max_lag = int(tau_max_time / dt)
    C = acf_fft(R_ss, max_lag=max_lag, demean=True, unbiased=True)
    tau_int, W = integrated_autocorr_time(C, dt=dt, window='automatic')
    return C, tau_int, W, R_ss

# ---------- tu simulación ----------
# Debes tener definida en tu entorno:
# kuramoto_sim_rk4(theta0, omega, K, dt, nsteps) -> (R_values, theta_final)

if __name__ == "__main__":
    # Parámetros
    N = 500                 # número de osciladores
    dt = 0.01               # paso de integración
    tmax = 10_000.0         # tiempo total
    sigma = 1.0             # desviación estándar de ω_i
    Kc = 2 * sigma * np.sqrt(2/np.pi)  # umbral teórico para g(ω) ~ N(0, σ^2)

    # Inicialización (misma semilla si quieres reproducibilidad)
    rng = np.random.default_rng(12345)
    omega = rng.normal(0.0, sigma, N)
    theta0 = rng.uniform(0.0, 2*np.pi, N)

    nsteps = int(tmax / dt)

    # Barrido en K
    Ks = np.linspace(0.5, 2.5, 20)


    # Elegir una ventana máxima en TIEMPO para comparar entre K
    tau_max_time = 200.0  # unidades de tiempo (ajústalo si quieres mirar colas más largas)

    # Almacenes
    acf_by_K = {}
    tauint_by_K = {}
    sel_curves = []  # guardará algunas curvas para graficar ACF

    for K in Ks:
        R_values = np.load(f"kuramoto\R_values\R_{K}.npy")[100_000:]

        # --- ACF y τ_int ---
        C, tau_int, W, R_ss = analyze_R(
            R_values,
            dt=dt,
            tau_max_time=tau_max_time
        )
        acf_by_K[K] = C
        tauint_by_K[K] = tau_int

        # Guarda 3-4 curvas representativas para graficar ACF (debajo, ~K<Kc, ~Kc, >Kc)
        # Selección simple por proximidad a Kc
        sel_curves.append((abs(K-Kc), K, C))
    # Elige 4 K representativos: el más bajo, cercano a Kc, un poco arriba, y el más alto
    sel_curves.sort(key=lambda t: t[0])
    chosen = set(k for _, k, _ in sel_curves[:2])  # 2 más cercanos a Kc
    chosen.add(Ks[0])                              # el más bajo
    chosen.add(Ks[-1])                             # el más alto

    # ---------- Plots ----------
    # ACF para Ks seleccionados
    plt.figure(figsize=(7,4))
    for K in sorted(chosen):
        C = acf_by_K[K]
        lags = np.arange(len(C)) * dt
        plt.plot(lags, C, label=f"K={K:.3f}")
    plt.xlabel(r"Delay $\tau$")
    plt.ylabel(r"ACF $C(\tau)$")
    plt.title("Autocorrelación de R(t) para distintos K")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # τ_int vs K (firma de la transición)
    Ks_sorted = np.array(sorted(tauint_by_K.keys()))
    tau_sorted = np.array([tauint_by_K[K] for K in Ks_sorted])
    plt.figure(figsize=(6.5,4))
    plt.plot(Ks_sorted, tau_sorted, marker='o')
    plt.axvline(Kc, color='k', ls='--', lw=1, label=f"Kc teórico ≈ {Kc:.3f}")
    plt.xlabel("K")
    plt.ylabel(r"$\tau_{\rm int}$")
    plt.title(r"Tiempo de autocorrelación integrado $\tau_{\rm int}$ vs K")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()