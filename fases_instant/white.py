import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ============================================================
# Configuración
# ============================================================

N = 4000
n_realizations = 500
tau = 1
seed = 1234

M_fourier = 30
n_grid = 2048
sigma_smooth = None   # si quieres suavizado: por ejemplo sigma_smooth = M_fourier / 2

rng = np.random.default_rng(seed)

# ============================================================
# Funciones auxiliares
# ============================================================

def wrap_pi(a):
    """
    Lleva ángulos al intervalo (-pi, pi].
    """
    return (a + np.pi) % (2 * np.pi) - np.pi


def fase_hilbert(x):
    """
    Fase instantánea de Hilbert en (-pi, pi].
    """
    z = hilbert(x)
    return np.angle(z)


def alphas_from_phases(theta1, theta2):
    """
    Construye P_n = (theta1_n, theta2_n), vectores geodésicos
    y ángulos firmados alpha entre vectores consecutivos.
    """
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)

    L = min(len(theta1), len(theta2))
    theta1 = theta1[:L]
    theta2 = theta2[:L]

    if L < 4:
        return np.array([])

    # vectores geodésicos v_n = P_{n+1} - P_n en el toro
    dx = wrap_pi(np.diff(theta1))
    dy = wrap_pi(np.diff(theta2))

    v0x = dx[:-1]
    v0y = dy[:-1]
    v1x = dx[1:]
    v1y = dy[1:]

    norm0 = np.sqrt(v0x**2 + v0y**2)
    norm1 = np.sqrt(v1x**2 + v1y**2)

    mask = (norm0 > 0) & (norm1 > 0)

    v0x = v0x[mask]
    v0y = v0y[mask]
    v1x = v1x[mask]
    v1y = v1y[mask]
    norm0 = norm0[mask]
    norm1 = norm1[mask]

    if len(v0x) == 0:
        return np.array([])

    dot = v0x * v1x + v0y * v1y
    cross = v0x * v1y - v0y * v1x

    alpha = np.arctan2(cross, dot)
    alpha = alpha % (2 * np.pi)

    return alpha


def S_eff_from_alpha(alpha, M=30, n_grid=2048, sigma=None):
    """
    Estimador continuo de S a partir de la densidad angular de alpha,
    usando expansión truncada de Fourier.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha = alpha[np.isfinite(alpha)]

    if len(alpha) < 10:
        return np.nan

    m = np.arange(1, M + 1)

    c = np.array([
        np.mean(np.exp(1j * k * alpha))
        for k in m
    ])

    if sigma is not None:
        # Suavizado gaussiano espectral
        c = c * np.exp(-(m**2) / (2 * sigma**2))

    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

    f = np.full_like(grid, 1 / (2 * np.pi), dtype=float)

    for ck, k in zip(c, m):
        f += (1 / np.pi) * np.real(ck * np.exp(-1j * k * grid))

    # Evitar pequeñas oscilaciones negativas por truncamiento
    f = np.maximum(f, 1e-15)

    # Renormalizar
    dtheta = 2 * np.pi / n_grid
    f = f / np.sum(f * dtheta)

    h = -np.sum(f * np.log(f) * dtheta)
    S = np.exp(h) / (2 * np.pi)

    return S


def S_theta_bivariante(x, y):
    """
    Caso bivariante:
    P_n = (theta_x[n], theta_y[n]).
    """
    thx = fase_hilbert(x)
    thy = fase_hilbert(y)

    alpha = alphas_from_phases(thx, thy)
    return S_eff_from_alpha(alpha, M=M_fourier, n_grid=n_grid, sigma=sigma_smooth)


def S_hat_theta_bivariante(x, y, start=1):
    """
    Caso bivariante con fases submuestreadas:
    hat(theta) = theta_1, theta_3, theta_5, ...

    Esto fuerza que las transiciones consecutivas en la caminata
    correspondan a saltos de lag 2 en la fase original.
    """
    thx = fase_hilbert(x)
    thy = fase_hilbert(y)

    thx_hat = thx[start::2]
    thy_hat = thy[start::2]

    alpha = alphas_from_phases(thx_hat, thy_hat)
    return S_eff_from_alpha(alpha, M=M_fourier, n_grid=n_grid, sigma=sigma_smooth)


def S_theta_univariante(x, tau=1):
    """
    Caso univariante:
    P_n = (theta[n + tau], theta[n]).
    """
    th = fase_hilbert(x)

    theta1 = th[tau:]
    theta2 = th[:-tau]

    alpha = alphas_from_phases(theta1, theta2)
    return S_eff_from_alpha(alpha, M=M_fourier, n_grid=n_grid, sigma=sigma_smooth)


def S_theta_univariante_phase_shuffle(x, tau=1, rng=None):
    """
    Control univariante con shuffle de fases.

    No lo usaría como modelo nulo principal de dinámica si el reporte
    elimina esa propuesta; aquí sirve sólo como control geométrico para
    mostrar que el solapamiento de componentes ya induce S bajo.
    """
    if rng is None:
        rng = np.random.default_rng()

    th = fase_hilbert(x)
    th_perm = rng.permutation(th)

    theta1 = rng.permutation(th[tau:])
    theta2 = rng.permutation(th[:-tau])

    alpha = alphas_from_phases(theta1, theta2)
    return S_eff_from_alpha(alpha, M=M_fourier, n_grid=n_grid, sigma=sigma_smooth)


def S_theta_univariante_direct_shuffle(x, tau=1, rng=None):
    """
    Alternativa si quieres reemplazar Pi theta por Pi x en la figura:
    primero se barajea la serie original y luego se calcula la fase de Hilbert.

    Para ruido blanco, este control suele quedar muy cerca de S_theta
    porque conserva el efecto geométrico univariante y la correlación
    intrínseca inducida por Hilbert.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_perm = rng.permutation(x)
    return S_theta_univariante(x_perm, tau=tau)


# ============================================================
# Simulación
# ============================================================

S_bi = []
S_bi_hat = []
S_uni = []
S_uni_pitheta = []

for r in range(n_realizations):
    # Bivariante: dos ruidos blancos independientes
    x = rng.normal(size=N)
    y = rng.normal(size=N)

    S_bi.append(S_theta_bivariante(x, y))
    S_bi_hat.append(S_hat_theta_bivariante(x, y, start=1))

    # Univariante: una sola serie
    u = rng.normal(size=N)

    S_uni.append(S_theta_univariante(u, tau=tau))
    S_uni_pitheta.append(S_theta_univariante_phase_shuffle(u, tau=tau, rng=rng))

S_bi = np.asarray(S_bi, dtype=float)
S_bi_hat = np.asarray(S_bi_hat, dtype=float)
S_uni = np.asarray(S_uni, dtype=float)
S_uni_pitheta = np.asarray(S_uni_pitheta, dtype=float)

# Quitar posibles NaN
S_bi = S_bi[np.isfinite(S_bi)]
S_bi_hat = S_bi_hat[np.isfinite(S_bi_hat)]
S_uni = S_uni[np.isfinite(S_uni)]
S_uni_pitheta = S_uni_pitheta[np.isfinite(S_uni_pitheta)]


# ============================================================
# Resumen numérico
# ============================================================

def resumen(nombre, arr):
    print(f"{nombre:28s} media = {np.mean(arr):.6f}   std = {np.std(arr, ddof=1):.6f}")

print("\nResumen de distribuciones:\n")
resumen(r"$S_\theta^{bi}$", S_bi)
resumen(r"$S_{\hat\theta}^{bi}$", S_bi_hat)
resumen(r"$S_\theta^{uni}$", S_uni)
resumen(r"$S_\theta^{(\Pi\theta),uni}$", S_uni_pitheta)


# ============================================================
# Gráfica única
# ============================================================

all_values = np.concatenate([S_bi, S_bi_hat, S_uni, S_uni_pitheta])

xmin = np.floor((np.min(all_values) - 0.01) * 100) / 100
xmax = 1.001

bins = np.linspace(xmin, xmax, 80)

plt.figure(figsize=(8.0, 4.8))

plt.hist(
    S_uni,
    bins=bins,
    density=True,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.7,
    label=r"$S_{\theta}^{\mathrm{uni}}(\tau=1)$"
)

plt.hist(
    S_uni_pitheta,
    bins=bins,
    density=True,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.7,
    label=r"$S_{\theta}^{(\Pi\theta),\mathrm{uni}}(\tau=1)$"
)

plt.hist(
    S_bi,
    bins=bins,
    density=True,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.7,
    label=r"$S_{\theta}^{\mathrm{bi}}$"
)

plt.hist(
    S_bi_hat,
    bins=bins,
    density=True,
    alpha=0.55,
    edgecolor="black",
    linewidth=0.7,
    label=r"$S_{\hat{\theta}}^{\mathrm{bi}}$"
)

plt.xlabel(r"$S$")
plt.ylabel("Densidad")
plt.legend(frameon=True,fontsize=14)
plt.tight_layout()

plt.savefig("S_ruido_blanco_resumen.png", dpi=300, bbox_inches="tight")
plt.show()