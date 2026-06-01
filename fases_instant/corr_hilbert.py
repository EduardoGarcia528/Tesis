import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from scipy.special import hyp2f1


# ============================================================
# Teoría discreta ideal
# ============================================================

def gamma_z_theory(tau, sigma=1.0):
    """
    Autocovarianza compleja teórica de la señal analítica z_n
    asociada a ruido blanco real x_n ~ N(0, sigma^2).

    Gamma_z(tau) = < z_{n+tau} conj(z_n) >

    Para la señal analítica discreta ideal:

        Gamma_z(0) = 2 sigma^2

        Gamma_z(tau) = 4 i sigma^2 / (pi tau), si tau impar
                     = 0, si tau par

    para tau != 0.
    """
    if tau == 0:
        return 2.0 * sigma**2

    if tau % 2 == 1:
        return 4j * sigma**2 / (np.pi * tau)
    else:
        return 0.0 + 0.0j


def rho_z_theory(tau, sigma=1.0):
    """
    Autocorrelación compleja normalizada:

        rho_z(tau) = Gamma_z(tau) / Gamma_z(0)

    Como Gamma_z(0) = 2 sigma^2:

        rho_z(tau) = 2 i / (pi tau), si tau impar
                   = 0, si tau par
    """
    return gamma_z_theory(tau, sigma=sigma) / gamma_z_theory(0, sigma=sigma)


def Ctheta_theory_from_rho(rho):
    """
    Correlación circular de fases teórica para dos variables complejas
    gaussianas circulares con correlación compleja rho.

        C_theta = < exp(i(theta_2 - theta_1)) >

    Si rho = 0, entonces C_theta = 0.

    Para |rho| <= 1:

        C_theta =
        (pi/4) rho * 2F1(1/2, 1/2; 2; |rho|^2)

    donde 2F1 es la función hipergeométrica de Gauss.
    """
    r = np.abs(rho)

    if r == 0:
        return 0.0 + 0.0j

    return (np.pi / 4.0) * rho * hyp2f1(0.5, 0.5, 2.0, r**2)


def Ctheta_theory(tau, sigma=1.0):
    """
    Correlación circular de fases teórica usando rho_z(tau).
    """
    rho = rho_z_theory(tau, sigma=sigma)
    return Ctheta_theory_from_rho(rho)


# ============================================================
# Estimación empírica
# ============================================================

def estimate_correlations(
    N=2**16,
    n_realizations=200,
    max_tau=20,
    sigma=1.0,
    seed=1234
):
    """
    Genera ruido blanco real, calcula la señal analítica con Hilbert,
    y estima Gamma_z(tau), rho_z(tau) y C_theta(tau).

    Parameters
    ----------
    N : int
        Longitud de cada realización.
    n_realizations : int
        Número de realizaciones independientes.
    max_tau : int
        Retardo máximo.
    sigma : float
        Desviación estándar del ruido blanco.
    seed : int
        Semilla aleatoria.

    Returns
    -------
    df : pandas.DataFrame
        Tabla con valores empíricos y teóricos.
    """

    rng = np.random.default_rng(seed)

    taus = np.arange(0, max_tau + 1)

    gamma_emp = np.zeros(len(taus), dtype=complex)
    rho_emp = np.zeros(len(taus), dtype=complex)
    ctheta_emp = np.zeros(len(taus), dtype=complex)

    for _ in range(n_realizations):

        # Ruido blanco real
        x = sigma * rng.standard_normal(N)

        # Señal analítica
        z = hilbert(x)

        # Fase instantánea
        theta = np.angle(z)

        # Varianza compleja empírica
        gamma0_emp = np.mean(z * np.conj(z))

        for j, tau in enumerate(taus):

            if tau == 0:
                z_tau = z
                z_0 = z
                theta_tau = theta
                theta_0 = theta
            else:
                z_tau = z[tau:]
                z_0 = z[:-tau]
                theta_tau = theta[tau:]
                theta_0 = theta[:-tau]

            # Gamma_z(tau) = < z_{n+tau} conj(z_n) >
            gamma = np.mean(z_tau * np.conj(z_0))

            # rho_z(tau) = Gamma_z(tau) / Gamma_z(0)
            rho = gamma / gamma0_emp

            # C_theta(tau) = < exp(i(theta_{n+tau} - theta_n)) >
            ctheta = np.mean(np.exp(1j * (theta_tau - theta_0)))

            gamma_emp[j] += gamma
            rho_emp[j] += rho
            ctheta_emp[j] += ctheta

    gamma_emp /= n_realizations
    rho_emp /= n_realizations
    ctheta_emp /= n_realizations

    gamma_th = np.array([gamma_z_theory(tau, sigma=sigma) for tau in taus])
    rho_th = np.array([rho_z_theory(tau, sigma=sigma) for tau in taus])
    ctheta_th = np.array([Ctheta_theory(tau, sigma=sigma) for tau in taus])

    df = pd.DataFrame({
        "tau": taus,

        "Gamma_emp_Re": gamma_emp.real,
        "Gamma_emp_Im": gamma_emp.imag,
        "Gamma_theo_Re": gamma_th.real,
        "Gamma_theo_Im": gamma_th.imag,
        "Gamma_abs_error": np.abs(gamma_emp - gamma_th),

        "rho_emp_Re": rho_emp.real,
        "rho_emp_Im": rho_emp.imag,
        "rho_theo_Re": rho_th.real,
        "rho_theo_Im": rho_th.imag,
        "rho_abs_error": np.abs(rho_emp - rho_th),

        "Ctheta_emp_Re": ctheta_emp.real,
        "Ctheta_emp_Im": ctheta_emp.imag,
        "Ctheta_theo_Re": ctheta_th.real,
        "Ctheta_theo_Im": ctheta_th.imag,
        "Ctheta_abs_error": np.abs(ctheta_emp - ctheta_th),

        "|Ctheta_emp|": np.abs(ctheta_emp),
        "|Ctheta_theo|": np.abs(ctheta_th),
    })

    return df


# ============================================================
# Ejecutar prueba
# ============================================================

df = estimate_correlations(
    N=2**16,
    n_realizations=200,
    max_tau=20,
    sigma=1.0,
    seed=2026
)

pd.set_option("display.precision", 5)
plt.figure(figsize=(8, 4))
plt.plot(df["tau"], df["rho_emp_Im"], "o-", label=r"Empírico $\operatorname{Im}\rho_z(\tau)$")
plt.plot(df["tau"], df["rho_theo_Im"], "k--", label=r"Teórico $\operatorname{Im}\rho_z(\tau)$")
plt.axhline(0, color="gray", lw=1)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\operatorname{Im}\rho_z(\tau)$")
plt.title("Autocorrelación compleja normalizada de la señal analítica")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
plt.plot(df["tau"], df["|Ctheta_emp|"], "o-", label=r"Empírico $|C_\theta(\tau)|$")
plt.plot(df["tau"], df["|Ctheta_theo|"], "k--", label=r"Teórico $|C_\theta(\tau)|$")
plt.axhline(0, color="gray", lw=1)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$|C_\theta(\tau)|$")
plt.title("Correlación circular de fases")
plt.legend()
plt.tight_layout()
plt.show()