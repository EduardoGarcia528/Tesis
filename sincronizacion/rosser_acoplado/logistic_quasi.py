import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import mi_libreria as ml


# ============================================================
# PARÁMETROS GENERALES
# ============================================================

OMEGA = (np.sqrt(5.0) - 1.0) / 2.0  # número áureo conjugado

N_TRANSITORIO = 50_000
N_ANALISIS = 50_000

X0 = 0.2
THETA0 = 0.123456789


# ============================================================
# MAPA LOGÍSTICO CUASIPERIÓDICAMENTE FORZADO
# ============================================================

def logistic_quasiperiodic(
    alpha,
    epsilon,
    n_transitorio=N_TRANSITORIO,
    n_analisis=N_ANALISIS,
    x0=X0,
    theta0=THETA0,
    omega=OMEGA,
):
    """
    Genera la serie x_n del mapa logístico cuasiperiódicamente forzado:

        x_{n+1} =
            alpha * [1 + epsilon*cos(2*pi*theta_n)]
            * x_n * (1 - x_n)

        theta_{n+1} = (theta_n + omega) mod 1

    Además calcula el exponente de Lyapunov transversal:

        lambda_x =
            < log | alpha [1 + epsilon cos(2*pi*theta_n)]
                    (1 - 2*x_n) | >

    El promedio de Lyapunov se realiza únicamente durante
    la región de análisis, después del transitorio.
    """

    total = n_transitorio + n_analisis

    x = float(x0)
    theta = float(theta0)

    serie = np.empty(n_analisis, dtype=float)

    suma_lyap = 0.0
    idx = 0

    tiny = np.finfo(float).tiny

    for n in range(total):

        factor = alpha * (
            1.0 + epsilon * np.cos(2.0 * np.pi * theta)
        )

        # Derivada respecto a x_n
        derivada = factor * (1.0 - 2.0 * x)

        if n >= n_transitorio:
            serie[idx] = x

            suma_lyap += np.log(
                max(abs(derivada), tiny)
            )

            idx += 1

        # Evolución del mapa
        x = factor * x * (1.0 - x)
        theta = (theta + omega) % 1.0

    lyap = suma_lyap / n_analisis

    return serie, lyap


# ============================================================
# CONVERSIÓN epsilon' -> epsilon
# ============================================================

def epsilon_from_epsilon_prime(alpha, epsilon_prime):
    """
    La parametrización usada en algunos trabajos define

        epsilon' = epsilon / (4/alpha - 1)

    por lo tanto

        epsilon = epsilon' * (4/alpha - 1)
    """

    return epsilon_prime * (4.0 / alpha - 1.0)


# ============================================================
# CÁLCULO DE J Y LYAPUNOV
# ============================================================

def calcular_J_y_lyap(
    alpha,
    epsilon,
    n_transitorio=N_TRANSITORIO,
    n_analisis=N_ANALISIS,
):
    serie, lyap = logistic_quasiperiodic(
        alpha=alpha,
        epsilon=epsilon,
        n_transitorio=n_transitorio,
        n_analisis=n_analisis,
    )

    J = ml.indice_J(serie)

    return J, lyap, serie


# ============================================================
# PRUEBA 1
# TRES CASOS DE REFERENCIA
# ============================================================

def prueba_tres_regimenes():

    casos = {
        "Toro": {
            "alpha": 3.4874,
            "epsilon_prime": 0.3,
        },

        "SNA": {
            "alpha": 3.4880,
            "epsilon_prime": 0.3,
        },

        "Caos": {
            "alpha": 3.6000,
            "epsilon_prime": 0.5,
        },
    }

    resultados = {}

    print("\n")
    print("=" * 70)
    print("PRUEBA DE TRES REGÍMENES")
    print("=" * 70)

    for nombre, pars in casos.items():

        alpha = pars["alpha"]
        eps_prime = pars["epsilon_prime"]

        epsilon = epsilon_from_epsilon_prime(
            alpha,
            eps_prime,
        )

        J, lyap, serie = calcular_J_y_lyap(
            alpha=alpha,
            epsilon=epsilon,
        )

        resultados[nombre] = {
            "alpha": alpha,
            "epsilon_prime": eps_prime,
            "epsilon": epsilon,
            "J": J,
            "lyap": lyap,
            "serie": serie,
        }

        print(f"\n{nombre}")
        print("-" * 40)
        print(f"alpha       = {alpha:.7f}")
        print(f"epsilon'    = {eps_prime:.7f}")
        print(f"epsilon     = {epsilon:.7f}")
        print(f"lambda      = {lyap:.8f}")
        print(f"J           = {J:.8f}")

    # --------------------------------------------------------
    # Figura de las series
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    n_plot = 1500

    for ax, (nombre, r) in zip(
        axes,
        resultados.items(),
    ):

        serie = r["serie"]

        ax.plot(
            np.arange(n_plot),
            serie[:n_plot],
            lw=0.8,
        )

        ax.set_ylabel(r"$x_n$")

        ax.set_title(
            f"{nombre} | "
            rf"$\alpha={r['alpha']}$, "
            rf"$\lambda={r['lyap']:.4f}$, "
            rf"$J={r['J']:.4f}$"
        )

        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(r"$n$")

    fig.tight_layout()

    fig.savefig(
        "SNA_tres_regimenes_series.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # Retratos (theta, x)
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
    )

    for ax, (nombre, r) in zip(
        axes,
        resultados.items(),
    ):

        alpha = r["alpha"]
        epsilon = r["epsilon"]

        theta_vals, x_vals = generar_atractor(
            alpha=alpha,
            epsilon=epsilon,
            n_transitorio=N_TRANSITORIO,
            n_analisis=20_000,
        )

        ax.scatter(
            theta_vals,
            x_vals,
            s=0.15,
            rasterized=True,
        )

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$x$")

        ax.set_title(
            f"{nombre}\n"
            rf"$\lambda={r['lyap']:.4f}$, "
            rf"$J={r['J']:.4f}$"
        )

    fig.tight_layout()

    fig.savefig(
        "SNA_tres_regimenes_atractores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return resultados


# ============================================================
# GENERAR ATRACTOR (theta_n, x_n)
# ============================================================

def generar_atractor(
    alpha,
    epsilon,
    n_transitorio=50_000,
    n_analisis=20_000,
    x0=X0,
    theta0=THETA0,
    omega=OMEGA,
):

    x = float(x0)
    theta = float(theta0)

    for _ in range(n_transitorio):

        factor = alpha * (
            1.0 + epsilon * np.cos(2.0 * np.pi * theta)
        )

        x = factor * x * (1.0 - x)
        theta = (theta + omega) % 1.0

    xs = np.empty(n_analisis)
    thetas = np.empty(n_analisis)

    for i in range(n_analisis):

        xs[i] = x
        thetas[i] = theta

        factor = alpha * (
            1.0 + epsilon * np.cos(2.0 * np.pi * theta)
        )

        x = factor * x * (1.0 - x)
        theta = (theta + omega) % 1.0

    return thetas, xs


# ============================================================
# PRUEBA 2
# BARRIDO EN alpha PARA epsilon = 0.05
# ============================================================

def barrido_alpha(
    alpha_min=3.33,
    alpha_max=3.43,
    delta_alpha=0.0002,
    epsilon=0.05,
):

    alphas = np.arange(
        alpha_min,
        alpha_max + delta_alpha / 2.0,
        delta_alpha,
    )

    Js = np.empty(len(alphas))
    lyaps = np.empty(len(alphas))

    print("\n")
    print("=" * 70)
    print("BARRIDO EN alpha")
    print("=" * 70)

    print(f"epsilon = {epsilon}")
    print(f"N alpha = {len(alphas)}")
    print(
        f"alpha = [{alpha_min}, {alpha_max}] "
        f"con delta = {delta_alpha}"
    )

    for i, alpha in enumerate(
        tqdm(alphas, desc="Calculando")
    ):

        J, lyap, _ = calcular_J_y_lyap(
            alpha=alpha,
            epsilon=epsilon,
        )

        Js[i] = J
        lyaps[i] = lyap

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    np.savez(
        "SNA_barrido_J_resultados.npz",
        alpha=alphas,
        J=Js,
        lyap=lyaps,
        epsilon=epsilon,
        omega=OMEGA,
        n_transitorio=N_TRANSITORIO,
        n_analisis=N_ANALISIS,
    )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    # J
    axes[0].plot(
        alphas,
        Js,
        lw=1.2,
    )

    axes[0].set_ylabel(r"$J$")
    axes[0].grid(alpha=0.25)

    # Lyapunov
    axes[1].plot(
        alphas,
        lyaps,
        lw=1.2,
    )

    axes[1].axhline(
        0.0,
        ls="--",
        lw=1.0,
    )

    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\lambda_x$")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        rf"Mapa logístico cuasiperiódicamente forzado, "
        rf"$\epsilon={epsilon}$"
    )

    fig.tight_layout()

    fig.savefig(
        "SNA_barrido_J_vs_Lyapunov.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return alphas, Js, lyaps


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Toro vs SNA vs caos
    # --------------------------------------------------------

    resultados = prueba_tres_regimenes()

    # --------------------------------------------------------
    # 2. Barrido completo
    # --------------------------------------------------------

    alphas, Js, lyaps = barrido_alpha(
        alpha_min=3.33,
        alpha_max=3.43,
        delta_alpha=0.0002,
        epsilon=0.05,
    )