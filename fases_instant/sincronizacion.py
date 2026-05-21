# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import hilbert

import mi_libreria as ml


# ============================================================
# Configuración
# ============================================================

RNG = np.random.default_rng(123)



# ============================================================
# Utilidades básicas
# ============================================================

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def fase_hilbert(x):
    """
    Fase instantánea usando Hilbert.

    Devuelve theta en [-pi, pi).
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    z = hilbert(x)
    return np.angle(z)


def delta_fase(theta):
    """
    Velocidad de fase envuelta.
    """
    return wrap_pi(np.diff(theta))


def align_lag(a, b, tau):
    """
    Alinea a_i con b_{i+tau}.

    tau > 0:
        compara a[:-tau] con b[tau:]

    tau < 0:
        compara a[-tau:] con b[:tau]

    tau = 0:
        compara a con b
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if tau > 0:
        return a[:-tau], b[tau:]
    elif tau < 0:
        return a[-tau:], b[:tau]
    else:
        return a, b


# ============================================================
# Modelos dinámicos
# ============================================================

def logistic_map(r=4.0, x0=0.12345, n_iter=10000, n_transient=2000):
    x = np.empty(n_iter + n_transient)
    x[0] = x0

    for i in range(n_iter + n_transient - 1):
        x[i+1] = r*x[i]*(1 - x[i])

    return x[n_transient:]


def henon_map(a=1.4, b=0.3, x0=0.1, y0=0.1, n_iter=10000, n_transient=2000):
    x = np.empty(n_iter + n_transient)
    y = np.empty(n_iter + n_transient)

    x[0] = x0
    y[0] = y0

    for i in range(n_iter + n_transient - 1):
        x[i+1] = 1 - a*x[i]**2 + y[i]
        y[i+1] = b*x[i]

    return x[n_transient:], y[n_transient:]


def mezcla_fuente_comun(N=10000, p=0.5, tauu=0, rng=None):
    """
    Caso 1:
        tauu = 0:
        x = (1-p)*xi + p*z
        y = (1-p)*eta + p*z

    Caso 2:
        tauu > 0:
        x = (1-p)*xi + p*z[tauu:]
        y = (1-p)*eta + p*z[:-tauu]
    """
    if rng is None:
        rng = np.random.default_rng()

    if tauu == 0:
        xi = rng.uniform(size=N)
        eta = rng.uniform(size=N)
        z = rng.uniform(size=N)

        x = (1-p)*xi + p*z
        y = (1-p)*eta + p*z

    else:
        xi = rng.uniform(size=N - tauu)
        eta = rng.uniform(size=N - tauu)
        z = rng.uniform(size=N)

        x = (1-p)*xi + p*z[tauu:]
        y = (1-p)*eta + p*z[:-tauu]

    return x, y


# ============================================================
# Sincronización de fase clásica
# ============================================================

def PLV_theta(x, y, tau=0, n=1, m=1):
    """
    Phase Locking Value clásico:

        PLV = | < exp(i(n theta_x - m theta_y_tau)) > |

    tau compara theta_x[i] con theta_y[i+tau].
    """
    theta_x = fase_hilbert(x)
    theta_y = fase_hilbert(y)

    tx, ty = align_lag(theta_x, theta_y, tau)
    L = min(len(tx), len(ty))
    tx = tx[:L]
    ty = ty[:L]

    return np.abs(np.mean(np.exp(1j*(n*tx - m*ty))))


def PLV_delta(x, y, tau=0):
    """
    PLV aplicado a velocidades de fase:

        | < exp(i(Delta theta_x - Delta theta_y_tau)) > |

    No es sincronización de fase clásica, pero es una comparación
    cercana a J_{Delta theta}.
    """
    theta_x = fase_hilbert(x)
    theta_y = fase_hilbert(y)

    dtx = delta_fase(theta_x)
    dty = delta_fase(theta_y)

    vx, vy = align_lag(dtx, dty, tau)
    L = min(len(vx), len(vy))
    vx = vx[:L]
    vy = vy[:L]

    return np.abs(np.mean(np.exp(1j*(vx - vy))))


def PLV_theta_shuffle_distribution(x, y, tau=0, n_surr=200, rng=None):
    """
    Nulo shuffle para PLV_theta.

    Se baraja theta_y después de alinear.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_x = fase_hilbert(x)
    theta_y = fase_hilbert(y)

    tx, ty = align_lag(theta_x, theta_y, tau)
    L = min(len(tx), len(ty))
    tx = tx[:L]
    ty = ty[:L]

    vals = np.empty(n_surr)

    for s in range(n_surr):
        ty_s = rng.permutation(ty)
        vals[s] = np.abs(np.mean(np.exp(1j*(tx - ty_s))))

    return vals


def PLV_delta_shuffle_distribution(x, y, tau=0, n_surr=200, rng=None):
    """
    Nulo shuffle para PLV_delta.

    Se barajan independientemente Delta theta_x y Delta theta_y.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_x = fase_hilbert(x)
    theta_y = fase_hilbert(y)

    dtx = delta_fase(theta_x)
    dty = delta_fase(theta_y)

    vx, vy = align_lag(dtx, dty, tau)
    L = min(len(vx), len(vy))
    vx = vx[:L]
    vy = vy[:L]

    vals = np.empty(n_surr)

    for s in range(n_surr):
        vx_s = rng.permutation(vx)
        vy_s = rng.permutation(vy)
        vals[s] = np.abs(np.mean(np.exp(1j*(vx_s - vy_s))))

    return vals

# ============================================================
# Tu medida J_{Delta theta}
# ============================================================

def J_delta_obs(x, y=None, tau=1):
    """
    Wrapper para tu índice J_{Delta theta}.

    Caso univariante:
        ml.indice_S_eff(x, seriey=None, tau=tau)

    Caso bivariante:
        ml.indice_S_eff(x, seriey=y, tau=tau)

    """
    if y is None:
        # alphas = ml.angulos_alpha_H(x,seriey=None, tau=tau)
        # return ml.entropia_shannon(alphas, False)
        return ml.indice_S_eff_fast(x, seriey=None, tau=tau, null="no",modo_univariante="global")
    # alphas = ml.angulos_alpha_H(x,seriey=y, tau=tau)
    # return ml.entropia_shannon(alphas, False)
    return ml.indice_S_eff_fast(x, seriey=y, tau=tau, null="no",modo_univariante="global")


def J_delta_shuffle_distribution(x, y=None, tau=1, n_surr=200):
    """
    Distribución nula de J_{Delta theta} usando tu null='shuffle'.

    Nota:
    Si ml.indice_S_eff(..., null='shuffle') genera un solo shuffle por llamada,
    esta función repite n_surr veces.
    """
    vals = np.empty(n_surr)

    for s in range(n_surr):
        if y is None:
            # alphas = ml.angulos_alpha_H(x, seriey=None, tau=tau,null="shuffle")
            # vals[s] = ml.entropia_shannon(alphas, False)
            x = np.random.permutation(x)
            vals[s] = ml.indice_S_eff_fast(x, seriey=None, tau=tau, null="no", modo_univariante="global")
        else:
            # alphas = ml.angulos_alpha_H(x,seriey=y, tau=tau,null="shuffle")
            # vals[s] = ml.entropia_shannon(alphas, False)
            x = np.random.permutation(x)
            y = np.random.permutation(y)
            vals[s] = ml.indice_S_eff_fast(x, seriey=y, tau=tau, null="no", modo_univariante="global")

    return vals


def zscore_from_null(obs, null_values, orientation="obs_greater"):
    """
    orientation:
        'obs_greater':
            Z = (obs - mean(null))/std(null)

        'obs_less':
            Z = (mean(null) - obs)/std(null)

    Para PLV normalmente usamos obs_greater.
    Para tu J_delta usamos obs_less, porque dependencia implica:
        J_obs < J_shuffle
    """
    mu = np.mean(null_values)
    sd = np.std(null_values, ddof=1)

    if sd == 0:
        return np.nan

    if orientation == "obs_greater":
        return (obs - mu) / sd
    elif orientation == "obs_less":
        return (mu - obs) / sd
    else:
        raise ValueError("orientation debe ser 'obs_greater' u 'obs_less'.")


def empirical_pvalue(obs, null_values, alternative="greater"):
    """
    p-value empírico con corrección +1.

    alternative='greater':
        p = P(null >= obs)

    alternative='less':
        p = P(null <= obs)
    """
    null_values = np.asarray(null_values)
    n = len(null_values)

    if alternative == "greater":
        count = np.sum(null_values >= obs)
    elif alternative == "less":
        count = np.sum(null_values <= obs)
    else:
        raise ValueError("alternative debe ser 'greater' o 'less'.")

    return (count + 1) / (n + 1)


# ============================================================
# Comparación para un par x,y en una lista de taus
# ============================================================

def comparar_medidas_tau(x, y,y2, taus, n_surr=200, rng=None):
    """
    Calcula, para cada tau:

    - PLV_theta
    - Z_PLV_theta
    - PLV_delta
    - Z_PLV_delta
    - J_delta
    - J_delta_shuffle_mean
    - Z_J_delta

    Para J_delta se usa la convención:
        Z_J_delta = (mean(J_shuffle) - J_obs)/std(J_shuffle)
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []

    for tau in taus:
        # ---------- PLV clásico ----------
        plv_t = PLV_theta(x, y, tau=tau)
        plv_t_null = PLV_theta_shuffle_distribution(
            x, y, tau=tau, n_surr=n_surr, rng=rng
        )

        z_plv_t = zscore_from_null(
            plv_t, plv_t_null, orientation="obs_greater"
        )

        p_plv_t = empirical_pvalue(
            plv_t, plv_t_null, alternative="greater"
        )

        # ---------- PLV sobre Delta theta ----------
        plv_d = PLV_delta(x, y, tau=tau)
        plv_d_null = PLV_delta_shuffle_distribution(
            x, y, tau=tau, n_surr=n_surr, rng=rng
        )

        z_plv_d = zscore_from_null(
            plv_d, plv_d_null, orientation="obs_greater"
        )

        p_plv_d = empirical_pvalue(
            plv_d, plv_d_null, alternative="greater"
        )

        # ---------- Tu J_delta ----------
        J_obs = J_delta_obs(x, y=y2, tau=tau)
        J_null = J_delta_shuffle_distribution(
            x, y=y2, tau=tau, n_surr=n_surr
        )

        z_J = zscore_from_null(
            J_obs, J_null, orientation="obs_less"
        )

        p_J = empirical_pvalue(
            J_obs, J_null, alternative="less"
        )

        rows.append({
            "tau": tau,

            "PLV_theta": plv_t,
            "PLV_theta_null_mean": np.mean(plv_t_null),
            "PLV_theta_null_std": np.std(plv_t_null, ddof=1),
            "Z_PLV_theta": z_plv_t,
            "p_PLV_theta": p_plv_t,

            "PLV_delta": plv_d,
            "PLV_delta_null_mean": np.mean(plv_d_null),
            "PLV_delta_null_std": np.std(plv_d_null, ddof=1),
            "Z_PLV_delta": z_plv_d,
            "p_PLV_delta": p_plv_d,

            "J_delta": J_obs,
            "J_delta_null_mean": np.mean(J_null),
            "J_delta_null_std": np.std(J_null, ddof=1),
            "D_J_delta": np.mean(J_null) - J_obs,
            "Z_J_delta": z_J,
            "p_J_delta": p_J,
        })

    return pd.DataFrame(rows)


# ============================================================
# Comparación para barrido de p
# ============================================================

def comparar_medidas_p(ps, tau_eval=0, tauu_modelo=0, N=10000, n_surr=200, rng=None):
    """
    Barrido de p para el modelo de fuente común.

    tauu_modelo=0:
        x=(1-p)xi+pz
        y=(1-p)eta+pz

    tauu_modelo=1:
        x=(1-p)xi+pz[1:]
        y=(1-p)eta+pz[:-1]

    tau_eval:
        retardo usado al calcular PLV y J.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []

    for p in ps:
        x, y = mezcla_fuente_comun(
            N=N, p=p, tauu=tauu_modelo, rng=rng
        )

        df_tau = comparar_medidas_tau(
            x, y,y2=y, taus=[tau_eval], n_surr=n_surr, rng=rng
        )

        row = df_tau.iloc[0].to_dict()
        row["p"] = p
        row["tauu_modelo"] = tauu_modelo
        row["tau_eval"] = tau_eval

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Gráficas
# ============================================================

def plot_zscores_tau(df, title=None):
    print(title)
    plt.figure(figsize=(8, 4))

    plt.plot(df["tau"], df["Z_PLV_theta"], marker="o", ms=3,
             label=r"$Z_{\mathrm{PLV}_\theta}$")

    plt.plot(df["tau"], df["Z_PLV_delta"], marker="o", ms=3,
             label=r"$Z_{\mathrm{PLV}_{\Delta\theta}}$")

    plt.plot(df["tau"], df["Z_J_delta"], marker="o", ms=3,
             label=r"$Z_{J_{\Delta\theta}}$")

    plt.axhline(0, lw=1, color="k")
    plt.xlabel(r"$\tau$")
    plt.ylabel("Z-score")
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/zscores_tau_{title.replace(' ', '_')}.png", dpi=300)
    plt.show(block = False)
    plt.pause(5)
    plt.close()


def plot_raw_tau(df, title=None):
    print(title)
    plt.figure(figsize=(8, 4))

    plt.plot(df["tau"], df["PLV_theta"], marker="o", ms=3,
             label=r"$\mathrm{PLV}_\theta$")

    plt.plot(df["tau"], df["PLV_delta"], marker="o", ms=3,
             label=r"$\mathrm{PLV}_{\Delta\theta}$")

    plt.plot(df["tau"], df["J_delta"], marker="o", ms=3,
             label=r"$J_{\Delta\theta}$")

    plt.plot(df["tau"], df["J_delta_null_mean"], marker="o", ms=3,
             label=r"$\langle J_{\Delta\theta,\mathrm{shuffle}}\rangle$")

    plt.xlabel(r"$\tau$")
    plt.ylabel("Valor")
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/raw_tau_{title.replace(' ', '_')}.png", dpi=300)
    plt.show(block = False)
    plt.pause(5)
    plt.close()


def plot_p_sweep(df, title=None):
    print(title)
    plt.figure(figsize=(8, 4))

    plt.plot(df["p"], df["Z_PLV_theta"], marker="o", ms=3,
             label=r"$Z_{\mathrm{PLV}_\theta}$")

    plt.plot(df["p"], df["Z_PLV_delta"], marker="o", ms=3,
             label=r"$Z_{\mathrm{PLV}_{\Delta\theta}}$")

    plt.plot(df["p"], df["Z_J_delta"], marker="o", ms=3,
             label=r"$Z_{J_{\Delta\theta}}$")

    plt.axhline(0, lw=1, color="k")
    plt.xlabel(r"$p$")
    plt.ylabel("Z-score")
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/zscores_p_{title.replace(' ', '_')}.png", dpi=300)
    plt.show(block = False)
    plt.pause(5)
    plt.close()


def plot_p_sweep_raw_J(df, title=None):
    print(title)
    plt.figure(figsize=(8, 4))

    plt.plot(df["p"], df["J_delta"], marker="o", ms=3,
             label=r"$J_{\Delta\theta}$")

    plt.plot(df["p"], df["J_delta_null_mean"], marker="o", ms=3,
             label=r"$\langle J_{\Delta\theta,\mathrm{shuffle}}\rangle$")

    plt.xlabel(r"$p$")
    plt.ylabel("J")
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/raw_p_{title.replace(' ', '_')}.png", dpi=300)
    plt.show(block = False)
    plt.pause(5)
    plt.close()


# ============================================================
# Pruebas principales
# ============================================================

def ejecutar_pruebas_basicas(
    N=10000,
    n_surr=100,
    tau_max=10,
    rng=None
):
    """
    Ejecuta las pruebas principales y devuelve un diccionario de DataFrames.
    """
    if rng is None:
        rng = np.random.default_rng()

    taus = np.arange(-tau_max, tau_max + 1)

    resultados = {}

    # --------------------------------------------------------
    # A) Ruido independiente
    # --------------------------------------------------------
    x = rng.uniform(size=N)
    y = rng.uniform(size=N)

    df = comparar_medidas_tau(x, y,y2=y, taus=taus, n_surr=n_surr, rng=rng)
    resultados["ruido_independiente"] = df

    plot_zscores_tau(df, title="Ruido blanco independiente")
    plot_raw_tau(df, title="Ruido blanco independiente")

    # --------------------------------------------------------
    # B) Identidad instantánea: y=x
    # --------------------------------------------------------
    x = rng.uniform(size=N)
    y = x.copy()

    df = comparar_medidas_tau(x, y,y2=None, taus=taus, n_surr=n_surr, rng=rng)
    resultados["identidad_instantanea"] = df

    plot_zscores_tau(df, title=r"Identidad instantánea y=x")
    plot_raw_tau(df, title=r"Identidad instantánea y=x")

    # --------------------------------------------------------
    # C) Copia retardada: x_i = z_{i+tau0}, y_i = z_i
    # --------------------------------------------------------
    tau0 = 1
    z = rng.uniform(size=N + tau0)
    x = z[tau0:]
    y = z[:-tau0]

    df = comparar_medidas_tau(x, y,y2=y, taus=taus, n_surr=n_surr, rng=rng)
    resultados["copia_retardada_tau1"] = df

    plot_zscores_tau(df, title=r"Copia retardada x_i=z_{i+1}, y_i=z_i")
    plot_raw_tau(df, title=r"Copia retardada x_i=z_{i+1}, y_i=z_i")

    # --------------------------------------------------------
    # D) Hénon x,y
    # --------------------------------------------------------
    x, y = henon_map(a=1.4, b=0.3, n_iter=N, n_transient=2000)

    df = comparar_medidas_tau(x, y, y2=y,taus=taus, n_surr=n_surr, rng=rng)
    resultados["henon_xy"] = df

    plot_zscores_tau(df, title=r"Hénon bivariante x,y")
    plot_raw_tau(df, title=r"Hénon bivariante x,y")

    # --------------------------------------------------------
    # E) Hénon x contra logístico independiente
    # --------------------------------------------------------
    x_h, _ = henon_map(a=1.4, b=0.3, n_iter=N, n_transient=2000)
    y_log = logistic_map(r=4.0, x0=0.23456, n_iter=N, n_transient=2000)

    L = min(len(x_h), len(y_log))
    x_h = x_h[:L]
    y_log = y_log[:L]

    df = comparar_medidas_tau(x_h, y_log,y2=y_log, taus=taus, n_surr=n_surr, rng=rng)
    resultados["henon_vs_logistico"] = df

    plot_zscores_tau(df, title=r"Hénon x vs logístico independiente")
    plot_raw_tau(df, title=r"Hénon x vs logístico independiente")

    return resultados


# ============================================================
# Pruebas de fuente común con barrido de p
# ============================================================

def ejecutar_barridos_p(
    N=10000,
    n_surr=100,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    ps = np.linspace(0, 1, 21)

    # Caso sin retardo:
    # x=(1-p)xi+pz
    # y=(1-p)eta+pz
    df_p_instantaneo = comparar_medidas_p(
        ps,
        tau_eval=0,
        tauu_modelo=0,
        N=N,
        n_surr=n_surr,
        rng=rng
    )

    plot_p_sweep(
        df_p_instantaneo,
        title=r"Fuente común instantánea x=(1-p)xi+pz, y=(1-p)eta+pz"
    )

    plot_p_sweep_raw_J(
        df_p_instantaneo,
        title=r"J en fuente común instantánea"
    )

    # Caso retardado:
    # x=(1-p)xi+pz[1:]
    # y=(1-p)eta+pz[:-1]
    df_p_retardado = comparar_medidas_p(
        ps,
        tau_eval=0,
        tauu_modelo=1,
        N=N,
        n_surr=n_surr,
        rng=rng
    )

    plot_p_sweep(
        df_p_retardado,
        title=r"Fuente común retardada x=(1-p)xi+pz_{i+1}, y=(1-p)eta+pz_i"
    )

    plot_p_sweep_raw_J(
        df_p_retardado,
        title=r"J en fuente común retardada"
    )

    return df_p_instantaneo, df_p_retardado


# ============================================================
# Ejecución
# ============================================================

if __name__ == "__main__":

    # Para pruebas rápidas usa n_surr=50.
    # Para resultados finales usa n_surr >= 200 o 500.
    resultados_tau = ejecutar_pruebas_basicas(
        N=10000,
        n_surr=100,
        tau_max=10,
        rng=RNG
    )

    df_p_instantaneo, df_p_retardado = ejecutar_barridos_p(
        N=10000,
        n_surr=100,
        rng=RNG
    )

    # Guardar resultados
    for name, df in resultados_tau.items():
        df.to_csv(f"comparacion_PLV_J_{name}.csv", index=False)

    df_p_instantaneo.to_csv("comparacion_PLV_J_fuente_comun_instantanea.csv", index=False)
    df_p_retardado.to_csv("comparacion_PLV_J_fuente_comun_retardada.csv", index=False)

    print("Listo. Resultados guardados en CSV.")