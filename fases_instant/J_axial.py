# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mi_libreria as ml


# ============================================================
# 1. Funciones base
# ============================================================

def J_from_fases(f1, f2, order=1):
    """
    Calcula J_q desde dos secuencias angulares f1, f2.

    order=1:
        J_1 = 1 - |<exp(i alpha_j)>|

    order=2:
        J_2 = 1 - |<exp(2i alpha_j)>|
    """
    puntos = ml.construir_puntos_toro(f1, f2)
    vectores = ml.construir_vectores_geodesicos(puntos)
    angulos = ml.calcular_angulos_entre_vectores(vectores)

    if len(angulos) == 0:
        return np.nan
    
    return 1.0 - np.abs(np.mean(np.exp(1j * order * angulos)))


def obtener_delta_fases(x, y, tau=0):
    """
    Usa tu función obtener_fases_instantaneas para obtener:

        f1 = Delta theta^x
        f2 = Delta theta^y desplazada por tau

    En este script se usa sólo caso bivariante.
    """
    f1, f2 = ml.obtener_fases_instantaneas(
        x,
        seriey=y,
        tau=tau,
        quitar_media=True,
        unwrap=False,
        modo_univariante="segmentos",
        null="no"
    )

    L = min(len(f1), len(f2))
    return np.asarray(f1[:L]), np.asarray(f2[:L])


def null_distribution(f1, f2,x,y,tau, order=1, null="shuffle", n_surr=200,
                      min_shift=50, rng=None):
    """
    Distribución nula para J_order.

    null="shuffle":
        barajea f1 y f2 independientemente.

    null="s":
        corrimiento circular relativo de f2.
        Preserva la estructura temporal interna de f1 y f2.
    """
    if rng is None:
        rng = np.random.default_rng()

    f1 = np.asarray(f1)
    f2 = np.asarray(f2)

    L = min(len(f1), len(f2))
    f1 = f1[:L]
    f2 = f2[:L]

    vals = np.empty(n_surr)

    for k in range(n_surr):

        if null == "shuffle":
            f1_s = rng.permutation(f1)
            f2_s = rng.permutation(f2)

        elif null == "s":
            if L <= 2 * min_shift + 1:
                raise ValueError("La serie es demasiado corta para min_shift.")

            s = rng.integers(min_shift, L - min_shift)
            f1_s = f1
            f2_s = np.roll(f2, s)

        else:
            raise ValueError("null debe ser 'shuffle' o 's'.")

        if order == 2 and null == "shuffle":
            vals[k] = ml.indice_S_eff_fast(x,y, tau=tau, null=null)
        else:
            vals[k] = J_from_fases(f1_s, f2_s, order=order)

    return vals


def zscore_dependencia(J_obs, J_null):
    """
    Convención:

        Z > 0  si J_obs < <J_null>

    Es decir, positivo significa que la trayectoria real es
    más organizada que el nulo.
    """
    mu = np.mean(J_null)
    sd = np.std(J_null, ddof=1)

    if sd == 0:
        return np.nan

    return (mu - J_obs) / sd


def pvalue_left(J_obs, J_null):
    """
    p-value empírico para la alternativa:

        J_obs < J_null

    Es decir, dependencia geométrica.
    """
    J_null = np.asarray(J_null)
    return (np.sum(J_null <= J_obs) + 1) / (len(J_null) + 1)


# ============================================================
# 2. Mapas deterministas
# ============================================================

def logistic_map(r=4.0, x0=0.123456, n_iter=12000, n_transient=2000):
    x = np.empty(n_iter + n_transient)
    x[0] = x0

    for n in range(n_iter + n_transient - 1):
        x[n + 1] = r * x[n] * (1.0 - x[n])

    return x[n_transient:]


def sine_map(a=0.99, x0=0.234567, n_iter=12000, n_transient=2000):
    """
    x_{n+1} = a sin(pi x_n)
    """
    x = np.empty(n_iter + n_transient)
    x[0] = x0

    for n in range(n_iter + n_transient - 1):
        x[n + 1] = a * np.sin(np.pi * x[n])

    return x[n_transient:]


def chebyshev_map(k=3, x0=0.123456, n_iter=12000, n_transient=2000):
    """
    x_{n+1} = cos(k arccos(x_n))
    Serie en [-1,1].
    """
    x = np.empty(n_iter + n_transient)
    x[0] = x0

    for n in range(n_iter + n_transient - 1):
        x[n + 1] = np.cos(k * np.arccos(np.clip(x[n], -1, 1)))

    return x[n_transient:]


def henon_map(a=1.4, b=0.3, x0=0.1, y0=0.1,
              n_iter=12000, n_transient=2000):
    x = np.empty(n_iter + n_transient)
    y = np.empty(n_iter + n_transient)

    x[0] = x0
    y[0] = y0

    for n in range(n_iter + n_transient - 1):
        x[n + 1] = 1.0 - a * x[n]**2 + y[n]
        y[n + 1] = b * x[n]

    return x[n_transient:], y[n_transient:]


def ikeda_map(u=0.9, x0=0.1, y0=0.1,
              n_iter=12000, n_transient=2000):
    """
    Mapa de Ikeda.
    """
    x = np.empty(n_iter + n_transient)
    y = np.empty(n_iter + n_transient)

    x[0] = x0
    y[0] = y0

    for n in range(n_iter + n_transient - 1):
        t = 0.4 - 6.0 / (1.0 + x[n]**2 + y[n]**2)
        x[n + 1] = 1.0 + u * (x[n] * np.cos(t) - y[n] * np.sin(t))
        y[n + 1] = u * (x[n] * np.sin(t) + y[n] * np.cos(t))

    return x[n_transient:], y[n_transient:]


def standard_map(K=1.2, q0=0.1, p0=0.2,
                 n_iter=12000, n_transient=2000):
    """
    Mapa estándar.
    Devuelve q y p en [0, 2pi).
    """
    q = np.empty(n_iter + n_transient)
    p = np.empty(n_iter + n_transient)

    q[0] = q0
    p[0] = p0

    twopi = 2.0 * np.pi

    for n in range(n_iter + n_transient - 1):
        p[n + 1] = (p[n] + K * np.sin(q[n])) % twopi
        q[n + 1] = (q[n] + p[n + 1]) % twopi

    return q[n_transient:], p[n_transient:]


def normalize_series(x):
    """
    Normaliza sólo para evitar escalas muy distintas antes de Hilbert.
    No cambia la estructura temporal.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    sd = np.std(x)

    if sd == 0:
        return x

    return x / sd


def make_systems(N=10000):
    """
    Diccionario de sistemas deterministas independientes.
    """
    systems = {}

    systems["logistic_r4_a"] = logistic_map(r=4.0, x0=0.123456, n_iter=N)
    systems["logistic_r4_b"] = logistic_map(r=4.0, x0=0.654321, n_iter=N)
    systems["logistic_r399"] = logistic_map(r=3.99, x0=0.314159, n_iter=N)

    systems["sine_a099"] = sine_map(a=0.99, x0=0.271828, n_iter=N)
    systems["chebyshev_k3"] = chebyshev_map(k=3, x0=0.13579, n_iter=N)
    systems["chebyshev_k4"] = chebyshev_map(k=4, x0=0.24680, n_iter=N)

    hx, hy = henon_map(a=1.4, b=0.3, x0=0.1, y0=0.1, n_iter=N)
    systems["henon_x"] = hx
    systems["henon_y"] = hy

    ix, iy = ikeda_map(u=0.9, x0=0.1, y0=0.1, n_iter=N)
    systems["ikeda_x"] = ix
    systems["ikeda_y"] = iy

    sq, sp = standard_map(K=1.2, q0=0.1, p0=0.2, n_iter=N)
    systems["standard_q"] = sq
    systems["standard_p"] = sp

    for key in list(systems.keys()):
        systems[key] = normalize_series(systems[key])

    return systems


# ============================================================
# 3. Evaluación de pares independientes
# ============================================================

def evaluar_par(x, y, pair_name, taus, orders=(1, 2),
                nulls=("shuffle", "s"), n_surr=200,
                min_shift=100, rng=None):
    """
    Evalúa un par x,y para varios tau, J_1/J_2 y nulos.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []

    for tau in taus:
        f1, f2 = obtener_delta_fases(x, y, tau=tau)

        for order in orders:
            if order == 2:
                J_obs = ml.indice_S_eff_fast(x,y, tau=tau)
            else:
                J_obs = J_from_fases(f1, f2, order=order)

            for null in nulls:

                J_null = null_distribution(
                    f1, f2,x,y,tau,
                    order=order,
                    null=null,
                    n_surr=n_surr,
                    min_shift=min_shift,
                    rng=rng
                )

                rows.append({
                    "pair": pair_name,
                    "tau": tau,
                    "order": order,
                    "null": null,
                    "J_obs": J_obs,
                    "J_null_mean": np.mean(J_null),
                    "J_null_std": np.std(J_null, ddof=1),
                    "D": np.mean(J_null) - J_obs,
                    "Z": zscore_dependencia(J_obs, J_null),
                    "p_left": pvalue_left(J_obs, J_null)
                })

    return pd.DataFrame(rows)


def evaluar_pares_independientes(systems, pairs, taus,
                                  n_surr=200, min_shift=100,
                                  rng=None):
    if rng is None:
        rng = np.random.default_rng()

    dfs = []

    for name_x, name_y in pairs:
        print(f"Evaluando {name_x} vs {name_y}...")

        x = systems[name_x]
        y = systems[name_y]

        L = min(len(x), len(y))
        x = x[:L]
        y = y[:L]

        pair_name = f"{name_x}__vs__{name_y}"

        df_pair = evaluar_par(
            x, y,
            pair_name=pair_name,
            taus=taus,
            orders=(1, 2),
            nulls=("shuffle", "s"),
            n_surr=n_surr,
            min_shift=min_shift,
            rng=rng
        )

        dfs.append(df_pair)

    return pd.concat(dfs, ignore_index=True)


def resumen_falsos_positivos(df, z_thr=5):
    """
    Resume posibles falsos positivos en pares independientes.

    Para pares independientes, idealmente Z no debería superar z_thr.
    """
    rows = []

    grouped = df.groupby(["pair", "order", "null"])

    for (pair, order, null), g in grouped:
        idx = g["Z"].idxmax()
        row_max = g.loc[idx]

        rows.append({
            "pair": pair,
            "order": order,
            "null": null,
            "max_Z": row_max["Z"],
            "tau_at_max_Z": row_max["tau"],
            "min_p_left": g["p_left"].min(),
            "frac_Z_gt_thr": np.mean(g["Z"] > z_thr),
            "mean_Z": g["Z"].mean(),
            "median_Z": g["Z"].median()
        })

    return pd.DataFrame(rows).sort_values(
        ["order", "null", "max_Z"],
        ascending=[True, True, False]
    )


# ============================================================
# 4. Controles positivos opcionales
# ============================================================

def evaluar_controles_positivos(N=10000, taus=np.arange(-10, 11),
                                n_surr=200, min_shift=100, rng=None):
    """
    Controles positivos:
    1. Identidad instantánea x=y.
    2. Copia retardada x_i=z_{i+1}, y_i=z_i.
    3. Hénon x,y acoplados.
    """
    if rng is None:
        rng = np.random.default_rng()

    dfs = []

    # Identidad
    z = logistic_map(r=4.0, x0=0.123456, n_iter=N)
    x = normalize_series(z)
    y = normalize_series(z.copy())

    dfs.append(evaluar_par(
        x, y,
        pair_name="control_identidad_x_eq_y",
        taus=taus,
        n_surr=n_surr,
        min_shift=min_shift,
        rng=rng
    ))

    # Copia retardada
    z = logistic_map(r=4.0, x0=0.654321, n_iter=N + 1)
    x = normalize_series(z[1:])
    y = normalize_series(z[:-1])

    dfs.append(evaluar_par(
        x, y,
        pair_name="control_copia_retardada_tau1",
        taus=taus,
        n_surr=n_surr,
        min_shift=min_shift,
        rng=rng
    ))

    # Hénon x,y
    hx, hy = henon_map(a=1.4, b=0.3, x0=0.1, y0=0.1, n_iter=N)
    hx = normalize_series(hx)
    hy = normalize_series(hy)

    dfs.append(evaluar_par(
        hx, hy,
        pair_name="control_henon_xy",
        taus=taus,
        n_surr=n_surr,
        min_shift=min_shift,
        rng=rng
    ))

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# 5. Gráficas
# ============================================================

def plot_pair(df, pair_name, order=1):
    """
    Grafica J_obs y nulos para un par y un order.
    """
    g = df[(df["pair"] == pair_name) & (df["order"] == order)]

    if g.empty:
        print(f"No hay datos para {pair_name}, order={order}.")
        return

    plt.figure(figsize=(8, 4))

    for null in ["shuffle", "s"]:
        h = g[g["null"] == null].sort_values("tau")

        plt.plot(
            h["tau"],
            h["J_null_mean"],
            marker="o",
            ms=3,
            label=fr"$\langle J_{order}, {null}\rangle$"
        )

        plt.fill_between(
            h["tau"],
            h["J_null_mean"] - 2*h["J_null_std"],
            h["J_null_mean"] + 2*h["J_null_std"],
            alpha=0.15
        )

    h0 = g[g["null"] == "shuffle"].sort_values("tau")
    plt.plot(
        h0["tau"],
        h0["J_obs"],
        marker="o",
        ms=3,
        color="k",
        label=fr"$J_{order}$ observado"
    )

    plt.axhline(0, lw=1, color="k")
    plt.xlabel(r"$\tau$")
    plt.ylabel(fr"$J_{order}$")
    plt.title(f"{pair_name} | J_{order}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pair_z(df, pair_name, order=1):
    """
    Grafica Z contra tau para shuffle y s.
    """
    g = df[(df["pair"] == pair_name) & (df["order"] == order)]

    if g.empty:
        print(f"No hay datos para {pair_name}, order={order}.")
        return

    plt.figure(figsize=(8, 4))

    for null in ["shuffle", "s"]:
        h = g[g["null"] == null].sort_values("tau")
        plt.plot(
            h["tau"],
            h["Z"],
            marker="o",
            ms=3,
            label=fr"$Z[J_{order}],\, {null}$"
        )

    plt.axhline(0, lw=1, color="k")
    plt.axhline(5, lw=1, color="k", linestyle="--")
    plt.xlabel(r"$\tau$")
    plt.ylabel("Z-score")
    plt.title(f"{pair_name} | Z de J_{order}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Ejecución principal
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(123)

    N = 10000
    taus = np.arange(-10, 11)

    # Para prueba rápida usa n_surr=50.
    # Para resultados definitivos usa n_surr=300, 500 o más.
    n_surr = 200

    systems = make_systems(N=N)

    # Pares deterministas independientes.
    # Evita incluir henon_x vs henon_y aquí, porque están acoplados.
    pairs_independientes = [
        ("logistic_r4_a", "logistic_r4_b"),
        ("logistic_r4_a", "logistic_r399"),
        ("logistic_r4_a", "sine_a099"),
        ("logistic_r4_a", "chebyshev_k3"),
        ("logistic_r399", "chebyshev_k4"),
        ("henon_x", "logistic_r4_a"),
        ("henon_x", "sine_a099"),
        ("henon_x", "chebyshev_k3"),
        ("ikeda_x", "logistic_r4_a"),
        ("ikeda_x", "henon_x"),
        ("standard_q", "logistic_r4_a"),
        ("standard_q", "henon_x"),
        ("standard_q", "ikeda_x"),
    ]

    df_ind = evaluar_pares_independientes(
        systems,
        pairs=pairs_independientes,
        taus=taus,
        n_surr=n_surr,
        min_shift=200,
        rng=rng
    )

    df_ind.to_csv("resultados_independientes_J1_J2_shuffle_s.csv", index=False)

    resumen_ind = resumen_falsos_positivos(df_ind, z_thr=5)
    resumen_ind.to_csv("resumen_falsos_positivos_independientes.csv", index=False)

    print("\nResumen de posibles falsos positivos:")
    print(resumen_ind)

    # Controles positivos
    df_ctrl = evaluar_controles_positivos(
        N=N,
        taus=taus,
        n_surr=n_surr,
        min_shift=200,
        rng=rng
    )

    df_ctrl.to_csv("resultados_controles_positivos_J1_J2_shuffle_s.csv", index=False)

    resumen_ctrl = resumen_falsos_positivos(df_ctrl, z_thr=5)
    resumen_ctrl.to_csv("resumen_controles_positivos.csv", index=False)

    print("\nResumen controles positivos:")
    print(resumen_ctrl)

    # Gráficas de algunos casos importantes
    ejemplos = [
        "henon_x__vs__logistic_r4_a",
        "ikeda_x__vs__henon_x",
        "standard_q__vs__logistic_r4_a",
    ]

    for pair_name in ejemplos:
        for order in [1, 2]:
            plot_pair(df_ind, pair_name, order=order)
            plot_pair_z(df_ind, pair_name, order=order)

    controles = [
        "control_identidad_x_eq_y",
        "control_copia_retardada_tau1",
        "control_henon_xy",
    ]

    for pair_name in controles:
        for order in [1, 2]:
            plot_pair(df_ctrl, pair_name, order=order)
            plot_pair_z(df_ctrl, pair_name, order=order)