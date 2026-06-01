import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mi_libreria as ml

# =========================
# MAPA LOGÍSTICO
# =========================
def logistic_map(r, x):
    return r * x * (1.0 - x)


# =========================
# GENERAR BIFURCACIÓN
# =========================
def generar_bifurcacion(
    r_min=3.0,
    r_max=4.0,
    resolucion_r=300,
    longitud_serie=1200,
    iter_descartar=2000,
    x0=0.6
):
    r_grid = np.linspace(r_min, r_max, resolucion_r)[1:]

    r_vals_plot = []
    x_vals_plot = []

    for r in r_grid:
        x = x0

        for _ in range(iter_descartar):
            x = np.clip(x, 0.0, 1.0)
            x = logistic_map(r, x)

        serie = np.empty(longitud_serie)
        for i in range(longitud_serie):
            x = np.clip(x, 0.0, 1.0)
            x = logistic_map(r, x)
            serie[i] = x

        r_vals_plot.extend([r] * longitud_serie)
        x_vals_plot.extend(serie)

    return r_grid, np.array(r_vals_plot), np.array(x_vals_plot)


# =========================
# GENERAR SERIE PARA UN r
# =========================
def generar_serie_logistica(r, longitud_serie=1200, iter_descartar=2000, x0=0.6):
    x = x0

    for _ in range(iter_descartar):
        x = np.clip(x, 0.0, 1.0)
        x = logistic_map(r, x)

    serie = np.empty(longitud_serie)
    for i in range(longitud_serie):
        x = np.clip(x, 0.0, 1.0)
        x = logistic_map(r, x)
        serie[i] = x

    return serie


# =========================
# FORMATO PAPER
# =========================
def aplicar_formato_panel(ax, ylabel=None, show_bottom_ticks=False, xlabel=None):
    ax.set_xlim(3.0, 4.0)

    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=360, labelpad=14, fontsize=13)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8)
    ax.tick_params(axis='both', which='minor', direction='out', length=2.5, width=0.6)

    if not show_bottom_ticks:
        ax.tick_params(labelbottom=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def poner_letra_panel(ax, letra):
    ax.text(
        0.985, 0.50, f"{letra})",
        transform=ax.transAxes,
        ha="right", va="center",
        fontsize=14
    )


# =========================
# CÁLCULO DE CURVAS
# =========================
def calcular_curvas_indices(
    r_grid,
    longitud_serie=1200,
    iter_descartar=2000,
    x0=0.6
):
    ms = [3, 4, 5, 6]
    ds = [2, 3, 4, 5, 6]
    mus = [2, 3, 4, 5, 6]

    # Diccionarios para guardar curvas
    pe_m = {m: np.empty(len(r_grid)) for m in ms}
    pe_alpha_m = {m: np.empty(len(r_grid)) for m in ms}

    cd = {d: np.empty(len(r_grid)) for d in ds}
    cd_alpha = {d: np.empty(len(r_grid)) for d in ds}
    c6_mu = {mu: np.empty(len(r_grid)) for mu in mus}

    for i, r in enumerate(r_grid):
        serie = generar_serie_logistica(
            r,
            longitud_serie=longitud_serie,
            iter_descartar=iter_descartar,
            x0=x0
        )

        alpha = ml.angulos_alpha(serie, False, tau=1)

        # -------- PE variando m --------
        for m in ms:
            pe_m[m][i] = ml.permutation_entropy(serie, m=m, tau=1)
            pe_alpha_m[m][i] = ml.permutation_entropy(alpha, m=m, tau=1)

        # -------- C_d variando d --------
        # max_gamma = 6 para asegurar tener C_2,...,C_6
        C, _ = ml.gamma_index_jacobs(serie, max_gamma=6, mu=5.0)

        # Para alpha: versión circular
        C_alpha, _ = ml.gamma_index_jacobs_circular(alpha, max_gamma=6, nu=5.0)

        for d in ds:
            cd[d][i] = C[d]
            cd_alpha[d][i] = C_alpha[d]

        # -------- C_6 variando mu --------
        for mu in mus:
            C_mu, _ = ml.gamma_index_jacobs(serie, max_gamma=6, mu=float(mu))
            c6_mu[mu][i] = C_mu[6]

        print(f"{i + 1}/{len(r_grid)}  r = {r:.6f}")

    return pe_m, pe_alpha_m, cd, cd_alpha, c6_mu


# =========================
# FIGURA PRINCIPAL
# =========================
def figura_multipanel_indices(
    r_min=3.0,
    r_max=4.0,
    resolucion_r=300,
    longitud_bif=4000,
    iter_descartar_bif=1000,
    longitud_indices=1200,
    iter_descartar_indices=2000,
    x0=0.6,
    guardar_como="indices_logistico.png"
):
    # ---------- estilos ----------
    colores_m = {
        3: "gray",
        4: "#2ECC71",
        5: "purple",
        6: "#E91E63",
    }

    colores_d = {
        2: "gray",
        3: "#2ECC71",
        4: "purple",
        5: "#CBAC00B8",
        6: "#E91E63",
    }

    colores_mu = {
        2: "gray",
        3: "#2ECC71",
        4: "purple",
        5: "#CBAC00B8",
        6: "#E91E63",
    }

    estilos = {
        2: "-",
        3: "--",
        4: "-.",
        5: ":",
        6: (0, (3, 1, 1, 1)),
    }

    estilos_4 = {
        1: "-",
        2: "--",
        3: "-.",
        4: ":",
    }

    # ---------- Bifurcación ----------
    r_grid, r_bif, x_bif = generar_bifurcacion(
        r_min=r_min,
        r_max=r_max,
        resolucion_r=resolucion_r,
        longitud_serie=longitud_bif,
        iter_descartar=iter_descartar_bif,
        x0=x0
    )

    # ---------- Índices ----------
    pe_m, pe_alpha_m, cd, cd_alpha, c6_mu = calcular_curvas_indices(
        r_grid=r_grid,
        longitud_serie=longitud_indices,
        iter_descartar=iter_descartar_indices,
        x0=x0
    )

    # ---------- Figura ----------
    fig, axes = plt.subplots(
        6, 1,
        figsize=(7.6, 13.8),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    # a) bifurcación
    ax = axes[0]
    ax.plot(r_bif, x_bif, ",", color="black", alpha=0.55)
    aplicar_formato_panel(ax, ylabel=r"$x$", show_bottom_ticks=False)
    poner_letra_panel(ax, "a")

    # b) PE variando m
    ax = axes[1]
    for m in [3, 4, 5, 6]:
        ax.plot(r_grid, pe_m[m], color=colores_m[m], ls=estilos[m], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$PE$", show_bottom_ticks=False)
    poner_letra_panel(ax, "b")

    # c) PE^alpha variando m
    ax = axes[2]
    for m in [3, 4, 5, 6]:
        ax.plot(r_grid, pe_alpha_m[m], color=colores_m[m], ls=estilos[m], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$PE^\alpha$", show_bottom_ticks=False)
    poner_letra_panel(ax, "c")

    # e) C_d variando d
    ax = axes[3]
    for d in [2, 3, 4, 5, 6]:
        ax.plot(r_grid, 1- cd[d], color=colores_d[d], ls=estilos[d], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$C_d$", show_bottom_ticks=False)
    poner_letra_panel(ax, "d")

    # f) C_d^alpha variando d
    ax = axes[4]
    for d in [2, 3, 4, 5, 6]:
        ax.plot(r_grid, 1-cd_alpha[d], color=colores_d[d], ls=estilos[d], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$C_d^\alpha$", show_bottom_ticks=False)
    poner_letra_panel(ax, "e")

    # g) C6 variando mu
    ax = axes[5]
    for mu in [2, 3, 4, 5, 6]:
        ax.plot(r_grid, 1-c6_mu[mu], color=colores_mu[mu], ls=estilos[mu], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$C_6$", show_bottom_ticks=True, xlabel=r"$r$")
    poner_letra_panel(ax, "f")

    # ---------- leyendas por panel ----------
    handles_m = [
        Line2D([0], [0], color=colores_m[m], lw=1.2, ls=estilos[m], label=fr"$m={m}$")
        for m in [3, 4, 5, 6]
    ]
    handles_d = [
        Line2D([0], [0], color=colores_d[d], lw=1.2, ls=estilos[d], label=fr"$d={d}$")
        for d in [2, 3, 4, 5, 6]
    ]
    handles_mu = [
        Line2D([0], [0], color=colores_mu[mu], lw=1.2, ls=estilos[mu], label=fr"$\mu={mu}$")
        for mu in [2, 3, 4, 5, 6]
    ]

    axes[1].legend(handles=handles_m, loc="upper left", frameon=False, fontsize=9, ncol=1)
    axes[2].legend(handles=handles_m, loc="upper left", frameon=False, fontsize=9, ncol=2)
    axes[3].legend(handles=handles_d, loc="upper left", frameon=False, fontsize=9, ncol=2)
    axes[4].legend(handles=handles_d, loc="upper left", frameon=False, fontsize=9, ncol=2)
    axes[5].legend(handles=handles_mu, loc="upper left", frameon=False, fontsize=9, ncol=2)

    fig.subplots_adjust(top=0.98, left=0.12, right=0.98, bottom=0.06, hspace=0.04)
    plt.savefig(guardar_como, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# EJECUCIÓN
# =========================
figura_multipanel_indices(
    r_min=3.0,
    r_max=4.0,
    resolucion_r=300,
    longitud_bif=1000,
    iter_descartar_bif=1000,
    longitud_indices=20_000,
    iter_descartar_indices=1000,
    x0=0.6,
    guardar_como="indices_logistico.png"
)