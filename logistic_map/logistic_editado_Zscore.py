import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


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
# CARGA DE Z-SCORES
# =========================
ARCHIVOS_Z = {
    "fases_shuffle_direct": "Z_fases_shuffle_direct_tau_{tau}.npy",
    "fases_shuffle_phase": "Z_fases_shuffle_phase_tau_{tau}.npy",
    "velocidades_shuffle_direct": "Z_velocidades_shuffle_direct_tau_{tau}.npy",
    "velocidades_shuffle_phase": "Z_velocidades_shuffle_phase_tau_{tau}.npy",
}


def ruta_archivo_Z(carpeta_Z, clave, tau):
    """
    Busca primero tau con dos dígitos: 01, 02, ..., 10.
    Si no existe, intenta con tau sin cero inicial: 1, 2, ..., 10.
    """
    tau_2d = f"{tau:02d}"
    tau_simple = str(tau)

    ruta = carpeta_Z / ARCHIVOS_Z[clave].format(tau=tau_2d)
    if ruta.exists():
        return ruta

    ruta_alt = carpeta_Z / ARCHIVOS_Z[clave].format(tau=tau_simple)
    if ruta_alt.exists():
        return ruta_alt

    raise FileNotFoundError(
        f"No encontré el archivo para clave='{clave}', tau={tau}. "
        f"Probé:\n  {ruta}\n  {ruta_alt}"
    )


def cargar_Zscore(carpeta_Z, clave, tau):
    """
    Carga un archivo .npy con estructura de diccionario.

    Se espera:
        Z['r_barrido']
        Z['Z']
    """
    ruta = ruta_archivo_Z(carpeta_Z, clave, tau)
    Z = np.load(ruta, allow_pickle=True).item()

    return {
        "r": np.asarray(Z["r_barrido"]),
        "Z": np.asarray(Z["Z"]),
        "ruta": ruta,
    }


def cargar_todas_las_curvas_Z(carpeta_Z, taus=range(1, 11)):
    datos = {clave: {} for clave in ARCHIVOS_Z}

    for clave in ARCHIVOS_Z:
        for tau in taus:
            datos[clave][tau] = cargar_Zscore(carpeta_Z, clave, tau)

    return datos


# =========================
# FORMATO PAPER
# =========================
def aplicar_formato_panel(ax, ylabel=None, show_bottom_ticks=False, xlabel=None):
    ax.set_xlim(3.45, 4.0)

    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=360, labelpad=18, fontsize=13)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)

    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.6)

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


def poner_linea_cero(ax):
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.45, zorder=0)


# =========================
# FIGURA PRINCIPAL
# =========================
def figura_multipanel_Zscore(
    r_min=3.0,
    r_max=4.0,
    resolucion_r=300,
    longitud_bif=4000,
    iter_descartar_bif=1000,
    x0=0.6,
    carpeta_Z=Path("data") / "resultados_logistico_S_eff_Zscore",
    taus=range(1, 11),
    guardar_como="Zscore_logistico_S_eff_multipanel.png"
):
    carpeta_Z = Path(carpeta_Z)
    taus = list(taus)

    # ---------- estilos ----------
    colores_tau = {
        tau: plt.cm.tab10((tau - 1) % 10)
        for tau in taus
    }

    estilos_tau = {
        1: "-",
        2: "--",
        3: "-.",
        4: ":",
        5: (0, (3, 1, 1, 1)),
        6: (0, (5, 1)),
        7: (0, (1, 1)),
        8: (0, (5, 2, 1, 2)),
        9: (0, (3, 2, 3, 2, 1, 2)),
        10: (0, (7, 2)),
    }

    curvas_tau1 = [
        (
            "fases_shuffle_direct",
            r"$S_{\theta}^{(\Pi x)}$",
            "black",
            "-"
        ),
        (
            "velocidades_shuffle_direct",
            r"$S_{\Delta\theta}^{(\Pi x)}$",
            "gray",
            "--"
        ),
        (
            "fases_shuffle_phase",
            r"$S_{\theta}^{(\Pi\theta)}$",
            "#1f77b4",
            "-."
        ),
        (
            "velocidades_shuffle_phase",
            r"$S_{\Delta\theta}^{(\Pi\Delta\theta)}$",
            "#d62728",
            ":"
        ),
    ]

    paneles_tau = [
        (
            "fases_shuffle_direct",
            r"$Z\!\left(S_{\theta}^{(\Pi x)}\right)$",
            "b"
        ),
        (
            "velocidades_shuffle_direct",
            r"$Z\!\left(S_{\Delta\theta}^{(\Pi x)}\right)$",
            "c"
        ),
        (
            "fases_shuffle_phase",
            r"$Z\!\left(S_{\theta}^{(\Pi\theta)}\right)$",
            "d"
        ),
        (
            "velocidades_shuffle_phase",
            r"$Z\!\left(S_{\Delta\theta}^{(\Pi\Delta\theta)}\right)$",
            "e"
        ),
    ]

    # ---------- Bifurcación ----------
    _, r_bif, x_bif = generar_bifurcacion(
        r_min=r_min,
        r_max=r_max,
        resolucion_r=resolucion_r,
        longitud_serie=longitud_bif,
        iter_descartar=iter_descartar_bif,
        x0=x0
    )

    # ---------- Z-score ----------
    datos_Z = cargar_todas_las_curvas_Z(carpeta_Z=carpeta_Z, taus=taus)

    # ---------- Figura ----------
    fig, axes = plt.subplots(
        5, 1,
        figsize=(7.6, 13.8),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    # a) bifurcación
    ax = axes[0]
    ax.plot(r_bif, x_bif, ",", color="black", alpha=0.55)
    aplicar_formato_panel(ax, ylabel=r"$x$", show_bottom_ticks=False)
    poner_letra_panel(ax, "a")

    # b) Comparación de los cuatro Z-score para tau = 1
    # ax = axes[1]
    # ax.invert_yaxis()
    # for clave, etiqueta, color, ls in curvas_tau1:
    #     curva = datos_Z[clave][1]
    #     ax.plot(
    #         curva["r"],
    #         curva["Z"],
    #         color=color,
    #         ls=ls,
    #         lw=1.15,
    #         label=etiqueta
    #     )

    # poner_linea_cero(ax)
    # aplicar_formato_panel(ax, ylabel=r"$Z$", show_bottom_ticks=False)
    # poner_letra_panel(ax, "b")
    # ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)

    # c-f) Barridos en tau para cada tipo de Z-score
    for ax, (clave, ylabel, letra) in zip(axes[1:], paneles_tau):
        ax.invert_yaxis()
        for tau in taus:
            curva = datos_Z[clave][tau]
            ax.plot(
                curva["r"],
                curva["Z"],
                color=colores_tau[tau],
                ls=estilos_tau.get(tau, "-"),
                lw=1.0,
                label=fr"$\tau={tau}$"
            )

        poner_linea_cero(ax)
        aplicar_formato_panel(
            ax,
            ylabel=ylabel,
            show_bottom_ticks=(letra == "f"),
            xlabel=r"$r$" if letra == "f" else None
        )
        poner_letra_panel(ax, letra)

    # Una sola leyenda para los paneles con barrido en tau
    handles_tau = [
        Line2D(
            [0], [0],
            color=colores_tau[tau],
            lw=1.2,
            ls=estilos_tau.get(tau, "-"),
            label=fr"$\tau={tau}$"
        )
        for tau in taus
    ]

    for ax in axes[1:]:
        ax.legend(
            handles=handles_tau,
            loc="lower right",
            frameon=False,
            fontsize=7.3,
            ncol=5,
            handlelength=2.4,
            columnspacing=0.8
        )

    fig.subplots_adjust(top=0.98, left=0.13, right=0.98, bottom=0.06, hspace=0.04)
    plt.savefig(guardar_como, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":
    figura_multipanel_Zscore(
        r_min=3.45,
        r_max=4.0,
        resolucion_r=300,
        longitud_bif=1000,
        iter_descartar_bif=1000,
        x0=0.6,
        carpeta_Z=Path("data") / "resultados_logistico_S_eff_Zscore",
        taus=range(1, 6),
        guardar_como="Zscore_logistico_S_eff_multipanel.png"
    )
