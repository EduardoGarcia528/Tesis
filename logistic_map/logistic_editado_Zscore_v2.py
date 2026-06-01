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
# CARGA DE ARCHIVOS
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
    Si no existe, intenta tau sin cero inicial: 1, 2, ..., 10.
    """
    carpeta_Z = Path(carpeta_Z)
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


def cargar_diccionario_Z(carpeta_Z, clave, tau):
    """
    Carga el diccionario guardado en .npy.

    Campos usados en esta figura:
        Z['r_barrido']
        Z['Z']
        Z['S_obs']
        Z['mu_null']
        Z['sigma_null']
    """
    ruta = ruta_archivo_Z(carpeta_Z, clave, tau)
    Z = np.load(ruta, allow_pickle=True).item()
    Z["ruta"] = ruta
    return Z


def extraer_array(Z, clave, nombre_archivo=None):
    if clave not in Z:
        origen = f" en {nombre_archivo}" if nombre_archivo is not None else ""
        raise KeyError(f"No existe la clave Z['{clave}']{origen}.")
    return np.asarray(Z[clave])


def cargar_curva_Zscore(carpeta_Z, clave, tau):
    Z = cargar_diccionario_Z(carpeta_Z, clave, tau)
    ruta = Z["ruta"]
    return {
        "r": extraer_array(Z, "r_barrido", ruta),
        "Z": extraer_array(Z, "Z", ruta),
        "ruta": ruta,
    }


def cargar_curvas_fases_shuffle_direct(carpeta_Z, taus=range(1, 6)):
    return {
        tau: cargar_curva_Zscore(carpeta_Z, "fases_shuffle_direct", tau)
        for tau in taus
    }


def cargar_curva_S_obs_vs_null(carpeta_Z, clave, tau=1):
    Z = cargar_diccionario_Z(carpeta_Z, clave, tau)
    ruta = Z["ruta"]
    return {
        "r": extraer_array(Z, "r_barrido", ruta),
        "S_obs": extraer_array(Z, "S_obs", ruta),
        "mu_null": extraer_array(Z, "mu_null", ruta),
        "sigma_null": extraer_array(Z, "sigma_null", ruta),
        "ruta": ruta,
    }


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


def graficar_S_obs_vs_null(
    ax,
    curva,
    etiqueta_obs,
    etiqueta_null,
    color_obs="black",
    color_null="#1f77b4"
):
    r = curva["r"]
    S_obs = curva["S_obs"]
    mu = curva["mu_null"]
    sigma = curva["sigma_null"]

    ax.plot(
        r,
        S_obs,
        color=color_obs,
        lw=1.15,
        label=etiqueta_obs
    )
    ax.plot(
        r,
        mu,
        color=color_null,
        lw=1.15,
        ls="--",
        label=etiqueta_null
    )
    ax.fill_between(
        r,
        mu - sigma,
        mu + sigma,
        color=color_null,
        alpha=0.22,
        linewidth=0,
        label=fr"{etiqueta_null} $\pm\sigma$"
    )


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
    taus_Z_fases_direct=range(1, 6),
    guardar_como="Zscore_logistico_S_eff_multipanel_v2.png"
):
    carpeta_Z = Path(carpeta_Z)
    taus_Z_fases_direct = list(taus_Z_fases_direct)

    # ---------- estilos ----------
    colores_tau = {
        tau: plt.cm.tab10((tau - 1) % 10)
        for tau in taus_Z_fases_direct
    }

    estilos_tau = {
        1: "-",
        2: "--",
        3: "-.",
        4: ":",
        5: (0, (3, 1, 1, 1)),
    }

    # ---------- Bifurcación ----------
    _, r_bif, x_bif = generar_bifurcacion(
        r_min=r_min,
        r_max=r_max,
        resolucion_r=resolucion_r,
        longitud_serie=longitud_bif,
        iter_descartar=iter_descartar_bif,
        x0=x0
    )

    # ---------- Datos cargados ----------
    Z_fases_direct = cargar_curvas_fases_shuffle_direct(
        carpeta_Z=carpeta_Z,
        taus=taus_Z_fases_direct
    )

    S_vel_direct_tau1 = cargar_curva_S_obs_vs_null(
        carpeta_Z=carpeta_Z,
        clave="velocidades_shuffle_direct",
        tau=5
    )

    S_fases_phase_tau1 = cargar_curva_S_obs_vs_null(
        carpeta_Z=carpeta_Z,
        clave="fases_shuffle_phase",
        tau=5
    )

    S_vel_phase_tau1 = cargar_curva_S_obs_vs_null(
        carpeta_Z=carpeta_Z,
        clave="velocidades_shuffle_phase",
        tau=5
    )

    # ---------- Figura ----------
    fig, axes = plt.subplots(
        5, 1,
        figsize=(7.6, 11.8),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    # a) Bifurcación
    ax = axes[0]
    ax.plot(r_bif, x_bif, ",", color="black", alpha=0.55)
    aplicar_formato_panel(ax, ylabel=r"$x$", show_bottom_ticks=False)
    poner_letra_panel(ax, "a")

    # b) Z-score de fases shuffle direct, tau = 1,...,5
    ax = axes[1]
    ax.invert_yaxis()
    for tau in taus_Z_fases_direct:
        curva = Z_fases_direct[tau]
        ax.plot(
            curva["r"],
            curva["Z"],
            color=colores_tau[tau],
            ls=estilos_tau.get(tau, "-"),
            lw=1.05,
            label=fr"$\tau={tau}$"
        )

    poner_linea_cero(ax)
    aplicar_formato_panel(
        ax,
        ylabel=r"$Z(S_{\theta}^{(\Pi x)})$",
        show_bottom_ticks=False
    )
    poner_letra_panel(ax, "b")

    handles_tau = [
        Line2D(
            [0], [0],
            color=colores_tau[tau],
            lw=1.2,
            ls=estilos_tau.get(tau, "-"),
            label=fr"$\tau={tau}$"
        )
        for tau in taus_Z_fases_direct
    ]
    ax.legend(
        handles=handles_tau,
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        ncol=5,
        handlelength=2.3,
        columnspacing=0.9
    )

    # c) Velocidades shuffle direct, tau = 1
    ax = axes[2]
    ax.set_ylim(0.0, 1.0)
    graficar_S_obs_vs_null(
        ax,
        S_vel_direct_tau1,
        etiqueta_obs=r"$S_{\Delta\theta}$",
        etiqueta_null=r"$\langle S_{\Delta\theta}^{(\Pi x)}\rangle$",
        color_obs="black",
        color_null="#1f77b4"
    )
    aplicar_formato_panel(ax, ylabel=r"$S$", show_bottom_ticks=False)
    poner_letra_panel(ax, "c")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2, ncol=1)

    # d) Fases shuffle phase, tau = 1
    ax = axes[3]
    ax.set_ylim(0.0, 1.0)
    graficar_S_obs_vs_null(
        ax,
        S_fases_phase_tau1,
        etiqueta_obs=r"$S_{\theta}$",
        etiqueta_null=r"$\langle S_{\theta}^{(\Pi\theta)}\rangle$",
        color_obs="black",
        color_null="#2ca02c"
    )
    aplicar_formato_panel(ax, ylabel=r"$S$", show_bottom_ticks=False)
    poner_letra_panel(ax, "d")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2, ncol=1)

    # e) Velocidades shuffle phase, tau = 1
    ax = axes[4]
    ax.set_ylim(0.0, 1.0)
    graficar_S_obs_vs_null(
        ax,
        S_vel_phase_tau1,
        etiqueta_obs=r"$S_{\Delta\theta}$",
        etiqueta_null=r"$\langle S_{\Delta\theta}^{(\Pi\Delta\theta)}\rangle$",
        color_obs="black",
        color_null="#d62728"
    )
    aplicar_formato_panel(ax, ylabel=r"$S$", show_bottom_ticks=True, xlabel=r"$r$")
    poner_letra_panel(ax, "e")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2, ncol=1)

    fig.subplots_adjust(top=0.98, left=0.13, right=0.98, bottom=0.065, hspace=0.04)
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
        taus_Z_fases_direct=range(1, 6),
        guardar_como="Zscore_logistico_S_eff_multipanel_v2.png"
    )
