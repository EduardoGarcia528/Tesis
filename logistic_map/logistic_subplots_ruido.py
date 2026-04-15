import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# MAPA
# =========================
def logistic_map(r, x):
    return r * x * (1 - x)


# =========================
# GENERAR BIFURCACIÓN SIN RUIDO
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
            x = np.clip(x, 0, 1)
            x = logistic_map(r, x)

        serie = np.empty(longitud_serie)
        for i in range(longitud_serie):
            x = np.clip(x, 0, 1)
            x = logistic_map(r, x)
            serie[i] = x

        r_vals_plot.extend([r] * longitud_serie)
        x_vals_plot.extend(serie)

    return r_grid, np.array(r_vals_plot), np.array(x_vals_plot)


# =========================
# CARGA DE MEDIDAS CON RUIDO
# =========================
def cargar_medidas_ruido(ruidos):
    data = {
        "J": {},
        "C6a": {},
        "SE": {},
        "PE": {},
        "C6": {},
        "PE_alpha": {}
    }

    for var_ruido in ruidos:
        data["J"][var_ruido] = np.load(f"data/RUIDOS/aditivo_J/J_por_ruido_{str(var_ruido)}.npy")    
        data["C6"][var_ruido] = 1-np.load(f"data/RUIDOS/aditivo_C6/C_por_ruido_{str(var_ruido)}.npy")
        data["C6a"][var_ruido] = 1-np.load(f"data/RUIDOS/aditivo_C6_alpha/C6_por_ruido_{str(var_ruido)}.npy")
        data["SE"][var_ruido] = np.load(f"data/RUIDOS/aditivo_shannon_alpha/S_por_ruido_{str(var_ruido)}.npy")
        data["PE"][var_ruido] = np.load(f"data/RUIDOS/aditivo_PE/PE_por_ruido_{str(var_ruido)}.npy")
        data["PE_alpha"][var_ruido] = np.load(f"data/RUIDOS/aditivo_PE_alpha/PE_por_ruido_{str(var_ruido)}.npy")
    return data


# =========================
# FORMATO PAPER
# =========================
def aplicar_formato_panel(ax, ylabel=None, show_bottom_ticks=False,xlabel=None):
    ax.set_xlim(3.0, 4.0)
    # ax.set_ylim(0, 1)

    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=360,labelpad=12,fontsize=14)
    if xlabel is not None:
        ax.set_xlabel(xlabel, rotation=360,fontsize=10)

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
        fontsize=15
    )


# =========================
# FIGURA PRINCIPAL
# =========================
def figura_multipanel_ruido_aditivo(
    r_min=3.0,
    r_max=4.0,
    resolucion_r=300,
    longitud_bif=1200,
    iter_descartar_bif=2000
):
    ruidos = [1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    colores = {
        1e-10: "gray",
        1e-8: "#2ECC71",
        1e-7: "purple",
        1e-6: "#CBAC00B8",
        1e-5: "#1ABC9C",
        1e-4: "#E91E63",
    }

    estilos = {
        1e-10: "-",
        1e-8: "--",
        1e-7: "-.",
        1e-6: ":",
        1e-5: (0, (5, 1)),
        1e-4: (0, (3, 1, 1, 1)),
    }

    # ---------- Bifurcación ----------
    r_grid, r_bif, x_bif = generar_bifurcacion(
        r_min=r_min,
        r_max=r_max,
        resolucion_r=resolucion_r,
        longitud_serie=longitud_bif,
        iter_descartar=iter_descartar_bif
    )

    # ---------- Medidas con ruido ----------
    data_ruido = cargar_medidas_ruido(ruidos)

    # ---------- Figura ----------
    fig, axes = plt.subplots(
        7, 1,
        figsize=(7.2, 13.2),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    # Panel a) bifurcación
    ax = axes[0]
    ax.plot(r_bif, x_bif, ",", color="black", alpha=0.55)
    aplicar_formato_panel(ax, ylabel=r"$x$", show_bottom_ticks=False)
    poner_letra_panel(ax, "a")

    # Panel b) J
    ax = axes[1]
    for vr in ruidos:
        ax.plot(r_grid, data_ruido["J"][vr], color=colores[vr], ls=estilos[vr], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$J$", show_bottom_ticks=False)
    poner_letra_panel(ax, "b")

    # Panel c) C6^alpha
    ax = axes[2]
    for vr in ruidos:
        ax.plot(r_grid, data_ruido["C6a"][vr], color=colores[vr], ls=estilos[vr], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$C_6^\alpha$", show_bottom_ticks=False)
    poner_letra_panel(ax, "c")

    # Panel d) C6
    ax = axes[3]
    for vr in ruidos:
        ax.plot(r_grid, data_ruido["C6"][vr], color=colores[vr], ls=estilos[vr], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$C_6$", show_bottom_ticks=True)
    poner_letra_panel(ax, "d")
    ax.set_xlabel(r"$r$")

    # Panel e) Shannon
    ax = axes[4]
    for vr in ruidos:
        ax.plot(r_grid, data_ruido["SE"][vr], color=colores[vr], ls=estilos[vr], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$SE^\alpha$", show_bottom_ticks=False)
    poner_letra_panel(ax, "e")

    # Panel f) PE
    ax = axes[5]
    for vr in ruidos:
        ax.plot(r_grid, data_ruido["PE"][vr], color=colores[vr], ls=estilos[vr], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$PE$", show_bottom_ticks=False)
    poner_letra_panel(ax, "f")

    # Panel f) PE
    ax = axes[6]
    for vr in ruidos: 
        ax.plot(r_grid, data_ruido["PE_alpha"][vr], color=colores[vr], ls=estilos[vr], lw=1.1)
    aplicar_formato_panel(ax, ylabel=r"$PE^{\alpha}$", show_bottom_ticks=True, xlabel=r"$r$")
    poner_letra_panel(ax, "g")
    # ---------- Leyenda global ----------
    handles = []
    for vr in ruidos:
        handles.append(
            Line2D(
                [0], [0],
                color=colores[vr],
                lw=1.2,
                ls=estilos[vr],
                label=fr"$\sigma^2 = {vr}$"
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10
    )

    fig.subplots_adjust(top=0.94, left=0.12, right=0.98, bottom=0.07, hspace=0.04)
    plt.savefig('ruido_aditivo.png')
    plt.show()


# =========================
# EJECUCIÓN
# =========================
figura_multipanel_ruido_aditivo(
    r_min=3.0,
    r_max=4.0,
    resolucion_r=300,
    longitud_bif=4000,
    iter_descartar_bif=1000
)