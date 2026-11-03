# -*- coding: utf-8 -*-

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mi_libreria as ml


# ============================================================
# CONFIG
# ============================================================

CACHE_DIR = "cache_pe_score"

BINS = 100
LOGLOG = True

FONT_GENERAL = 12
FONT_TICKS = 11
TITLE_SIZE = 12
LINE_WIDTH = 1.6

PANEL_W = 5.0
PANEL_H = 3.5

SAVE_FIG = True
SAVE_CSV = True

FIG_NAME = "distribuciones_Zscore_por_rango_BRF.png"
CSV_NAME = "ajustes_BRF_distribuciones_Zscore.csv"


# ============================================================
# UTILIDADES
# ============================================================

def natural_key(text):
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", str(text))
    ]


def list_cache_pkls(cache_dir=CACHE_DIR):
    files = [
        f for f in os.listdir(cache_dir)
        if f.lower().endswith(".pkl")
    ]
    return sorted(files, key=natural_key)


def parse_panel_title_from_filename(fname):
    mD = re.search(r"_D(\d+)", fname)
    mt = re.search(r"_tau(\d+)", fname)

    if mD and mt:
        return f"m={mD.group(1)}, tau={mt.group(1)}"

    return os.path.splitext(fname)[0]


def parse_m_tau_from_filename(fname):
    mD = re.search(r"_D(\d+)", fname)
    mt = re.search(r"_tau(\d+)", fname)

    m = int(mD.group(1)) if mD else np.nan
    tau = int(mt.group(1)) if mt else np.nan

    return m, tau


def load_z_values(pkl_path):
    """
    Mantiene la misma convención de dist_global.py:
        z_plot = -z
    """
    df = pd.read_pickle(pkl_path)

    if "z" not in df.columns:
        raise ValueError(
            f"El archivo no contiene columna 'z': {pkl_path}"
        )

    z = df["z"].to_numpy(dtype=float)
    z = z[np.isfinite(z)]

    return -z


def choose_grid(n_panels):
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


# ============================================================
# DISTRIBUCIÓN EMPÍRICA ORDENADA POR RANGO
# ============================================================

def ranked_histogram_distribution(z, bins=100):
    """
    Construye un histograma de Z-score, convierte los conteos
    en probabilidades discretas y las ordena de mayor a menor.

        p_i = n_i / sum_j n_j

        p_(1) >= p_(2) >= ... >= p_(R)

    Solo se conservan bins ocupados, porque fit_rank trabaja
    en espacio logarítmico y no admite p=0.
    """

    counts, edges = np.histogram(z, bins=bins)

    total = counts.sum()

    if total == 0:
        return np.array([]), counts, edges

    probs = counts.astype(float) / total

    probs = probs[probs > 0]

    probs_ranked = np.sort(probs)[::-1]

    return probs_ranked, counts, edges


# ============================================================
# AJUSTE BRF
# ============================================================

def fit_brf_rank_distribution(p_rank):
    """
    Ajusta una Beta Rank Function normalizada:

        p(r) propto (1-u)^b / u^a

    con

        u = r/(N+1)

    usando las funciones incluidas en mi_libreria.
    """

    p_rank = np.asarray(p_rank, dtype=float)
    p_rank = p_rank[np.isfinite(p_rank) & (p_rank > 0)]

    if len(p_rank) < 3:
        raise ValueError(
            "Se necesitan al menos 3 probabilidades positivas."
        )

    # Ajuste BRF normalizado: devuelve a y b
    a_fit, b_fit = ml.fit_rank(
        p_rank,
        usar_BRF=True,
        normalizar=True
    )

    N = len(p_rank)

    # A deja de ser independiente al normalizar.
    p_fit = ml.BRF(
        N,
        A=1.0,
        a=a_fit,
        b=b_fit,
        DGBD=False,
        normalizar=True
    )

    return p_fit, a_fit, b_fit


# ============================================================
# MÉTRICAS
# ============================================================

def fit_metrics(p_obs, p_fit):
    """
    Métricas en espacio logarítmico, consistente con fit_rank().
    """

    log_obs = np.log(p_obs)
    log_fit = np.log(p_fit)

    residuals = log_obs - log_fit

    rmse_log = np.sqrt(np.mean(residuals**2))
    mae_log = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (log_obs - np.mean(log_obs))**2
    )

    if ss_tot > 0:
        r2_log = 1.0 - ss_res / ss_tot
    else:
        r2_log = np.nan

    return rmse_log, mae_log, r2_log


# ============================================================
# MAIN
# ============================================================

pkls = list_cache_pkls(CACHE_DIR)

if len(pkls) == 0:
    raise FileNotFoundError(
        f"No encontré archivos .pkl en: {CACHE_DIR}"
    )

n = len(pkls)
nrows, ncols = choose_grid(n)

fig_w = PANEL_W * ncols
fig_h = PANEL_H * nrows

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(fig_w, fig_h),
    sharex=False,
    sharey=False
)

axs = np.atleast_1d(axs).ravel()

summary = []


for ax, fname in zip(axs, pkls):

    pkl_path = os.path.join(CACHE_DIR, fname)

    z = load_z_values(pkl_path)

    title = parse_panel_title_from_filename(fname)
    m, tau = parse_m_tau_from_filename(fname)

    ax.set_title(title, fontsize=TITLE_SIZE)

    if z.size == 0:
        ax.text(
            0.5,
            0.5,
            "Sin Z finitos",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_GENERAL
        )
        ax.grid(alpha=0.3)
        continue

    # --------------------------------------------------------
    # Histograma -> distribución -> ranking
    # --------------------------------------------------------

    p_rank, counts, edges = ranked_histogram_distribution(
        z,
        bins=BINS
    )

    if len(p_rank) < 3:
        ax.text(
            0.5,
            0.5,
            "Muy pocos bins ocupados",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_GENERAL
        )
        ax.grid(alpha=0.3)
        continue

    ranks = np.arange(1, len(p_rank) + 1)

    # --------------------------------------------------------
    # Ajuste BRF
    # --------------------------------------------------------

    try:

        p_fit, a_fit, b_fit = fit_brf_rank_distribution(
            p_rank
        )

        rmse_log, mae_log, r2_log = fit_metrics(
            p_rank,
            p_fit
        )

        # Distribución observada
        ax.scatter(
            ranks,
            p_rank,
            s=24,
            alpha=0.8,
            label="Distribución observada"
        )

        # Ajuste BRF
        ax.plot(
            ranks,
            p_fit,
            linewidth=LINE_WIDTH,
            label=(
                rf"BRF: $a={a_fit:.3f}$, "
                rf"$b={b_fit:.3f}$"
            )
        )

        summary.append({
            "archivo": fname,
            "m": m,
            "tau": tau,
            "n_Z": len(z),
            "bins_totales": BINS,
            "bins_ocupados": len(p_rank),
            "a_BRF": a_fit,
            "b_BRF": b_fit,
            "RMSE_log": rmse_log,
            "MAE_log": mae_log,
            "R2_log": r2_log
        })

    except Exception as e:

        ax.scatter(
            ranks,
            p_rank,
            s=24,
            alpha=0.8,
            label="Distribución observada"
        )

        ax.text(
            0.05,
            0.08,
            f"Error ajuste BRF:\n{e}",
            transform=ax.transAxes,
            fontsize=9
        )

    # --------------------------------------------------------
    # FORMATO
    # --------------------------------------------------------

    if LOGLOG:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(alpha=0.3)

    ax.tick_params(
        axis="both",
        labelsize=FONT_TICKS
    )

    ax.legend(fontsize=9)


# Ejes sobrantes
for ax in axs[n:]:
    ax.axis("off")


# Etiquetas globales
fig.text(
    0.5,
    0.02,
    "Rango $r$",
    ha="center",
    fontsize=FONT_GENERAL
)

fig.text(
    0.02,
    0.5,
    r"Probabilidad $p_{(r)}$",
    va="center",
    rotation="vertical",
    fontsize=FONT_GENERAL
)


plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.08,
    wspace=0.28,
    hspace=0.38
)


# ============================================================
# GUARDADO
# ============================================================

if SAVE_FIG:
    plt.savefig(
        FIG_NAME,
        dpi=250,
        bbox_inches="tight"
    )

if SAVE_CSV and summary:
    pd.DataFrame(summary).to_csv(
        CSV_NAME,
        index=False
    )

plt.show()
