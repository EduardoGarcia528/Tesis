# -*- coding: utf-8 -*-
"""Delta RMSE entre ajuste exponencial y power law de C_d, por compositor."""

import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mi_libreria as ml


# =========================================================
# PARÁMETROS
# =========================================================
DATA_DIR = "data"
METHOD = "hybrid"

# "midi" o "intervals"
ANALYSIS_MODE = "intervals"
# True reproduce la convención del archivo Cd_bestfit_C1null_corpus.py.
# False usa intervalos con signo, como np.diff(x).
INTERVAL_ABS = False

C1_NULL_TARGET = 0.5807
MAX_GAMMA = 7
FIT_D_MIN = 1
FIT_D_MAX = 8

DOT_SIZE = 22
EDGE_ALPHA = 0.55
JITTER_STD = 0.06
LINE_WIDTH = 0.9
FONT_GENERAL = 12
FONT_TICKS = 11
LEGEND_FONTSIZE = 11
JITTER_SEED = 12345

SAVE_PNG = False
OUTPUT_PNG = "delta_RMSE_Cd_por_compositor.png"
PNG_DPI = 220

SAVE_CSV = False
OUTPUT_CSV = "delta_RMSE_Cd_por_compositor.csv"


# =========================================================
# ORDEN DE LOS COMPOSITORES: igual que dists(1).py
# =========================================================
def natural_key(text):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(text))]


def composer_sort_key(item):
    composer, meta = item
    idx = meta.get("Indice", np.inf)
    byear = meta.get("Birth_year", "999999")
    try:
        byear = int(byear)
    except Exception:
        byear = 999999
    return idx, byear, composer


def composer_labels(datos_composers):
    ordered = sorted(datos_composers.items(), key=composer_sort_key)
    names, labels = [], []
    for composer, meta in ordered:
        byear = meta.get("Birth_year", "")
        names.append(composer)
        labels.append(f"{composer} {byear}".strip())
    return names, labels


# =========================================================
# K FIJADO POR C1_NULL
# =========================================================
def exact_C1_null_from_counts(counts, K):
    counts = np.asarray(counts, dtype=np.int64).ravel()
    N = int(np.sum(counts))
    if N < 2:
        return np.nan

    q = len(counts)
    K = int(np.clip(K, 0, max(q - 1, 0)))

    close_pairs = np.sum(counts * (counts - 1) // 2, dtype=np.int64)
    for h in range(1, K + 1):
        close_pairs += np.sum(counts[:-h] * counts[h:], dtype=np.int64)

    return float(close_pairs / (N * (N - 1) / 2.0))


def determine_K_from_C1_null(x, target=C1_NULL_TARGET):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    _, counts = np.unique(x, return_counts=True)
    n_unique = len(counts)

    c1_values = np.array([
        exact_C1_null_from_counts(counts, K)
        for K in range(n_unique)
    ])

    admissible = np.where(c1_values <= target)[0]
    K = int(admissible[-1]) if len(admissible) else 0

    return K, n_unique, float(c1_values[K])


# =========================================================
# C_d Y RMSE DE LOS DOS MODELOS
# =========================================================
def compute_C(x, K):
    C, _ = ml.gamma_index_rank_ties(
        np.asarray(x, dtype=float),
        max_gamma=MAX_GAMMA,
        K=int(0.0),
    )
    C = np.asarray(C, dtype=float).ravel()

    if len(C) < FIT_D_MAX + 1:
        raise ValueError(f"C tiene longitud {len(C)}; se necesitan {FIT_D_MAX + 1} valores.")

    return C[:FIT_D_MAX + 1]


def rmse_exponential(C):
    """log(C_d) = b - lambda*d."""
    C = np.asarray(C, dtype=float)
    d = np.arange(len(C), dtype=float)

    mask = ((d >= FIT_D_MIN) & (d <= FIT_D_MAX)
            & np.isfinite(C) & (C > 0))
    x = d[mask]
    y = np.log(C[mask])

    if len(x) < 2:
        return np.nan

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    return float(np.sqrt(np.mean((y - y_fit) ** 2)))


def rmse_powerlaw(C):
    """log(C_d) = b - alpha*log(d)."""
    C = np.asarray(C, dtype=float)
    d = np.arange(len(C), dtype=float)

    mask = ((d >= FIT_D_MIN) & (d <= FIT_D_MAX)
            & (d > 0) & np.isfinite(C) & (C > 0))
    x = np.log(d[mask])
    y = np.log(C[mask])

    if len(x) < 2:
        return np.nan

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    return float(np.sqrt(np.mean((y - y_fit) ** 2)))


def preprocess(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if ANALYSIS_MODE == "midi":
        return x

    if ANALYSIS_MODE == "intervals":
        dx = np.diff(x)
        return np.abs(dx) if INTERVAL_ABS else dx

    raise ValueError("ANALYSIS_MODE debe ser 'midi' o 'intervals'.")


# =========================================================
# CÁLCULO POR SERIE
# =========================================================
def calculate_all(composers, datos_composers):
    composer_names, _ = composer_labels(datos_composers)
    rows = []

    for composer in composer_names:
        if composer not in composers:
            continue

        print(composer)

        for serie in sorted(composers[composer], key=natural_key):
            try:
                x = preprocess(composers[composer][serie])
                if len(x) < FIT_D_MAX + 2:
                    continue

                K, n_unique, c1_null = determine_K_from_C1_null(x)
                C = compute_C(x, K)

                rmse_exp = rmse_exponential(C)
                rmse_pl = rmse_powerlaw(C)

                if not (np.isfinite(rmse_exp) and np.isfinite(rmse_pl)):
                    continue

                delta = rmse_exp - rmse_pl

                rows.append({
                    "composer": composer,
                    "serie": serie,
                    "delta_rmse": delta,
                    "rmse_exponential": rmse_exp,
                    "rmse_powerlaw": rmse_pl,
                    "K": K,
                    "n_unique": n_unique,
                    "C1_null_exact": c1_null,
                    "best_model": "exponential" if delta < 0 else "powerlaw" if delta > 0 else "tie",
                })

            except Exception as exc:
                print(f"  {serie}: {exc}")

    return pd.DataFrame(rows)


# =========================================================
# SCATTER: formato tomado de dists(1).py
# =========================================================
def make_figure(df, composer_names, labels):
    rng = np.random.default_rng(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(16, 7))

    medians = []

    for i, composer in enumerate(composer_names, start=1):
        vals = df.loc[df["composer"] == composer, "delta_rmse"].to_numpy(float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            medians.append(np.nan)
            continue

        x = rng.normal(i, JITTER_STD, len(vals))
        colors = np.where(vals < 0, "blue",
                          np.where(vals > 0, "red", "gray"))

        ax.scatter(
            x, vals,
            s=DOT_SIZE,
            alpha=EDGE_ALPHA,
            facecolors="none",
            edgecolors=colors,
            linewidths=0.8,
        )

        medians.append(np.median(vals))

    medians = np.asarray(medians, dtype=float)
    idx = np.arange(1, len(composer_names) + 1)
    ok = np.isfinite(medians)

    ax.axhline(0, color="gray", linestyle="--",
               linewidth=LINE_WIDTH, alpha=0.9)

    ax.scatter(idx[ok], medians[ok], color="black", s=DOT_SIZE,
               zorder=3, label="Mediana")

    ax.plot([], [], "o", markerfacecolor="none", markeredgecolor="blue",
            label=r"Exponencial ($\Delta RMSE<0$)")
    ax.plot([], [], "o", markerfacecolor="none", markeredgecolor="red",
            label=r"Power law ($\Delta RMSE>0$)")

    ax.set_ylabel(r"$\Delta RMSE = RMSE_{exp}-RMSE_{PL}$",
                  fontsize=FONT_GENERAL)

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90, fontsize=FONT_TICKS)
    ax.tick_params(axis="y", labelsize=FONT_TICKS)

    ax.grid(axis="both", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11),
              ncol=3, fontsize=LEGEND_FONTSIZE, frameon=False)

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.30, top=0.88)
    plt.margins(x=0.02)

    return fig


# =========================================================
# MAIN
# =========================================================
def main():
    with open(os.path.join(DATA_DIR, f"composers_{METHOD}.pkl"), "rb") as f:
        composers = pickle.load(f)

    with open(os.path.join(DATA_DIR, f"datos_composers_{METHOD}.pkl"), "rb") as f:
        datos_composers = pickle.load(f)

    composer_names, labels = composer_labels(datos_composers)
    df = calculate_all(composers, datos_composers)

    if SAVE_CSV:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"CSV guardado: {OUTPUT_CSV}")

    fig = make_figure(df, composer_names, labels)

    if SAVE_PNG:
        fig.savefig(OUTPUT_PNG, dpi=PNG_DPI, bbox_inches="tight")
        print(f"Figura guardada: {OUTPUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()
