# -*- coding: utf-8 -*-
import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modified_PE import modified_permutation_entropy
from funciones import angulos_alpha, permutation_entropy
from iaaft import iaaft
from gamma_4 import gamma_index_jacobs
from gamma_5 import gamma_index_jacobs_rank_ties
# from circular_gamma import gamma_index_jacobs_circular

# =========================================================
# ESTILO GENERAL
# =========================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

DOT_SIZE = 32
EDGE_ALPHA = 0.75
LINE_WIDTH = 0.9

FONT_GENERAL = 12
FONT_TICKS = 10
LEGEND_FONTSIZE = 11
TITLE_SIZE = 13

BOTTOM_MARGIN = 0.28
LEFT_MARGIN = 0.07
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.90

# =========================================================
# PARÁMETROS DEL TEST NULO
# =========================================================
N_SURROGATES = 800
RANDOM_STATE = 12345
ALTERNATIVE = "less"   # 'two-sided', 'greater', 'less'
NORMALIZE = True

# rojo si ningún surrogate fue tan extremo como el dato
USE_P_RAW_ZERO_FOR_RED = True
ALPHA = 0.05   # solo se usa si USE_P_RAW_ZERO_FOR_RED = False

# =========================================================
# CONFIGURACIÓN DE LOS PANELES
# =========================================================
PANEL_CONFIGS = [
    {"measure": "gamma_rank", "D": 1, "tau": 2, "type_null": "shuffle", "title": fr"$\gamma_1$ (notes, shuffle): $\mu=2$"},
    {"measure": "mPE_interval", "D": 5, "tau": 1, "type_null": "shuffle", "title": r"mPE (interval, shuffle): $m=5,\ \tau=1$"},
    {"measure": "mPE",          "D": 5, "tau": 1, "type_null": "iaaft",   "title": r"mPE (notes, IAAFT): $m=5,\ \tau=1$"},
    {"measure": "mPE_interval", "D": 5, "tau": 1, "type_null": "iaaft",   "title": r"mPE (interval, IAAFT): $m=5,\ \tau=1$"},
]

# =========================================================
# DATOS DE ENTRADA
# =========================================================
MELODIES_FOLDER = Path("melodies")
MAX_FILES = 23

# =========================================================
# CACHÉ
# =========================================================
CACHE_DIR = "cache_zscore_melodies"
FORCE_RECOMPUTE = False

# =========================================================
# AUXILIARES
# =========================================================
def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', str(text))]

def load_melody_dataset(folder, max_files=23):
    files = sorted(folder.glob("*.npy"), key=lambda p: natural_key(p.stem))

    if len(files) < max_files:
        raise ValueError(
            f"Se encontraron solo {len(files)} archivos .npy en '{folder}', "
            f"pero se esperaban al menos {max_files}."
        )

    files = files[:max_files]

    series_dict = {}
    labels = []

    for file_path in files:
        serie = np.load(file_path, allow_pickle=True)
        serie = np.asarray(serie).squeeze()

        if serie.ndim != 1:
            raise ValueError(f"El archivo {file_path.name} no contiene una serie univariante.")

        label = file_path.stem
        labels.append(label)
        series_dict[label] = serie.astype(float)

    return series_dict, labels

def empirical_p_raw(pe_surrogates, pe_obs, mu_null, alternative="two-sided"):
    pe_surrogates = np.asarray(pe_surrogates, dtype=float)

    if alternative == "greater":
        return np.mean(pe_surrogates >= pe_obs)
    elif alternative == "less":
        return np.mean(pe_surrogates <= pe_obs)
    elif alternative == "two-sided":
        return np.mean(np.abs(pe_surrogates - mu_null) >= np.abs(pe_obs - mu_null))
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")

def compute_measure(x, measure, D, tau):
    x = np.asarray(x, dtype=float)

    if "PE" in measure:
        if "interval" in measure:
            return modified_permutation_entropy(np.diff(x), m=D, tau=tau)
        else:
            return modified_permutation_entropy(x, m=D, tau=tau)

    elif "Cd" in measure:
        if "interval" in measure:
            C, _ = gamma_index_jacobs(np.diff(x), max_gamma=D, mu=tau)
            return 1 - C[-1]
        else:
            C, _ = gamma_index_jacobs(x, max_gamma=D, mu=tau)
            return 1 - C[-1]
    elif "gamma_rank" in measure:
        if "interval" in measure:
            _, C = gamma_index_jacobs_rank_ties(np.diff(x), max_gamma=D, mu=tau)
            return 1 - C[0]
        else:
            _, C = gamma_index_jacobs_rank_ties(x, max_gamma=D, mu=tau)
            return 1 - C[0]

    else:
        raise ValueError(f"Medida no reconocida: {measure}")

def pe_stats_for_series(
    x,
    measure,
    D,
    tau,
    n_surrogates,
    type_null,
    alternative,
    random_state=None
):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)

    measure_obs = compute_measure(x, measure, D, tau)

    surrogates_values = np.empty(n_surrogates, dtype=float)

    if type_null == "shuffle":
        for k in range(n_surrogates):
            if "interval" in measure:
                x_surr = rng.permutation(np.diff(x))
            else:
                x_surr = rng.permutation(x)

            if "PE" in measure:
                surrogates_values[k] = modified_permutation_entropy(x_surr, m=D, tau=tau)
            elif "Cd" in measure:
                C, _ = gamma_index_jacobs(x_surr, max_gamma=D, mu=tau)
                surrogates_values[k] = 1 - C[-1]
            elif "gamma_rank" in measure:
                _, C = gamma_index_jacobs_rank_ties(x_surr, max_gamma=D, mu=tau)
                surrogates_values[k] = 1 - C[0]

    elif type_null == "iaaft":
        if "interval" in measure:
            x_base = np.diff(x)
        else:
            x_base = x

        x_surr_all = iaaft(x_base, n_surrogates)

        for k in range(n_surrogates):
            x_surr = x_surr_all[k, :]

            if "PE" in measure:
                surrogates_values[k] = modified_permutation_entropy(x_surr, m=D, tau=tau)
            elif "Cd" in measure:
                C, _ = gamma_index_jacobs(x_surr, max_gamma=D, mu=tau)
                surrogates_values[k] = 1 - C[-1]
    else:
        raise ValueError("type_null debe ser 'shuffle' o 'iaaft'.")

    mu_null = np.mean(surrogates_values)
    sigma_null = np.std(surrogates_values, ddof=1)

    if sigma_null == 0:
        z = np.nan
    else:
        z = (measure_obs - mu_null) / sigma_null

    if alternative == "greater":
        p_value = (np.sum(surrogates_values >= measure_obs) + 1) / (n_surrogates + 1)
    elif alternative == "less":
        p_value = (np.sum(surrogates_values <= measure_obs) + 1) / (n_surrogates + 1)
    elif alternative == "two-sided":
        p_value = (np.sum(np.abs(surrogates_values - mu_null) >= np.abs(measure_obs - mu_null)) + 1) / (n_surrogates + 1)
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")

    p_raw = empirical_p_raw(surrogates_values, measure_obs, mu_null, alternative=alternative)

    return {
        "measure_obs": measure_obs,
        "mu_null": mu_null,
        "sigma_null": sigma_null,
        "z": z,
        "p_value": p_value,
        "p_raw": p_raw
    }

def build_cache_key(measure, D, tau, type_null, normalize, alternative):
    if "PE" in measure:
        return f"{measure}_D{D}_tau{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    elif "Cd" in measure:
        return f"{measure}_d{D}_mu{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    else:
        return f"{measure}_{type_null}_{alternative}_norm{int(normalize)}"

def get_cache_path(measure, cache_dir, D, tau, type_null, normalize, alternative):
    os.makedirs(cache_dir, exist_ok=True)
    fname = build_cache_key(measure, D, tau, type_null, normalize, alternative) + ".pkl"
    return os.path.join(cache_dir, fname)

def compute_panel_dataframe(
    series_dict,
    file_labels,
    measure,
    D,
    tau,
    n_surrogates,
    type_null,
    normalize,
    alternative,
    random_state
):
    rows = []
    seed_base = int(random_state) if random_state is not None else None

    for idx, label in enumerate(file_labels):
        x = np.asarray(series_dict[label], dtype=float)

        if seed_base is None:
            seed_here = None
        else:
            seed_here = seed_base + 100000 * D + 1000 * tau + idx

        stats = pe_stats_for_series(
            x=x,
            measure=measure,
            D=D,
            tau=tau,
            n_surrogates=n_surrogates,
            type_null=type_null,
            alternative=alternative,
            random_state=seed_here
        )

        rows.append({
            "file_label": label,
            "file_index": idx + 1,
            "length": len(x),
            "D": D,
            "tau": tau,
            "n_surrogates": n_surrogates,
            "normalize": normalize,
            "alternative": alternative,
            "random_state": seed_here,
            "measure_obs": stats["measure_obs"],
            "mu_null": stats["mu_null"],
            "sigma_null": stats["sigma_null"],
            "z": stats["z"],
            "p_value": stats["p_value"],
            "p_raw": stats["p_raw"]
        })

    return pd.DataFrame(rows)

def get_or_compute_panel_dataframe(
    series_dict,
    file_labels,
    measure,
    D,
    tau,
    n_surrogates,
    type_null,
    normalize,
    alternative,
    random_state,
    cache_dir=CACHE_DIR,
    force_recompute=FORCE_RECOMPUTE
):
    cache_path = get_cache_path(
        measure=measure,
        cache_dir=cache_dir,
        D=D,
        tau=tau,
        type_null=type_null,
        normalize=normalize,
        alternative=alternative,
    )

    if (not force_recompute) and os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        print(f"[CACHE] Cargado: {cache_path}")
        return df

    print(f"[CACHE] Calculando panel {measure} D={D}, tau={tau}, {type_null}...")
    df = compute_panel_dataframe(
        series_dict=series_dict,
        file_labels=file_labels,
        measure=measure,
        D=D,
        tau=tau,
        n_surrogates=n_surrogates,
        type_null=type_null,
        normalize=normalize,
        alternative=alternative,
        random_state=random_state
    )
    df.to_pickle(cache_path)
    print(f"[CACHE] Guardado: {cache_path}")
    return df

def point_is_red(p_raw, p_value, use_p_raw_zero=True, alpha=0.05):
    if use_p_raw_zero:
        return (p_raw == 0.0)
    return (p_value <= alpha)

def plot_panel(ax, df_panel, file_labels, title,
               use_p_raw_zero=True, alpha=0.05):
    df_panel = df_panel.copy()
    df_panel["file_label"] = pd.Categorical(df_panel["file_label"], categories=file_labels, ordered=True)
    df_panel = df_panel.sort_values("file_label")

    x_positions = np.arange(1, len(file_labels) + 1)

    zvals = df_panel["z"].to_numpy(dtype=float)
    pvals = df_panel["p_value"].to_numpy(dtype=float)
    praws = df_panel["p_raw"].to_numpy(dtype=float)

    finite_mask = np.isfinite(zvals)
    zvals_plot = zvals[finite_mask]
    x_plot = x_positions[finite_mask]
    pvals_plot = pvals[finite_mask]
    praws_plot = praws[finite_mask]

    red_mask = np.array([
        point_is_red(pr, pv, use_p_raw_zero=use_p_raw_zero, alpha=alpha)
        for pr, pv in zip(praws_plot, pvals_plot)
    ], dtype=bool)

    colors = np.where(red_mask, "red", "blue")

    ax.scatter(
        x_plot, zvals_plot,
        s=DOT_SIZE,
        alpha=EDGE_ALPHA,
        facecolors="none",
        edgecolors=colors,
        linewidths=1.0
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=LINE_WIDTH, alpha=0.9)

    if use_p_raw_zero:
        red_label = r"Significativa ($p_{\mathrm{raw}}=0$)"
    else:
        red_label = fr"Significativa ($p \leq {alpha}$)"

    ax.plot([], [], marker='o', color='none', markeredgecolor='red', label=red_label)
    ax.plot([], [], marker='o', color='none', markeredgecolor='blue', label='No significativa')

    total_points = len(zvals_plot)
    total_red_points = np.sum(red_mask)

    if total_points > 0:
        total_percentage_red = 100.0 * total_red_points / total_points
        ax.text(
            0.01, 0.98,
            f"Total: {total_percentage_red:.1f}%",
            ha="left", va="top",
            fontsize=FONT_GENERAL,
            color="black",
            transform=ax.transAxes
        )

    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(axis='y', alpha=0.4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(file_labels, rotation=90, fontsize=FONT_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_TICKS)

    return zvals_plot

# =========================================================
# CARGA DE DATOS
# =========================================================
series_dict, file_labels = load_melody_dataset(MELODIES_FOLDER, max_files=MAX_FILES)

# =========================================================
# OBTENER DATOS DESDE CACHÉ O CÁLCULO
# =========================================================
panel_dfs = []
all_z_global = []

for cfg in PANEL_CONFIGS:
    df_panel = get_or_compute_panel_dataframe(
        series_dict=series_dict,
        file_labels=file_labels,
        measure=cfg["measure"],
        D=cfg["D"],
        tau=cfg["tau"],
        n_surrogates=N_SURROGATES,
        type_null=cfg["type_null"],
        normalize=NORMALIZE,
        alternative=ALTERNATIVE,
        random_state=RANDOM_STATE,
        cache_dir=CACHE_DIR,
        force_recompute=FORCE_RECOMPUTE
    )
    panel_dfs.append(df_panel)

# =========================================================
# PLOTEO 2x2
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=False, sharey=True)

for ax, cfg, df_panel in zip(axs.ravel(), PANEL_CONFIGS, panel_dfs):
    zvals_panel = plot_panel(
        ax=ax,
        df_panel=df_panel,
        file_labels=file_labels,
        title=cfg["title"],
        use_p_raw_zero=USE_P_RAW_ZERO_FOR_RED,
        alpha=ALPHA
    )
    if zvals_panel.size > 0:
        all_z_global.extend(zvals_panel[np.isfinite(zvals_panel)].tolist())

all_z_global = np.array(all_z_global, dtype=float)
if all_z_global.size > 0:
    zmax = np.nanmax(np.abs(all_z_global))
    if np.isfinite(zmax) and zmax > 0:
        pad = 0.08 * zmax
        ylim = (-zmax - pad, zmax + pad)
        for ax in axs.ravel():
            ax.set_ylim(*ylim)

for ax in axs.ravel():
    ax.set_ylabel("Z-score", fontsize=FONT_GENERAL)

handles, labels_leg = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels_leg,
    loc='upper center',
    ncol=3,
    fontsize=LEGEND_FONTSIZE,
    frameon=False
)

plt.subplots_adjust(
    left=LEFT_MARGIN,
    right=RIGHT_MARGIN,
    bottom=BOTTOM_MARGIN,
    top=TOP_MARGIN,
    wspace=0.08,
    hspace=0.18
)

plt.margins(x=0.02)
plt.show()

# =========================================================
# TABLA RESUMEN
# =========================================================
summary_dict = {}
for cfg, df_panel in zip(PANEL_CONFIGS, panel_dfs):
    summary_dict[cfg["title"]] = df_panel.set_index("file_label")["z"].reindex(file_labels)

df_zscores = pd.DataFrame(summary_dict, index=file_labels)

print("\nZ-scores por archivo:")
print(df_zscores)