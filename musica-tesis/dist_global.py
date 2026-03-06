# -*- coding: utf-8 -*-
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CACHE_DIR = "cache_pe_zscore"   # carpeta donde guardaste los .pkl

FONT_GENERAL = 12
FONT_TICKS = 11
TITLE_SIZE = 12
LINE_WIDTH = 1.2

BINS = 100
DENSITY = True   # True -> PDF; False -> conteos

# Tamaño por panel (aprox). La figura final escala con filas/cols.
PANEL_W = 5.0
PANEL_H = 3.0

# =========================
# UTILIDADES
# =========================
def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", str(text))]

def list_cache_pkls(cache_dir=CACHE_DIR):
    files = [f for f in os.listdir(cache_dir) if f.lower().endswith(".pkl")]
    return sorted(files, key=natural_key)

def parse_panel_title_from_filename(fname):
    mD = re.search(r"_D(\d+)", fname)
    mt = re.search(r"_tau(\d+)", fname)
    if mD and mt:
        return f"m={mD.group(1)}, tau={mt.group(1)}"
    return os.path.splitext(fname)[0]

def load_z_values(pkl_path):
    df = pd.read_pickle(pkl_path)
    if "z" not in df.columns:
        raise ValueError(f"El archivo no contiene columna 'z': {pkl_path}")
    z = df["z"].to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    return z

def choose_grid(n_panels):
    """
    Elige (nrows, ncols) lo más 'cuadrado' posible.
    Ej: 1->(1,1), 2->(1,2), 3->(2,2), 4->(2,2), 5->(2,3), 6->(2,3), 7->(3,3)...
    """
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols

# =========================
# PLOTEO
# =========================
pkls = list_cache_pkls(CACHE_DIR)
if len(pkls) == 0:
    raise FileNotFoundError(f"No encontré archivos .pkl en: {CACHE_DIR}")

n = len(pkls)
nrows, ncols = choose_grid(n)

fig_w = PANEL_W * ncols
fig_h = PANEL_H * nrows
fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)

axs = np.atleast_1d(axs).ravel()

for ax, fname in zip(axs, pkls):
    pkl_path = os.path.join(CACHE_DIR, fname)
    z = load_z_values(pkl_path)

    title = parse_panel_title_from_filename(fname)
    ax.set_title(title, fontsize=TITLE_SIZE)

    if z.size == 0:
        ax.text(0.5, 0.5, "Sin Z finitos", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_GENERAL)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=FONT_TICKS)
        continue

    med = np.median(z)

    ax.hist(z, bins=BINS, density=DENSITY, alpha=0.9)
    ax.set_xlim(min(z) - 0.5, max(z) + 0.5)
    ax.axvline(0.0, linestyle="--", linewidth=LINE_WIDTH, color="black", alpha=0.9)
    ax.axvline(med, linestyle="-", linewidth=LINE_WIDTH, color="black", alpha=0.9)

    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=FONT_TICKS)

# Apaga ejes sobrantes si la grilla es más grande que n
for ax in axs[n:]:
    ax.axis("off")

# Etiquetas globales
fig.text(0.5, 0.02, "Z-score (PE)", ha="center", fontsize=FONT_GENERAL)
fig.text(0.02, 0.5, "Densidad" if DENSITY else "Conteo", va="center", rotation="vertical",
         fontsize=FONT_GENERAL)

plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
plt.show()