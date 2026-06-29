import numpy as np
import matplotlib.pyplot as plt
import math

from automutualinformation import sequential_mutual_information as smi

# ============================================================
# Configuración
# ============================================================

MELODY_DIR = "melodies"
n_melodies = 23

distances = np.arange(1, 11)

# Si True: sombreado = MI ± sqrt(var)
# Si False: sombreado = MI ± var
SHADE_STD = True

# ============================================================
# Funciones auxiliares
# ============================================================

def to_1d_array(x):
    """
    Convierte la salida de smi a arreglo 1D.
    Útil porque algunas funciones regresan shape (1, n_distances)
    cuando la entrada fue [signal].
    """
    x = np.asarray(x, dtype=float)
    return np.squeeze(x).ravel()


def compute_smi_for_signal(signal, distances):
    """
    Calcula MI y MI shuffle para una melodía.
    """
    (MI, MI_var), (shuff_MI, shuff_MI_var) = smi(
        [signal],
        distances=distances
    )

    MI = to_1d_array(MI)
    MI_var = to_1d_array(MI_var)
    shuff_MI = to_1d_array(shuff_MI)
    shuff_MI_var = to_1d_array(shuff_MI_var)

    return MI, MI_var, shuff_MI, shuff_MI_var


# ============================================================
# Calcular información mutua para todas las melodías
# ============================================================

results = {}

for piece_id in range(1, n_melodies + 1):
    path = f"{MELODY_DIR}/{piece_id}.npy"
    signal = np.load(path)

    # Seguridad: vector 1D y sin NaN
    signal = np.asarray(signal, dtype=float).ravel()
    signal = signal[~np.isnan(signal)]

    MI, MI_var, shuff_MI, shuff_MI_var = compute_smi_for_signal(
        signal,
        distances
    )

    results[piece_id] = {
        "MI": MI,
        "MI_var": MI_var,
        "shuff_MI": shuff_MI,
        "shuff_MI_var": shuff_MI_var
    }

# ============================================================
# Límites comunes del eje y
# ============================================================

y_values = []

for piece_id in results:
    MI = results[piece_id]["MI"]
    MI_var = results[piece_id]["MI_var"]
    shuff_MI = results[piece_id]["shuff_MI"]
    shuff_MI_var = results[piece_id]["shuff_MI_var"]

    if SHADE_STD:
        MI_err = np.sqrt(np.maximum(MI_var, 0))
        shuff_err = np.sqrt(np.maximum(shuff_MI_var, 0))
    else:
        MI_err = MI_var
        shuff_err = shuff_MI_var

    y_values.extend(MI)
    y_values.extend(shuff_MI)
    y_values.extend(MI - MI_err)
    y_values.extend(MI + MI_err)
    y_values.extend(shuff_MI - shuff_err)
    y_values.extend(shuff_MI + shuff_err)

y_values = np.asarray(y_values, dtype=float)
y_values = y_values[np.isfinite(y_values)]

ymin = np.min(y_values)
ymax = np.max(y_values)

padding = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
ymin -= padding
ymax += padding

# Como MI no debería ser negativa, opcionalmente fijamos el mínimo en 0
ymin = min(0, ymin)

# ============================================================
# Graficar subplots
# ============================================================

ncols = 5
nrows = math.ceil(n_melodies / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(16, 10),
    sharex=True,
    sharey=True
)

axes = axes.ravel()

for idx, piece_id in enumerate(range(1, n_melodies + 1)):
    ax = axes[idx]

    MI = results[piece_id]["MI"]
    MI_var = results[piece_id]["MI_var"]
    shuff_MI = results[piece_id]["shuff_MI"]
    shuff_MI_var = results[piece_id]["shuff_MI_var"]

    if SHADE_STD:
        MI_err = np.sqrt(np.maximum(MI_var, 0))
        shuff_err = np.sqrt(np.maximum(shuff_MI_var, 0))
    else:
        MI_err = MI_var
        shuff_err = shuff_MI_var

    # MI observada
    ax.plot(
        distances,
        MI,
        linewidth=1.5,
        label="MI"
    )

    ax.fill_between(
        distances,
        MI - MI_err,
        MI + MI_err,
        alpha=0.25
    )

    # MI shuffle
    ax.plot(
        distances,
        shuff_MI,
        linewidth=1.5,
        linestyle="--",
        label="Shuffle MI"
    )

    ax.fill_between(
        distances,
        shuff_MI - shuff_err,
        shuff_MI + shuff_err,
        alpha=0.25
    )

    ax.set_title(f"Melodía {piece_id}")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

# Ocultar subplots vacíos
for j in range(n_melodies, len(axes)):
    axes[j].axis("off")

# Etiquetas comunes
fig.supxlabel("Distancia")
fig.supylabel("Información mutua secuencial")

# Leyenda única
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

import pandas as pd

rows = []

for piece_id in results:
    for i, d in enumerate(distances):
        rows.append({
            "piece_id": piece_id,
            "distance": d,
            "MI": results[piece_id]["MI"][i],
            "MI_var": results[piece_id]["MI_var"][i],
            "shuff_MI": results[piece_id]["shuff_MI"][i],
            "shuff_MI_var": results[piece_id]["shuff_MI_var"][i],
        })

df_smi = pd.DataFrame(rows)
df_smi.to_csv("sequential_mutual_information_melodies.csv", index=False)

print(df_smi.head())