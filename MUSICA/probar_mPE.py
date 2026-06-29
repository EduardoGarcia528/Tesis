import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import mi_libreria as ml

# ============================================================
# Configuración
# ============================================================

MELODY_DIR = "melodies"

m_values = np.arange(3, 8)   # m = 3,4,5,6,7
n_melodies = 23

tau = 1
n_surrogates = 300
random_seed = 1234

# Si True: sombreado = media ± std
# Si False: sombreado = media ± varianza
SHADE_STD = True

rng = np.random.default_rng(random_seed)

# ============================================================
# Funciones auxiliares
# ============================================================

def as_scalar(x):
    """
    Convierte la salida de la función a escalar float.
    """
    x = np.asarray(x, dtype=float)
    return float(np.squeeze(x))


def compute_modified_PE(arr, m, tau=1):
    """
    Calcula modified permutation entropy para una melodía.
    """
    return as_scalar(ml.modified_permutation_entropy(arr, m, tau=tau,norm=True))


def compute_shuffle_PE_stats(arr, m, tau=1, n_surrogates=200, rng=None):
    """
    Calcula PE sobre shuffles explícitos de arr.
    
    Regresa:
        mu_shuff : media del nulo
        var_shuff: varianza del nulo
        values   : todos los valores del nulo
    """
    if rng is None:
        rng = np.random.default_rng()

    values = np.empty(n_surrogates, dtype=float)

    for s in range(n_surrogates):
        arr_shuff = rng.permutation(arr)
        values[s] = compute_modified_PE(arr_shuff, m, tau=tau)

    mu_shuff = np.nanmean(values)
    var_shuff = np.nanvar(values, ddof=1)

    return mu_shuff, var_shuff, values


# ============================================================
# Calcular PE observada y PE shuffle
# ============================================================

rows = []

for piece_id in range(1, n_melodies + 1):
    path = f"{MELODY_DIR}/{piece_id}.npy"
    arr = np.load(path)
    print(len(arr))

    # Seguridad: vector 1D y sin NaN
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    for m in m_values:
        PE_obs = compute_modified_PE(arr, m, tau=tau)

        shuff_PE_mean, shuff_PE_var, shuff_values = compute_shuffle_PE_stats(
            arr,
            m,
            tau=tau,
            n_surrogates=n_surrogates,
            rng=rng
        )

        rows.append({
            "piece_id": piece_id,
            "m": m,
            "PE_obs": PE_obs,
            "shuff_PE_mean": shuff_PE_mean,
            "shuff_PE_var": shuff_PE_var,
            "shuff_PE_std": np.sqrt(shuff_PE_var),
            "Z_PE": (PE_obs - shuff_PE_mean) / np.sqrt(shuff_PE_var)
                    if shuff_PE_var > 0 else np.nan
        })

df_PE = pd.DataFrame(rows)

print(df_PE.head())
print()
print("Resumen Z_PE:")
print(df_PE["Z_PE"].describe())

# Opcional: guardar resultados
df_PE.to_csv("modified_permutation_entropy_m3_m7.csv", index=False)

# ============================================================
# Límites comunes del eje y
# ============================================================

y_values = []

for _, row in df_PE.iterrows():
    PE_obs = row["PE_obs"]
    mu = row["shuff_PE_mean"]
    var = row["shuff_PE_var"]

    err = np.sqrt(var) if SHADE_STD else var

    y_values.extend([
        PE_obs,
        mu,
        mu - err,
        mu + err
    ])

y_values = np.asarray(y_values, dtype=float)
y_values = y_values[np.isfinite(y_values)]

ymin = np.min(y_values)
ymax = np.max(y_values)

padding = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
ymin -= padding
ymax += padding

# Si tu modified PE está normalizada, esto fija visualmente el rango natural
# Comenta estas líneas si tu PE no está normalizada.
ymin = min(0, ymin)
ymax = max(1, ymax)

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

    sub = df_PE[df_PE["piece_id"] == piece_id].sort_values("m")

    m = sub["m"].to_numpy()
    PE_obs = sub["PE_obs"].to_numpy()
    shuff_mu = sub["shuff_PE_mean"].to_numpy()
    shuff_var = sub["shuff_PE_var"].to_numpy()

    if SHADE_STD:
        shuff_err = np.sqrt(np.maximum(shuff_var, 0))
    else:
        shuff_err = shuff_var

    # PE observada
    ax.plot(
        m,
        PE_obs,
        marker="o",
        linewidth=1.5,
        label="mPE observado"
    )

    # Media del modelo nulo shuffle
    ax.plot(
        m,
        shuff_mu,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="Shuffle mPE"
    )

    # Sombreado del modelo nulo
    ax.fill_between(
        m,
        shuff_mu - shuff_err,
        shuff_mu + shuff_err,
        alpha=1.0)

    ax.set_title(f"Melodía {piece_id}")
    ax.set_xticks(m_values)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

# Ocultar subplots vacíos
for j in range(n_melodies, len(axes)):
    axes[j].axis("off")

# Etiquetas comunes
fig.supxlabel("m")
fig.supylabel(r"modified permutation entropy")

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