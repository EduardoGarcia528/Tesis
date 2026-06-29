import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import mi_libreria as ml

# ============================================================
# Configuración
# ============================================================

MELODY_DIR = "melodies"

tau_values = np.arange(1, 11)   # tau = 1,2,...,10
n_melodies = 23
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


def compute_S_eff(arr, tau=1):
    """
    Calcula S_eff univariante para una melodía.
    """
    return as_scalar(
        ml.entropia_J(arr, None, tau=tau)
    )


def compute_shuffle_S_eff_stats(arr, tau=1, n_surrogates=200, rng=None):
    """
    Calcula S_eff sobre shuffles explícitos de arr.

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
        values[s] = compute_S_eff(arr_shuff, tau=tau)

    mu_shuff = np.nanmean(values)
    var_shuff = np.nanvar(values, ddof=1)

    return mu_shuff, var_shuff, values


# ============================================================
# Calcular S_eff observada y S_eff shuffle
# ============================================================

rows = []

for piece_id in range(1, n_melodies + 1):
    path = f"{MELODY_DIR}/{piece_id}.npy"
    arr = np.load(path)

    # Seguridad: vector 1D y sin NaN
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    for tau in tau_values:
        S_obs = compute_S_eff(arr, tau=tau)

        shuff_S_mean, shuff_S_var, shuff_values = compute_shuffle_S_eff_stats(
            arr,
            tau=tau,
            n_surrogates=n_surrogates,
            rng=rng
        )

        rows.append({
            "piece_id": piece_id,
            "tau": tau,
            "S_obs": S_obs,
            "shuff_S_mean": shuff_S_mean,
            "shuff_S_var": shuff_S_var,
            "shuff_S_std": np.sqrt(shuff_S_var),
            "Z_S": (S_obs - shuff_S_mean) / np.sqrt(shuff_S_var)
                   if shuff_S_var > 0 else np.nan
        })

df_S = pd.DataFrame(rows)

print(df_S.head())
print()
print("Resumen Z_S:")
print(df_S["Z_S"].describe())

# Opcional: guardar resultados
df_S.to_csv("S_eff_tau1_tau10.csv", index=False)

# ============================================================
# Límites comunes del eje y
# ============================================================

y_values = []

for _, row in df_S.iterrows():
    S_obs = row["S_obs"]
    mu = row["shuff_S_mean"]
    var = row["shuff_S_var"]

    err = np.sqrt(var) if SHADE_STD else var

    y_values.extend([
        S_obs,
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

# S_eff normalmente vive en [0, 1]
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

    sub = df_S[df_S["piece_id"] == piece_id].sort_values("tau")

    tau = sub["tau"].to_numpy()
    S_obs = sub["S_obs"].to_numpy()
    shuff_mu = sub["shuff_S_mean"].to_numpy()
    shuff_var = sub["shuff_S_var"].to_numpy()

    if SHADE_STD:
        shuff_err = np.sqrt(np.maximum(shuff_var, 0))
    else:
        shuff_err = shuff_var

    # S_eff observada
    ax.plot(
        tau,
        S_obs,
        marker="o",
        linewidth=1.5,
        label=r"$S_{\mathrm{eff}}$ observado"
    )

    # Media del modelo nulo shuffle
    ax.plot(
        tau,
        shuff_mu,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label=r"Shuffle $S_{\mathrm{eff}}$"
    )

    # Sombreado del modelo nulo
    ax.fill_between(
        tau,
        shuff_mu - shuff_err,
        shuff_mu + shuff_err,
        alpha=0.3
    )

    ax.set_title(f"Melodía {piece_id}")
    ax.set_xticks(tau_values)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

# Ocultar subplots vacíos
for j in range(n_melodies, len(axes)):
    axes[j].axis("off")

# Etiquetas comunes
fig.supxlabel(r"$\tau$")
fig.supylabel(r"$S_{\mathrm{eff}}$")

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