import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import mi_libreria as ml

# ============================================================
# Configuración
# ============================================================

MELODY_DIR = "melodies"

max_gamma = 10
gamma_values = np.arange(1, max_gamma + 1)      # gamma_1, ..., gamma_10
C_values = np.arange(0, max_gamma + 2)          # C_0, ..., C_{max_gamma+1}

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

def as_1d_float_array(x):
    """
    Convierte la salida de una función a array 1D de float.
    """
    return np.asarray(x, dtype=float).ravel()


def compute_gamma_rank(arr, max_gamma=10):
    """
    Calcula las correlaciones de integración C_d y los índices gamma_d.

    Regresa:
        C : array con C_0, C_1, ..., C_{max_gamma+1}
        g : array con gamma_1, ..., gamma_{max_gamma}
    """
    C, g = ml.gamma_index_rank_ties(arr, max_gamma=max_gamma,mu=1)

    C = as_1d_float_array(C)
    g = as_1d_float_array(g)

    return C, g


def compute_shuffle_gamma_stats(arr, max_gamma=10, n_surrogates=200, rng=None):
    """
    Calcula C_d y gamma_d sobre shuffles explícitos de arr.

    Regresa:
        mu_C     : media del nulo para C_d
        var_C    : varianza del nulo para C_d
        C_values : matriz con valores C_d de todos los sustitutos

        mu_g     : media del nulo para gamma_d
        var_g    : varianza del nulo para gamma_d
        g_values : matriz con valores gamma_d de todos los sustitutos
    """
    if rng is None:
        rng = np.random.default_rng()

    C_values_null = np.empty((n_surrogates, max_gamma + 2), dtype=float)
    g_values_null = np.empty((n_surrogates, max_gamma), dtype=float)

    for s in range(n_surrogates):
        arr_shuff = rng.permutation(arr)

        C_s, g_s = compute_gamma_rank(
            arr_shuff,
            max_gamma=max_gamma
        )

        C_values_null[s, :] = C_s
        g_values_null[s, :] = g_s

    mu_C = np.nanmean(C_values_null, axis=0)
    var_C = np.nanvar(C_values_null, axis=0, ddof=1)

    mu_g = np.nanmean(g_values_null, axis=0)
    var_g = np.nanvar(g_values_null, axis=0, ddof=1)

    return mu_C, var_C, C_values_null, mu_g, var_g, g_values_null


# ============================================================
# Calcular gamma observado y gamma shuffle
# ============================================================

rows_gamma = []
rows_C = []

for piece_id in range(1, n_melodies + 1):
    print(piece_id)
    path = f"{MELODY_DIR}/{piece_id}.npy"
    arr = np.load(path)

    # Seguridad: vector 1D y sin NaN
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    C_obs, g_obs = compute_gamma_rank(
        arr,
        max_gamma=max_gamma
    )

    shuff_C_mean, shuff_C_var, shuff_C_values, shuff_g_mean, shuff_g_var, shuff_g_values = (
        compute_shuffle_gamma_stats(
            arr,
            max_gamma=max_gamma,
            n_surrogates=n_surrogates,
            rng=rng
        )
    )

    # Guardar gamma_1, ..., gamma_max
    for gamma_idx in gamma_values:
        i = gamma_idx - 1

        rows_gamma.append({
            "piece_id": piece_id,
            "gamma": gamma_idx,
            "g_obs": g_obs[i],
            "shuff_g_mean": shuff_g_mean[i],
            "shuff_g_var": shuff_g_var[i],
            "shuff_g_std": np.sqrt(shuff_g_var[i]),
            "Z_g": (g_obs[i] - shuff_g_mean[i]) / np.sqrt(shuff_g_var[i])
                   if shuff_g_var[i] > 0 else np.nan
        })

    # Guardar C_0, ..., C_{max_gamma+1}
    for d in C_values:
        rows_C.append({
            "piece_id": piece_id,
            "d": d,
            "C_obs": C_obs[d],
            "shuff_C_mean": shuff_C_mean[d],
            "shuff_C_var": shuff_C_var[d],
            "shuff_C_std": np.sqrt(shuff_C_var[d]),
            "Z_C": (C_obs[d] - shuff_C_mean[d]) / np.sqrt(shuff_C_var[d])
                   if shuff_C_var[d] > 0 else np.nan
        })

df_gamma = pd.DataFrame(rows_gamma)
df_C = pd.DataFrame(rows_C)

print(df_gamma.head())
print()
print("Resumen Z_g:")
print(df_gamma["Z_g"].describe())

print()
print(df_C.head())
print()
print("Resumen Z_C:")
print(df_C["Z_C"].describe())

# Guardar resultados
df_gamma.to_csv("gamma_rank_ties_gamma1_K1.csv", index=False)
df_C.to_csv("gamma_rank_ties_Cd_K1.csv", index=False)


# ============================================================
# Límites comunes del eje y para gamma
# ============================================================

y_values = []

for _, row in df_gamma.iterrows():
    g_obs = row["g_obs"]
    mu = row["shuff_g_mean"]
    var = row["shuff_g_var"]

    err = np.sqrt(var) if SHADE_STD else var

    y_values.extend([
        g_obs,
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


# ============================================================
# Graficar subplots de gamma
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

    sub = df_gamma[df_gamma["piece_id"] == piece_id].sort_values("gamma")

    gamma = sub["gamma"].to_numpy()
    g_obs = sub["g_obs"].to_numpy()
    shuff_mu = sub["shuff_g_mean"].to_numpy()
    shuff_var = sub["shuff_g_var"].to_numpy()

    if SHADE_STD:
        shuff_err = np.sqrt(np.maximum(shuff_var, 0))
    else:
        shuff_err = shuff_var

    # gamma observado
    ax.plot(
        gamma,
        g_obs,
        marker="o",
        linewidth=1.5,
        label=r"$\gamma^{(R)}$ observado"
    )

    # Media del modelo nulo shuffle
    ax.plot(
        gamma,
        shuff_mu,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label=r"Shuffle $\gamma^{(R)}$"
    )

    # Sombreado del modelo nulo
    ax.fill_between(
        gamma,
        shuff_mu - shuff_err,
        shuff_mu + shuff_err,
        alpha=0.3
    )

    ax.set_title(f"Melodía {piece_id}")
    ax.set_xticks(gamma_values)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

# Ocultar subplots vacíos
for j in range(n_melodies, len(axes)):
    axes[j].axis("off")

fig.supxlabel(r"$d$")
fig.supylabel(r"$\gamma_d^{(R)}$")

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