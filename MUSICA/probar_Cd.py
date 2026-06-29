import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ============================================================
# Configuración
# ============================================================

CSV_PATH = "gamma_rank_ties_Cd_mu1.csv"

# Si True: sombreado = media ± std
# Si False: sombreado = media ± varianza
SHADE_STD = True

# Si True: usa escala logarítmica en y
# Útil si C_d cae muy rápido con d
LOG_Y = True

# ============================================================
# Cargar datos
# ============================================================

df_C = pd.read_csv(CSV_PATH)

# Seguridad: ordenar e inferir valores
df_C = df_C.sort_values(["piece_id", "d"]).reset_index(drop=True)

piece_ids = np.sort(df_C["piece_id"].unique())
d_values = np.sort(df_C["d"].unique())

n_melodies = len(piece_ids)

# Si no existe shuff_C_std, calcularla desde shuff_C_var
if "shuff_C_std" not in df_C.columns:
    df_C["shuff_C_std"] = np.sqrt(np.maximum(df_C["shuff_C_var"], 0))

# Si no existe Z_C, calcularlo
if "Z_C" not in df_C.columns:
    df_C["Z_C"] = np.where(
        df_C["shuff_C_var"] > 0,
        (df_C["C_obs"] - df_C["shuff_C_mean"]) / np.sqrt(df_C["shuff_C_var"]),
        np.nan
    )

print(df_C.head())
print()
print("Resumen Z_C:")
print(df_C["Z_C"].describe())

# ============================================================
# Límites comunes del eje y
# ============================================================

y_values = []

for _, row in df_C.iterrows():
    C_obs = row["C_obs"]
    mu = row["shuff_C_mean"]
    var = row["shuff_C_var"]

    err = np.sqrt(var) if SHADE_STD else var

    y_values.extend([
        C_obs,
        mu,
        mu - err,
        mu + err
    ])

y_values = np.asarray(y_values, dtype=float)
y_values = y_values[np.isfinite(y_values)]

if LOG_Y:
    y_values = y_values[y_values > 0]

ymin = np.min(y_values)
ymax = np.max(y_values)

padding = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
ymin -= padding
ymax += padding

if not LOG_Y:
    ymin = min(0, ymin)

# ============================================================
# Graficar subplots de C_d
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

for idx, piece_id in enumerate(piece_ids):
    ax = axes[idx]

    sub = df_C[df_C["piece_id"] == piece_id].sort_values("d")

    d = sub["d"].to_numpy()
    C_obs = sub["C_obs"].to_numpy()
    shuff_mu = sub["shuff_C_mean"].to_numpy()
    shuff_var = sub["shuff_C_var"].to_numpy()

    if SHADE_STD:
        shuff_err = np.sqrt(np.maximum(shuff_var, 0))
    else:
        shuff_err = shuff_var

    # C_d observado
    ax.plot(
        d,
        C_obs,
        marker="o",
        linewidth=1.5,
        label=r"$C_d$ observado"
    )

    # Media del modelo nulo shuffle
    ax.plot(
        d,
        shuff_mu,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label=r"Shuffle $C_d$"
    )

    # Sombreado del modelo nulo
    ax.fill_between(
        d,
        shuff_mu - shuff_err,
        shuff_mu + shuff_err,
        alpha=0.3
    )

    ax.set_title(f"Melodía {piece_id}")
    ax.set_xticks(d_values)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    if LOG_Y:
        ax.set_yscale("log")

# Ocultar subplots vacíos
for j in range(n_melodies, len(axes)):
    axes[j].axis("off")

# Etiquetas comunes
fig.supxlabel(r"$d$")
fig.supylabel(r"Integral de correlación $C_d$")

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