import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ============================================================
# Configuración
# ============================================================

CSV_PATH = "h_orbit_shuffle_m3_m7.csv"

m_values = np.arange(3, 8)   # m = 3,4,5,6,7
n_melodies = 23

# Si True: sombreado = media ± std
# Si False: sombreado = media ± varianza
SHADE_STD = True

# Columnas del CSV de Julia
OBS_COL = "H_obs"
MU_COL = "mu_null"
STD_COL = "sigma_null"

# Si tu CSV tiene varianza en vez de std, cambia esto:
VAR_COL = None  # ejemplo: "var_null"

# ============================================================
# Cargar datos de H_orbit
# ============================================================

df_H = pd.read_csv(CSV_PATH)

df_H = df_H.sort_values(["piece_id", "m"]).reset_index(drop=True)

# Seguridad: asegurar tipos numéricos
for col in ["piece_id", "m", OBS_COL, MU_COL, STD_COL]:
    if col in df_H.columns:
        df_H[col] = pd.to_numeric(df_H[col], errors="coerce")

# Si no existe sigma_null pero sí existe var_null
if STD_COL not in df_H.columns:
    if VAR_COL is not None and VAR_COL in df_H.columns:
        df_H[STD_COL] = np.sqrt(np.maximum(df_H[VAR_COL].to_numpy(float), 0))
    else:
        raise ValueError(
            f"No encontré la columna {STD_COL}. "
            "Necesito sigma_null o una columna de varianza."
        )

# Calcular varianza para conservar formato similar al código de PE
df_H["shuff_H_orbit_var"] = df_H[STD_COL] ** 2
df_H["shuff_H_orbit_std"] = df_H[STD_COL]

# Z-score, por si no viene en el CSV
if "Z" not in df_H.columns:
    df_H["Z"] = np.where(
        df_H[STD_COL] > 0,
        (df_H[OBS_COL] - df_H[MU_COL]) / df_H[STD_COL],
        np.nan
    )

print(df_H.head())
print()
print("Resumen Z_H_orbit:")
print(df_H["Z"].describe())

# Opcional: guardar una copia limpia
df_H.to_csv("H_orbit_shuffle_m3_m7_python_format.csv", index=False)

# ============================================================
# Límites comunes del eje y
# ============================================================

y_values = []

for _, row in df_H.iterrows():
    H_obs = row[OBS_COL]
    mu = row[MU_COL]
    std = row[STD_COL]
    var = std**2

    err = std if SHADE_STD else var

    y_values.extend([
        H_obs,
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

# No fijamos [0,1], porque aquí se usa H_orbit sin normalizar.

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

    sub = df_H[df_H["piece_id"] == piece_id].sort_values("m")

    m = sub["m"].to_numpy()
    H_obs = sub[OBS_COL].to_numpy()
    shuff_mu = sub[MU_COL].to_numpy()
    shuff_std = sub[STD_COL].to_numpy()

    if SHADE_STD:
        shuff_err = shuff_std
    else:
        shuff_err = shuff_std**2

    # H_orbit observado
    ax.plot(
        m,
        H_obs,
        marker="o",
        linewidth=1.5,
        label=r"$H_{\mathrm{orbit}}$ observado"
    )

    # Media del modelo nulo shuffle
    ax.plot(
        m,
        shuff_mu,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label=r"Shuffle $H_{\mathrm{orbit}}$"
    )

    # Sombreado del modelo nulo
    ax.fill_between(
        m,
        shuff_mu - shuff_err,
        shuff_mu + shuff_err,
        alpha=1.0
    )

    ax.set_title(f"Melodía {piece_id}")
    ax.set_xticks(m_values)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

# Ocultar subplots vacíos
for j in range(n_melodies, len(axes)):
    axes[j].axis("off")

# Etiquetas comunes
fig.supxlabel("m")
fig.supylabel(r"$H_{\mathrm{orbit}}$")

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