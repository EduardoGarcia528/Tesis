import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuración
# ============================================================

archivo = "figuras_Cd_shuffle_absinterv_exponencial_shuffle/resumen_ajuste_exponencial_Cd_observado_y_shuffle.csv"

orden_voces = ["vln I", "vln II", "viola", "cello"]

nombres_voces = {
    "vln I": "Violín I",
    "vln II": "Violín II",
    "viola": "Viola",
    "cello": "Violonchelo"
}

# ============================================================
# Leer y preparar los datos
# ============================================================

df = pd.read_csv(archivo)

# Conservar solamente las cuatro voces analizadas
datos = df[df["voice"].isin(orden_voces)].copy()

# El valor nulo se representa mediante la media de los shuffles
datos["lambda_ratio"] = (
    datos["lambda_observed"] / datos["lambda_null_mean"]
)

# Eliminar valores no válidos, por seguridad
datos = datos.replace([float("inf"), -float("inf")], pd.NA)
datos = datos.dropna(
    subset=["voice", "lambda_observed",
            "lambda_null_mean", "lambda_ratio"]
)

datos = datos[datos["lambda_null_mean"] > 0]

# Etiquetas en español
datos["voz"] = datos["voice"].map(nombres_voces)

# ============================================================
# Graficar las distribuciones
# ============================================================

sns.set_theme(style="ticks", context="talk")

fig, ax = plt.subplots(figsize=(10, 6))

for voz in orden_voces:
    valores = datos.loc[
        datos["voice"] == voz,
        "lambda_ratio"
    ]

    sns.kdeplot(
        x=valores,
        label=nombres_voces[voz],
        fill=False,
        linewidth=2.2,
        common_norm=False,
        ax=ax
    )

# Valor en el que el decaimiento observado y el nulo son iguales
ax.axvline(
    x=1,
    linestyle="--",
    linewidth=1.7,
    color="black",
    label=r"$\lambda_{\mathrm{obs}}=\lambda_{\mathrm{null}}$"
)

ax.set_xlabel(
    r"$\lambda_{\mathrm{obs}}/\langle\lambda_{\mathrm{null}}\rangle$"
)
ax.set_ylabel("Densidad")
ax.set_title(
    "Distribución del decaimiento observado relativo al modelo nulo"
)

ax.legend(
    title="Voz",
    frameon=False
)

sns.despine()
fig.tight_layout()

# plt.savefig(
#     "distribucion_lambda_obs_sobre_lambda_null_por_voz.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()