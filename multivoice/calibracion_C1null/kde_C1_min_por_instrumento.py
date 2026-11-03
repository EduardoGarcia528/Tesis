import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# KDE robusta de C1_min por instrumento
# ============================================================

CSV_FILE = "calibracion_C1null_por_voz.csv"
OUTPUT_FILE = "kde_C1_min_por_instrumento.png"

VOICE_ORDER = ["vln I", "vln II", "viola", "cello"]
VOICE_LABELS = {
    "vln I": "Violin I",
    "vln II": "Violin II",
    "viola": "Viola",
    "cello": "Cello",
}

# Percentiles usados SOLO para estimar la KDE.
# Los valores fuera de este rango se consideran eventos extremos
# y no influyen en la forma suavizada de la distribución.
LOW_Q = 0.000
HIGH_Q = 0.99

# Factor de ancho de banda de scipy.stats.gaussian_kde.
# "scott" suele ser una buena elección automática.
BW_METHOD = "scott"

# Número de puntos para evaluar cada kernel.
N_GRID = 800


def robust_values(values, low_q=LOW_Q, high_q=HIGH_Q):
    """Recorta únicamente las colas extremas para estimar la KDE."""
    q_low, q_high = np.quantile(values, [low_q, high_q])
    central = values[(values >= q_low) & (values <= q_high)]
    return central, q_low, q_high


def main():
    df = pd.read_csv(CSV_FILE)

    required_columns = {"voice", "C1_min"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en el CSV: {sorted(missing)}"
        )

    data = df.loc[
        df["voice"].isin(VOICE_ORDER),
        ["voice", "C1_min"]
    ].copy()

    data["C1_min"] = pd.to_numeric(data["C1_min"], errors="coerce")
    data = data.dropna(subset=["C1_min"])

    if data.empty:
        raise ValueError("No hay valores válidos de C1_min para graficar.")

    # --------------------------------------------------------
    # Preparar datos robustos y rango común de evaluación
    # --------------------------------------------------------
    robust_data = {}
    medians = {}

    for voice in VOICE_ORDER:
        values = data.loc[data["voice"] == voice, "C1_min"].to_numpy()

        if len(values) < 2:
            continue

        # La mediana se calcula con TODOS los datos.
        medians[voice] = np.mean(values)

        # Solo la KDE ignora las colas extremas.
        central, q_low, q_high = robust_values(values)
        robust_data[voice] = central

    if not robust_data:
        raise ValueError("No hay suficientes datos para calcular las KDE.")

    # Rango común basado en los datos centrales de las cuatro voces.
    x_min = min(values.min() for values in robust_data.values())
    x_max = max(values.max() for values in robust_data.values())

    padding = 0.04 * (x_max - x_min)
    if padding == 0:
        padding = 1e-6

    x_grid = np.linspace(x_min - padding, x_max + padding, N_GRID)

    # --------------------------------------------------------
    # Gráfica
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.5))

    for voice in VOICE_ORDER:
        if voice not in robust_data:
            continue

        central = robust_data[voice]

        # gaussian_kde requiere varianza distinta de cero.
        if np.allclose(central, central[0]):
            continue

        kde = gaussian_kde(central, bw_method=BW_METHOD)
        density = kde(x_grid)

        line, = ax.plot(
            x_grid,
            density,
            linewidth=2.2,
            label=VOICE_LABELS[voice],
        )

        # Mediana de la distribución COMPLETA.
        median = medians[voice]

        # Altura de la KDE en la mediana, para dibujar una línea corta
        # en vez de una línea vertical que cruce toda la figura.
        y_median = float(kde([median])[0])

   


    ax.set_xlabel(r"$C_{1,\min}$", fontsize=13)
    ax.set_ylabel("Probability density", fontsize=13)

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.legend(
        frameon=False,
        fontsize=10,
        title="Instrument",
        title_fontsize=10,
    )

    ax.grid(axis="y", alpha=0.20)
    ax.margins(x=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
