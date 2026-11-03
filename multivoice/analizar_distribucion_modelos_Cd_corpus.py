# -*- coding: utf-8 -*-
"""
Distribución de caídas exponenciales y power-law de C_d en el corpus.

Este script analiza los CSV producidos por Cd_bestfit_C1null_corpus.py.
La clasificación principal usa:

    best_model_observed

obtenida al comparar, para cada voz, los ajustes

    exponencial: C_d = A exp(-lambda d)
    power-law:   C_d = A d^(-alpha)

mediante el menor RMSE en log(C_d).

Preguntas que responde
----------------------
1. ¿Qué fracción global del corpus se clasifica como exponencial o power-law?
2. ¿Hay compositores con mayor proporción de power-law?
3. ¿Cambia la proporción entre períodos/épocas?
4. ¿Cambia entre voces/instrumentos?
5. ¿Se concentra el comportamiento power-law en movimientos completos?
6. ¿Qué tan fuerte es la victoria de un modelo sobre el otro?

Además de la clasificación binaria, se usa

    Delta_RMSE = RMSE_exp - RMSE_power

por lo que:
    Delta_RMSE > 0  -> favorece power-law
    Delta_RMSE < 0  -> favorece exponencial
    Delta_RMSE = 0  -> empate numérico

Salidas
-------
Se crea un directorio con CSV resumen y figuras PNG.

CSV principales:
    resumen_global_modelos.csv
    resumen_por_compositor.csv
    resumen_por_periodo.csv
    resumen_por_voz.csv
    resumen_compositor_voz.csv
    resumen_por_movimiento.csv
    pruebas_asociacion_modelo.csv

Figuras principales:
    00_distribucion_global.png
    01_compositor_conteos.png
    02_compositor_fraccion_powerlaw.png
    03_periodo_fraccion_powerlaw.png
    04_voz_fraccion_powerlaw.png
    05_heatmap_compositor_voz_powerlaw.png
    06_powerlaws_por_movimiento.png
    07_delta_rmse_por_compositor.png
    08_delta_rmse_por_periodo.png
    09_delta_rmse_por_voz.png

Notas estadísticas
------------------
Las pruebas chi-cuadrada son exploratorias. Las cuatro voces de un mismo
movimiento no son observaciones estrictamente independientes; por ello el
p-valor no debe interpretarse como una prueba definitiva sin un modelo
jerárquico o una estrategia de remuestreo por pieza/movimiento.

Requiere:
    numpy
    pandas
    matplotlib
    scipy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path("figuras_C1null_shuffle_K0")

MODEL_SELECTION_CSV = (
    BASE_DIR / "resumen_seleccion_modelo_Cd_observado_y_shuffle_exponencial.csv"
)

# Se usa sólo como verificación de consistencia de filas/metadatos.
# La selección exponencial vs power-law está en MODEL_SELECTION_CSV.
LEGACY_CSV = (
    BASE_DIR / "resumen_ajuste_ley_potencia_Cd_observado_y_shuffle.csv"
)

OUTPUT_DIR = BASE_DIR / "distribucion_exponencial_vs_powerlaw"
FIGURE_DIR = OUTPUT_DIR / "figuras"

SAVE_PNG = True
PNG_DPI = 240

# Si True, el script intenta cotejar las filas del CSV nuevo con el legado.
VALIDATE_WITH_LEGACY_CSV = True

# Intervalo de confianza descriptivo para una proporción binomial.
CI_Z = 1.959963984540054  # 95 %

# Para rankings impresos en terminal. No elimina grupos de los CSV/figuras.
MIN_N_FOR_RANKING = 8

# Orden preferido de las voces. Las voces adicionales se agregan después.
VOICE_ORDER = ["vln I", "vln II", "viola", "cello"]

# Si se deja vacío, los períodos se ordenan según aparición en el CSV.
# Ejemplo de uso:
# PERIOD_ORDER = ["Baroque", "Classical", "Romantic", "Modern"]
PERIOD_ORDER: list[str] = []

# Mostrar etiquetas numéricas en barras y heatmaps.
ANNOTATE_PLOTS = True

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.titlesize": 14,
    }
)


# ============================================================
# UTILIDADES DE LECTURA Y LIMPIEZA
# ============================================================


def locate_csv(path: Path) -> Path:
    """Busca primero la ruta configurada y luego el mismo nombre en cwd."""
    candidates = [path, Path(path.name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró el CSV requerido. Se intentó:\n"
        + "\n".join(f"  - {p.resolve()}" for p in candidates)
    )


def normalize_model_label(value: Any) -> str | float:
    """Normaliza variantes textuales a 'exponential' o 'powerlaw'."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    compact = (
        text.replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("law", "law")
    )

    if compact in {"exponential", "exp", "exponencial"}:
        return "exponential"
    if compact in {"powerlaw", "power", "leypotencia", "leydepotencia"}:
        return "powerlaw"

    return text


def make_movement_key(df: pd.DataFrame) -> pd.Series:
    """Construye una clave robusta para identificar un movimiento."""
    def safe_col(name: str, default: str = "") -> pd.Series:
        if name in df.columns:
            return df[name].fillna(default).astype(str)
        return pd.Series(default, index=df.index, dtype=str)

    return (
        safe_col("composer")
        + "||"
        + safe_col("work_id")
        + "||"
        + safe_col("work_title")
        + "||mov="
        + safe_col("movement_no")
        + "||"
        + safe_col("filename")
    )


def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el CSV de selección y devuelve (válidas, descartadas)."""
    csv_path = locate_csv(MODEL_SELECTION_CSV)
    df = pd.read_csv(csv_path)

    required = {
        "composer",
        "period",
        "voice",
        "best_model_observed",
        "rmse_log_exponential_observed",
        "rmse_log_powerlaw_observed",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Faltan columnas necesarias en el CSV de selección: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["model"] = df["best_model_observed"].map(normalize_model_label)

    valid_model = df["model"].isin(["exponential", "powerlaw"])
    discarded = df.loc[~valid_model].copy()
    df = df.loc[valid_model].copy()

    # Recalcula Delta_RMSE para no depender de que la columna exista.
    df["delta_rmse"] = (
        pd.to_numeric(df["rmse_log_exponential_observed"], errors="coerce")
        - pd.to_numeric(df["rmse_log_powerlaw_observed"], errors="coerce")
    )

    df["is_powerlaw"] = (df["model"] == "powerlaw").astype(int)
    df["is_exponential"] = (df["model"] == "exponential").astype(int)
    df["movement_key"] = make_movement_key(df)

    # Strings limpios para agrupación.
    for col in ["composer", "period", "voice"]:
        df[col] = df[col].fillna("Desconocido").astype(str).str.strip()
        df.loc[df[col].eq(""), col] = "Desconocido"

    if discarded.shape[0] > 0:
        warnings.warn(
            f"Se descartaron {len(discarded)} filas porque best_model_observed "
            "no era exponential ni powerlaw."
        )

    return df, discarded


# ============================================================
# VALIDACIÓN CONTRA CSV LEGADO
# ============================================================


def validate_against_legacy(df: pd.DataFrame) -> dict[str, Any]:
    """Coteja identidad de filas con el CSV legado si está disponible."""
    result: dict[str, Any] = {
        "legacy_found": False,
        "n_model_selection": len(df),
        "n_legacy": np.nan,
        "n_keys_in_both": np.nan,
        "n_only_model_selection": np.nan,
        "n_only_legacy": np.nan,
    }

    if not VALIDATE_WITH_LEGACY_CSV:
        return result

    candidates = [LEGACY_CSV, Path(LEGACY_CSV.name)]
    legacy_path = next((p for p in candidates if p.exists()), None)
    if legacy_path is None:
        warnings.warn(
            "No se encontró el CSV legado. El análisis continúa porque no es "
            "necesario para clasificar los modelos."
        )
        return result

    legacy = pd.read_csv(legacy_path)
    result["legacy_found"] = True
    result["n_legacy"] = len(legacy)

    key_cols = [
        c
        for c in ["composer", "work_id", "work_title", "movement_no", "voice"]
        if c in df.columns and c in legacy.columns
    ]

    if not key_cols:
        warnings.warn("No hay columnas comunes suficientes para cotejar ambos CSV.")
        return result

    left = df[key_cols].fillna("").astype(str).drop_duplicates()
    right = legacy[key_cols].fillna("").astype(str).drop_duplicates()

    merged = left.merge(right, on=key_cols, how="outer", indicator=True)
    counts = merged["_merge"].value_counts()

    result["n_keys_in_both"] = int(counts.get("both", 0))
    result["n_only_model_selection"] = int(counts.get("left_only", 0))
    result["n_only_legacy"] = int(counts.get("right_only", 0))

    return result


# ============================================================
# RESÚMENES
# ============================================================


def wilson_interval(k: int, n: int, z: float = CI_Z) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción binomial."""
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    den = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / den
    half = (
        z
        * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))
        / den
    )
    return max(0.0, center - half), min(1.0, center + half)


def aggregate_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Resumen por una categoría: conteos, fracciones, IC y Delta_RMSE."""
    rows: list[dict[str, Any]] = []

    for group, sub in df.groupby(group_col, dropna=False, sort=False):
        n = len(sub)
        n_power = int(sub["is_powerlaw"].sum())
        n_exp = int(sub["is_exponential"].sum())
        low, high = wilson_interval(n_power, n)

        rows.append(
            {
                group_col: group,
                "n_total": n,
                "n_exponential": n_exp,
                "n_powerlaw": n_power,
                "fraction_exponential": n_exp / n if n else np.nan,
                "fraction_powerlaw": n_power / n if n else np.nan,
                "percent_powerlaw": 100.0 * n_power / n if n else np.nan,
                "powerlaw_ci95_low": low,
                "powerlaw_ci95_high": high,
                "powerlaw_ci95_low_percent": 100.0 * low,
                "powerlaw_ci95_high_percent": 100.0 * high,
                "mean_delta_rmse": sub["delta_rmse"].mean(),
                "median_delta_rmse": sub["delta_rmse"].median(),
                "std_delta_rmse": sub["delta_rmse"].std(ddof=1),
                "mean_abs_delta_rmse": sub["delta_rmse"].abs().mean(),
                "n_movements": sub["movement_key"].nunique(),
                "n_works": (
                    sub["work_id"].nunique(dropna=True)
                    if "work_id" in sub.columns
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def global_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila con la composición global del corpus analizado."""
    n = len(df)
    n_power = int(df["is_powerlaw"].sum())
    n_exp = int(df["is_exponential"].sum())
    low, high = wilson_interval(n_power, n)

    return pd.DataFrame(
        [
            {
                "n_total": n,
                "n_exponential": n_exp,
                "n_powerlaw": n_power,
                "fraction_exponential": n_exp / n if n else np.nan,
                "fraction_powerlaw": n_power / n if n else np.nan,
                "percent_powerlaw": 100.0 * n_power / n if n else np.nan,
                "powerlaw_ci95_low": low,
                "powerlaw_ci95_high": high,
                "mean_delta_rmse": df["delta_rmse"].mean(),
                "median_delta_rmse": df["delta_rmse"].median(),
                "n_composers": df["composer"].nunique(),
                "n_periods": df["period"].nunique(),
                "n_voices": df["voice"].nunique(),
                "n_movements": df["movement_key"].nunique(),
                "n_works": (
                    df["work_id"].nunique(dropna=True)
                    if "work_id" in df.columns
                    else np.nan
                ),
            }
        ]
    )


def movement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Número de voces power-law por movimiento."""
    metadata_cols = [
        c
        for c in [
            "composer",
            "composer_id",
            "period",
            "work_id",
            "work_title",
            "catalogue",
            "movement_no",
            "filename",
            "analysis_mode",
        ]
        if c in df.columns
    ]

    rows: list[dict[str, Any]] = []

    for key, sub in df.groupby("movement_key", sort=False):
        n = len(sub)
        n_power = int(sub["is_powerlaw"].sum())
        n_exp = int(sub["is_exponential"].sum())

        if n_power > n_exp:
            majority = "powerlaw"
        elif n_exp > n_power:
            majority = "exponential"
        else:
            majority = "tie"

        row: dict[str, Any] = {"movement_key": key}
        for col in metadata_cols:
            row[col] = sub.iloc[0][col]

        row.update(
            {
                "n_voices_analyzed": n,
                "n_exponential": n_exp,
                "n_powerlaw": n_power,
                "fraction_powerlaw": n_power / n if n else np.nan,
                "majority_model": majority,
                "mean_delta_rmse": sub["delta_rmse"].mean(),
                "median_delta_rmse": sub["delta_rmse"].median(),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def composer_voice_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz larga compositor x voz con proporción power-law."""
    rows: list[dict[str, Any]] = []
    for (composer, voice), sub in df.groupby(["composer", "voice"], sort=False):
        n = len(sub)
        n_power = int(sub["is_powerlaw"].sum())
        low, high = wilson_interval(n_power, n)
        rows.append(
            {
                "composer": composer,
                "voice": voice,
                "n_total": n,
                "n_powerlaw": n_power,
                "n_exponential": n - n_power,
                "fraction_powerlaw": n_power / n if n else np.nan,
                "percent_powerlaw": 100.0 * n_power / n if n else np.nan,
                "powerlaw_ci95_low": low,
                "powerlaw_ci95_high": high,
                "mean_delta_rmse": sub["delta_rmse"].mean(),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# PRUEBAS DE ASOCIACIÓN
# ============================================================


def cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    """V de Cramér sin corrección de sesgo."""
    r, k = shape
    denom = n * min(r - 1, k - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


def association_test(df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    """Chi-cuadrada entre grupo y tipo de modelo."""
    table = pd.crosstab(df[group_col], df["model"])

    # Fuerza presencia de ambas columnas para consistencia.
    for model in ["exponential", "powerlaw"]:
        if model not in table.columns:
            table[model] = 0
    table = table[["exponential", "powerlaw"]]

    # Elimina categorías sin observaciones tras cualquier filtrado.
    table = table.loc[table.sum(axis=1) > 0]

    if table.shape[0] < 2 or np.any(table.sum(axis=0).to_numpy() == 0):
        return {
            "variable": group_col,
            "n": int(table.to_numpy().sum()),
            "n_categories": int(table.shape[0]),
            "chi2": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "min_expected_count": np.nan,
            "n_expected_cells_lt5": np.nan,
            "note": "Prueba no definida: faltan categorías/modelos con observaciones.",
        }

    chi2, p, dof, expected = chi2_contingency(table.to_numpy(), correction=False)
    n = int(table.to_numpy().sum())
    min_expected = float(np.min(expected))
    n_lt5 = int(np.sum(expected < 5.0))

    note = "Exploratoria; las voces dentro de un movimiento no son independientes."
    if n_lt5 > 0:
        note += " Hay celdas con frecuencia esperada < 5."

    return {
        "variable": group_col,
        "n": n,
        "n_categories": int(table.shape[0]),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": cramers_v(float(chi2), n, table.shape),
        "min_expected_count": min_expected,
        "n_expected_cells_lt5": n_lt5,
        "note": note,
    }


# ============================================================
# ORDEN DE CATEGORÍAS
# ============================================================


def ordered_unique(values: Iterable[Any]) -> list[str]:
    """Únicos preservando orden de aparición."""
    return list(dict.fromkeys(str(v) for v in values))


def voice_order(df: pd.DataFrame) -> list[str]:
    present = ordered_unique(df["voice"])
    preferred = [v for v in VOICE_ORDER if v in present]
    extras = [v for v in present if v not in preferred]
    return preferred + extras


def period_order(df: pd.DataFrame) -> list[str]:
    present = ordered_unique(df["period"])
    if not PERIOD_ORDER:
        return present
    preferred = [p for p in PERIOD_ORDER if p in present]
    extras = [p for p in present if p not in preferred]
    return preferred + extras


# ============================================================
# FUNCIONES DE GRÁFICA
# ============================================================


def save_figure(fig: plt.Figure, filename: str) -> None:
    if SAVE_PNG:
        fig.savefig(FIGURE_DIR / filename, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def annotate_bar_values(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    if not ANNOTATE_PLOTS:
        return
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        if not np.isfinite(width) or not np.isfinite(height):
            continue
        if width > 0:
            ax.text(
                patch.get_x() + width,
                patch.get_y() + height / 2,
                "  " + fmt.format(width),
                va="center",
                ha="left",
                fontsize=8.5,
            )


def plot_global(df: pd.DataFrame) -> None:
    counts = df["model"].value_counts().reindex(["exponential", "powerlaw"], fill_value=0)
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    labels = ["Exponencial", "Power law"]
    values = counts.to_numpy(dtype=float)
    bars = ax.bar(labels, values)

    if ANNOTATE_PLOTS:
        for bar, count in zip(bars, values):
            pct = 100.0 * count / total if total else np.nan
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count)}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_ylabel("Número de voces")
    ax.set_title("Distribución global del mejor modelo para $C_d$")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "00_distribucion_global.png")


def plot_group_counts(summary: pd.DataFrame, group_col: str, filename: str, title: str) -> None:
    data = summary.sort_values("n_powerlaw", ascending=True).copy()
    y = np.arange(len(data))

    fig_height = max(5.0, 0.38 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))

    ax.barh(y, data["n_exponential"], label="Exponencial")
    ax.barh(
        y,
        data["n_powerlaw"],
        left=data["n_exponential"],
        label="Power law",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data[group_col])
    ax.set_xlabel("Número de voces")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.22)

    if ANNOTATE_PLOTS:
        for yi, (_, row) in enumerate(data.iterrows()):
            total = int(row["n_total"])
            pct = float(row["percent_powerlaw"])
            ax.text(
                total,
                yi,
                f"  n={total}, PL={pct:.1f}%",
                va="center",
                ha="left",
                fontsize=8.2,
            )
        xmax = max(1.0, float(data["n_total"].max()))
        ax.set_xlim(0, xmax * 1.28)

    fig.tight_layout()
    save_figure(fig, filename)


def plot_fraction_powerlaw(
    summary: pd.DataFrame,
    group_col: str,
    filename: str,
    title: str,
    fixed_order: list[str] | None = None,
) -> None:
    data = summary.copy()

    if fixed_order is None:
        data = data.sort_values("fraction_powerlaw", ascending=True)
    else:
        order_map = {name: i for i, name in enumerate(fixed_order)}
        data["_order"] = data[group_col].map(order_map).fillna(len(order_map))
        data = data.sort_values("_order", ascending=False)

    y = np.arange(len(data))
    p = data["fraction_powerlaw"].to_numpy(dtype=float)
    lo = data["powerlaw_ci95_low"].to_numpy(dtype=float)
    hi = data["powerlaw_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([p - lo, hi - p])

    fig_height = max(4.6, 0.42 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.errorbar(
        p,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        linewidth=1.3,
        markersize=5.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data[group_col])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Fracción clasificada como power law")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    if ANNOTATE_PLOTS:
        for yi, (_, row) in enumerate(data.iterrows()):
            ax.text(
                min(1.0, float(row["powerlaw_ci95_high"])) + 0.012,
                yi,
                f"n={int(row['n_total'])}",
                va="center",
                ha="left",
                fontsize=8.2,
            )

    fig.tight_layout()
    save_figure(fig, filename)


def plot_composer_voice_heatmap(cv: pd.DataFrame, df: pd.DataFrame) -> None:
    composers = (
        aggregate_by(df, "composer")
        .sort_values("fraction_powerlaw", ascending=False)["composer"]
        .tolist()
    )
    voices = voice_order(df)

    pivot = cv.pivot(index="composer", columns="voice", values="fraction_powerlaw")
    pivot = pivot.reindex(index=composers, columns=voices)

    matrix = pivot.to_numpy(dtype=float)

    fig_height = max(5.2, 0.42 * len(composers) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(voices)))
    ax.set_xticklabels(voices)
    ax.set_yticks(np.arange(len(composers)))
    ax.set_yticklabels(composers)
    ax.set_title("Fracción power law por compositor y voz")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Fracción power law")

    if ANNOTATE_PLOTS:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "05_heatmap_compositor_voz_powerlaw.png")


def plot_powerlaws_per_movement(mov: pd.DataFrame) -> None:
    if mov.empty:
        return

    max_voices = int(mov["n_voices_analyzed"].max())
    # Para el corpus de cuartetos normalmente será 4, pero se hace genérico.
    possible = np.arange(max_voices + 1)
    counts = mov["n_powerlaw"].value_counts().reindex(possible, fill_value=0)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bars = ax.bar(possible, counts.to_numpy())
    ax.set_xticks(possible)
    ax.set_xlabel("Número de voces power law en el movimiento")
    ax.set_ylabel("Número de movimientos")
    ax.set_title("Agrupamiento de caídas power law dentro de movimientos")
    ax.grid(axis="y", alpha=0.24)

    if ANNOTATE_PLOTS:
        for bar, count in zip(bars, counts.to_numpy()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    save_figure(fig, "06_powerlaws_por_movimiento.png")


def plot_delta_rmse_boxplots(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    filename: str,
    title: str,
) -> None:
    groups = [g for g in groups if g in set(df[group_col])]
    if not groups:
        return

    values = [
        df.loc[df[group_col] == group, "delta_rmse"].dropna().to_numpy()
        for group in groups
    ]

    nonempty = [(g, v) for g, v in zip(groups, values) if len(v) > 0]
    if not nonempty:
        return

    groups, values = map(list, zip(*nonempty))

    fig_height = max(4.8, 0.42 * len(groups) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.boxplot(values, vert=False, showfliers=False)
    ax.set_yticks(np.arange(1, len(groups) + 1))
    ax.set_yticklabels(groups)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$\Delta RMSE = RMSE_{exp}-RMSE_{power}$")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    ax.text(
        0.01,
        1.01,
        "< 0 favorece exponencial     |     > 0 favorece power law",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    save_figure(fig, filename)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df, discarded = load_and_clean_data()

    if df.empty:
        raise ValueError("No quedaron filas válidas para analizar.")

    validation = validate_against_legacy(df)

    # --------------------------------------------------------
    # CSV resumen
    # --------------------------------------------------------
    global_df = global_summary(df)
    composer_df = aggregate_by(df, "composer")
    period_df = aggregate_by(df, "period")
    voice_df = aggregate_by(df, "voice")
    cv_df = composer_voice_summary(df)
    movement_df = movement_summary(df)

    tests_df = pd.DataFrame(
        [
            association_test(df, "composer"),
            association_test(df, "period"),
            association_test(df, "voice"),
        ]
    )

    validation_df = pd.DataFrame([validation])

    global_df.to_csv(OUTPUT_DIR / "resumen_global_modelos.csv", index=False)
    composer_df.to_csv(OUTPUT_DIR / "resumen_por_compositor.csv", index=False)
    period_df.to_csv(OUTPUT_DIR / "resumen_por_periodo.csv", index=False)
    voice_df.to_csv(OUTPUT_DIR / "resumen_por_voz.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "resumen_compositor_voz.csv", index=False)
    movement_df.to_csv(OUTPUT_DIR / "resumen_por_movimiento.csv", index=False)
    tests_df.to_csv(OUTPUT_DIR / "pruebas_asociacion_modelo.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validacion_csv_entrada.csv", index=False)

    if not discarded.empty:
        discarded.to_csv(OUTPUT_DIR / "filas_descartadas.csv", index=False)

    # También se guarda una copia limpia fila-a-fila con Delta_RMSE.
    df.to_csv(OUTPUT_DIR / "datos_clasificacion_limpios.csv", index=False)

    # --------------------------------------------------------
    # Figuras
    # --------------------------------------------------------
    plot_global(df)

    plot_group_counts(
        composer_df,
        "composer",
        "01_compositor_conteos.png",
        "Conteos de modelos por compositor",
    )

    plot_fraction_powerlaw(
        composer_df,
        "composer",
        "02_compositor_fraccion_powerlaw.png",
        "Fracción power law por compositor (IC 95% de Wilson)",
    )

    p_order = period_order(df)
    plot_fraction_powerlaw(
        period_df,
        "period",
        "03_periodo_fraccion_powerlaw.png",
        "Fracción power law por período/época",
        fixed_order=p_order,
    )

    v_order = voice_order(df)
    plot_fraction_powerlaw(
        voice_df,
        "voice",
        "04_voz_fraccion_powerlaw.png",
        "Fracción power law por voz/instrumento",
        fixed_order=v_order,
    )

    plot_composer_voice_heatmap(cv_df, df)
    plot_powerlaws_per_movement(movement_df)

    composer_order_delta = (
        composer_df.sort_values("median_delta_rmse", ascending=True)["composer"].tolist()
    )
    plot_delta_rmse_boxplots(
        df,
        "composer",
        composer_order_delta,
        "07_delta_rmse_por_compositor.png",
        r"Ventaja relativa del ajuste por compositor",
    )

    plot_delta_rmse_boxplots(
        df,
        "period",
        p_order,
        "08_delta_rmse_por_periodo.png",
        r"Ventaja relativa del ajuste por período/época",
    )

    plot_delta_rmse_boxplots(
        df,
        "voice",
        v_order,
        "09_delta_rmse_por_voz.png",
        r"Ventaja relativa del ajuste por voz/instrumento",
    )

    # --------------------------------------------------------
    # Resumen en terminal
    # --------------------------------------------------------
    g = global_df.iloc[0]
    print("\nProceso terminado.")
    print(f"Directorio de salida: {OUTPUT_DIR.resolve()}")
    print(
        "\nDistribución global: "
        f"exponencial={int(g['n_exponential'])} "
        f"({100*g['fraction_exponential']:.1f}%), "
        f"powerlaw={int(g['n_powerlaw'])} "
        f"({100*g['fraction_powerlaw']:.1f}%)."
    )

    eligible = composer_df.loc[composer_df["n_total"] >= MIN_N_FOR_RANKING].copy()
    if not eligible.empty:
        top = eligible.sort_values(
            ["fraction_powerlaw", "mean_delta_rmse"],
            ascending=[False, False],
        ).head(10)
        print(
            f"\nCompositores con mayor fracción power law "
            f"(n >= {MIN_N_FOR_RANKING}):"
        )
        for _, row in top.iterrows():
            print(
                f"  {row['composer']}: {100*row['fraction_powerlaw']:.1f}% "
                f"({int(row['n_powerlaw'])}/{int(row['n_total'])}), "
                f"mediana Delta_RMSE={row['median_delta_rmse']:.5g}"
            )

    print("\nPruebas chi-cuadrada exploratorias:")
    for _, row in tests_df.iterrows():
        print(
            f"  {row['variable']}: p={row['p_value']:.4g}, "
            f"V_Cramer={row['cramers_v']:.3f}"
            if np.isfinite(row["p_value"])
            else f"  {row['variable']}: prueba no definida"
        )

    print(
        "\nNota: para interpretar diferencias entre compositores/períodos/voces, "
        "conviene mirar conjuntamente la fracción power law, su tamaño muestral "
        "y la distribución de Delta_RMSE."
    )


if __name__ == "__main__":
    main()
