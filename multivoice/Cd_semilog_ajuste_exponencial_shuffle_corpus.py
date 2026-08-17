# -*- coding: utf-8 -*-
"""
Análisis semilogarítmico de las integrales de correlación C_d.

Para cada movimiento del corpus:
    - crea una figura 2x2, una subgráfica por voz;
    - calcula C_d observado sobre intervalos melódicos;
    - ajusta un decaimiento exponencial entre d=1 y d=8;
    - calcula un nulo por shuffle;
    - muestra la curva central del nulo y su banda de dispersión;
    - ajusta también un decaimiento exponencial al nulo;
    - guarda parámetros y diagnósticos en un CSV.

Modelo ajustado:
    log(C_d) = log(A) - lambda * d
    C_d = A * exp(-lambda * d)

La escala característica de decaimiento es:
    xi = 1 / lambda,
siempre que lambda > 0.

IMPORTANTE:
    C_0 = 1 no se ajusta porque es un valor trivial fijado por definición.
    El ajuste se realiza sobre d=1,...,8 y el R² se calcula en el espacio
    transformado log(C_d) frente a d.

Requiere:
    numpy
    pandas
    matplotlib
    mi_libreria
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mi_libreria as ml


# ============================================================
# CONFIGURACIÓN
# ============================================================

CSV_PATH = Path("quartet_subcorpus_all_movements.csv")
OUTPUT_DIR = Path("figuras_Cd_semilog_exponencial_shuffle")
SUMMARY_CSV = OUTPUT_DIR / "resumen_ajuste_exponencial_Cd_observado_y_shuffle.csv"

# Se trabaja sobre intervalos melódicos:
# Delta p_n = p_{n+1} - p_n
ANALYSIS_MODE = "intervals" # "midi" o "intervals"

MAX_GAMMA = 7
MU = 2

# C = [C_0, C_1, ..., C_10].
# C_0=1 se excluye por ser trivial; se ajustan d=1,...,8.
FIT_D_MIN = 1
FIT_D_MAX = 8

N_SHUFFLES = 200
LOW_PERCENTILE = 0.0
HIGH_PERCENTILE = 100.0
RNG_SEED = 12345

SAVE_PNG = True
SAVE_PDF = False
PNG_DPI = 220

VOICE_ORDER = ["vln I", "vln II", "viola", "cello"]

VOICE_TITLES = {
    "vln I": "Violín I",
    "vln II": "Violín II",
    "viola": "Viola",
    "cello": "Violonchelo",
}

VOICE_COLORS = {
    "vln I": "#1f77b4",
    "vln II": "#ff7f0e",
    "viola": "#2ca02c",
    "cello": "#d62728",
}

NULL_COLOR = "#5b5b5b"

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 13,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
    }
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def slugify(value: Any) -> str:
    """Convierte un texto en un nombre seguro para archivos."""
    text = str(value).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s/]+", "_", text)
    return text.strip("_")


def parse_literal(value: Any) -> Any:
    """Interpreta listas almacenadas como texto en el CSV."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_voices(row: pd.Series) -> dict[str, np.ndarray]:
    """
    Lee voices_pitches y roles, y devuelve un diccionario:
        nombre de voz -> secuencia MIDI
    """
    voices = parse_literal(row["voices_pitches"])

    if not isinstance(voices, (list, tuple)):
        raise TypeError("voices_pitches no contiene una lista de voces.")

    roles_value = row.get("roles", "vln I|vln II|viola|cello")
    roles = [str(role).strip() for role in str(roles_value).split("|")]

    if len(voices) != len(roles):
        raise ValueError(
            f"Número de voces ({len(voices)}) distinto del número de roles "
            f"({len(roles)})."
        )

    voice_map: dict[str, np.ndarray] = {}

    for role, pitches in zip(roles, voices):
        arr = np.asarray(pitches, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        voice_map[role] = arr

    missing = [voice for voice in VOICE_ORDER if voice not in voice_map]
    if missing:
        raise ValueError(f"Faltan voces esperadas: {missing}")

    return {voice: voice_map[voice] for voice in VOICE_ORDER}


def preprocess_sequence(pitches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convierte alturas MIDI en intervalos melódicos."""
    pitches = np.asarray(pitches, dtype=float).ravel()
    pitches = pitches[np.isfinite(pitches)]
    return pitches, np.diff(pitches)


def compute_C(array: np.ndarray) -> np.ndarray:
    """
    Calcula C = [C_0, ..., C_8] con max_gamma=7.
    """
    C, _ = ml.gamma_index_rank_ties(
        np.asarray(array, dtype=float),
        max_gamma=MAX_GAMMA,
        mu=MU,
    )

    C = np.asarray(C, dtype=float).ravel()

    expected_length = FIT_D_MAX + 1
    if len(C) < expected_length:
        raise ValueError(
            f"Se esperaban al menos {expected_length} valores de C, "
            f"pero se obtuvieron {len(C)}."
        )

    return C[:expected_length]


def fit_exponential_decay(
    C: np.ndarray,
    d_min: int = FIT_D_MIN,
    d_max: int = FIT_D_MAX,
) -> dict[str, Any]:
    """
    Ajusta un decaimiento exponencial:

        C_d = A * exp(-lambda_decay * d)

    equivalente a:

        log(C_d) = log(A) - lambda_decay * d.

    Devuelve lambda_decay, pendiente, intercepto, A, la escala
    característica xi=1/lambda_decay, R², RMSE logarítmico y curva ajustada.

    Tanto R² como RMSE se calculan en el espacio transformado log(C_d).
    """
    C = np.asarray(C, dtype=float).ravel()
    d_all = np.arange(len(C), dtype=float)

    mask = (
        (d_all >= d_min)
        & (d_all <= d_max)
        & np.isfinite(C)
        & (C > 0)
    )

    d = d_all[mask]
    C_valid = C[mask]
    d_fit = np.arange(d_min, d_max + 1, dtype=float)

    if len(d) < 2:
        return {
            "lambda_decay": np.nan,
            "decay_length": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "A": np.nan,
            "r_squared": np.nan,
            "rmse_log": np.nan,
            "d_fit": d_fit,
            "C_fit": np.full(d_fit.shape, np.nan, dtype=float),
            "n_fit": len(d),
        }

    x = d
    y = np.log(C_valid)

    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = intercept + slope * x

    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_log = float(np.sqrt(np.mean(residuals**2)))

    lambda_decay = -float(slope)
    A = float(np.exp(intercept))
    decay_length = (
        float(1.0 / lambda_decay)
        if np.isfinite(lambda_decay) and lambda_decay > 0
        else np.nan
    )

    C_fit = A * np.exp(-lambda_decay * d_fit)

    return {
        "lambda_decay": lambda_decay,
        "decay_length": decay_length,
        "slope": float(slope),
        "intercept": float(intercept),
        "A": A,
        "r_squared": r_squared,
        "rmse_log": rmse_log,
        "d_fit": d_fit,
        "C_fit": C_fit,
        "n_fit": len(d),
    }


def geometric_mean_positive(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Media geométrica ignorando valores no positivos o no finitos.

    Es una curva central natural cuando el ajuste se hace en log(C).
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0)

    log_values = np.where(valid, np.log(values), np.nan)

    with np.errstate(invalid="ignore"):
        result = np.exp(np.nanmean(log_values, axis=axis))

    return result


def compute_shuffle_statistics(
    array: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Baraja la secuencia N_SHUFFLES veces.

    Devuelve:
        - todas las curvas C_d del nulo;
        - curva central: media geométrica;
        - percentiles punto a punto;
        - ajuste semilogarítmico de la curva central;
        - distribución de tasas lambda de cada shuffle.
    """

    array = np.asarray(array, dtype=float).ravel()
    C_shuffles = np.empty((N_SHUFFLES, FIT_D_MAX + 1), dtype=float)

    lambda_values = np.full(N_SHUFFLES, np.nan, dtype=float)
    r2_values = np.full(N_SHUFFLES, np.nan, dtype=float)
    # iaafts = ml.iaaft(array, N_SHUFFLES)
    for iteration in range(N_SHUFFLES):
        # shuffled = rng.permutation(array)
        # shuffled = iaafts[iteration]
        if ANALYSIS_MODE == "intervals":
            # Nulo shuffle(Delta X): primero se calculan los intervalos
            # y después se destruye su orden temporal.
            shuffled = rng.permutation(np.abs(np.diff(array)))
            # shuffled = np.diff(rng.permutation(array))
        elif ANALYSIS_MODE == "midi":
            shuffled = rng.permutation(array)
        else:
            raise ValueError(
                "ANALYSIS_MODE debe ser 'intervals' o 'midi'."
            )

        C_null = compute_C(shuffled)
        C_shuffles[iteration] = C_null

        fit_null = fit_exponential_decay(C_null)
        lambda_values[iteration] = fit_null["lambda_decay"]
        r2_values[iteration] = fit_null["r_squared"]

    C_center = geometric_mean_positive(C_shuffles, axis=0)
    C_low = np.nanpercentile(C_shuffles, LOW_PERCENTILE, axis=0)
    C_high = np.nanpercentile(C_shuffles, HIGH_PERCENTILE, axis=0)

    fit_center = fit_exponential_decay(C_center)

    return {
        "C_all": C_shuffles,
        "C_center": C_center,
        "C_low": C_low,
        "C_high": C_high,
        "fit_center": fit_center,
        "lambda_mean": float(np.nanmean(lambda_values)),
        "lambda_std": float(np.nanstd(lambda_values, ddof=1)),
        "lambda_low": float(np.nanpercentile(lambda_values, LOW_PERCENTILE)),
        "lambda_high": float(np.nanpercentile(lambda_values, HIGH_PERCENTILE)),
        "r2_mean": float(np.nanmean(r2_values)),
    }


def positive_limits(values: list[float]) -> tuple[float, float]:
    """Límites comunes del eje vertical para los cuatro subplots."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]

    if arr.size == 0:
        return 1e-4, 1.0

    ymin = max(float(np.min(arr)) * 0.75, 1e-12)
    ymax = float(np.max(arr)) * 1.30

    return 0.01, 1.0


def add_summary_values(
    summary: dict[str, Any],
    prefix: str,
    values: np.ndarray,
) -> None:
    """Añade los valores C_0, ..., C_d disponibles al registro."""
    for d, value in enumerate(np.asarray(values, dtype=float)):
        summary[f"{prefix}_{d}"] = value


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus:\n{CSV_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(CSV_PATH)

    required_columns = {
        "composer",
        "work_title",
        "movement_no",
        "roles",
        "voices_pitches",
    }

    missing_columns = required_columns.difference(corpus.columns)
    if missing_columns:
        raise KeyError(
            f"Faltan columnas necesarias en el CSV: {sorted(missing_columns)}"
        )

    rng_master = np.random.default_rng(RNG_SEED)
    summary_rows: list[dict[str, Any]] = []

    total_movements = len(corpus)

    for row_index, row in corpus.iterrows():
        composer = str(row["composer_canonical"]) if (
            "composer_canonical" in corpus.columns
            and pd.notna(row["composer_canonical"])
        ) else str(row["composer"])

        work_title = str(row["work_title"])
        movement_no = row["movement_no"]

        print(
            f"[{row_index + 1:03d}/{total_movements:03d}] "
            f"{composer} | {work_title} | movimiento {movement_no}"
        )

        try:
            voices = parse_voices(row)
        except Exception as exc:
            print(f"    ERROR al leer voces: {exc}")
            continue

        movement_results: dict[str, dict[str, Any]] = {}

        for voice_index, voice in enumerate(VOICE_ORDER):
            midi, intervals = preprocess_sequence(voices[voice])
            if ANALYSIS_MODE == "intervals":
                sequence = np.abs(intervals)
            else:
                sequence = midi

            if sequence.size < FIT_D_MAX + 2:
                print(
                    f"    ADVERTENCIA: {voice} tiene solo "
                    f"{sequence.size} datos útiles."
                )
                continue

            try:
                C_observed = compute_C(sequence)
                fit_observed = fit_exponential_decay(C_observed)

                local_seed = int(rng_master.integers(0, 2**32 - 1))
                local_rng = np.random.default_rng(local_seed)

                null = compute_shuffle_statistics(midi, local_rng)

            except Exception as exc:
                print(f"    ERROR en {voice}: {exc}")
                continue

            movement_results[voice] = {
                "sequence": sequence,
                "C_observed": C_observed,
                "fit_observed": fit_observed,
                "null": null,
            }

            summary = {
                "composer": composer,
                "composer_id": row.get("composer", np.nan),
                "work_id": row.get("work_id", np.nan),
                "work_title": work_title,
                "catalogue": row.get("catalogue", np.nan),
                "period": row.get("period", np.nan),
                "movement_no": movement_no,
                "filename": row.get("filename", np.nan),
                "voice": voice,
                "analysis_mode": ANALYSIS_MODE,
                "n_notes": len(voices[voice]),
                "n_intervals": len(sequence),
                "fit_d_min": FIT_D_MIN,
                "fit_d_max": FIT_D_MAX,
                "n_shuffles": N_SHUFFLES,
                "lambda_observed": fit_observed["lambda_decay"],
                "decay_length_observed": fit_observed["decay_length"],
                "slope_semilog_observed": fit_observed["slope"],
                "intercept_semilog_observed": fit_observed["intercept"],
                "A_observed": fit_observed["A"],
                "r_squared_observed": fit_observed["r_squared"],
                "rmse_log_observed": fit_observed["rmse_log"],
                "lambda_null_mean": null["lambda_mean"],
                "lambda_null_std": null["lambda_std"],
                f"lambda_null_p{LOW_PERCENTILE:g}": null["lambda_low"],
                f"lambda_null_p{HIGH_PERCENTILE:g}": null["lambda_high"],
                "lambda_fit_null_center": null["fit_center"]["lambda_decay"],
                "decay_length_fit_null_center": null["fit_center"]["decay_length"],
                "r_squared_fit_null_center": null["fit_center"]["r_squared"],
                "r_squared_null_mean": null["r2_mean"],
            }

            add_summary_values(summary, "C_observed", C_observed)
            add_summary_values(summary, "C_null_geometric_mean", null["C_center"])
            add_summary_values(
                summary,
                f"C_null_p{LOW_PERCENTILE:g}",
                null["C_low"],
            )
            add_summary_values(
                summary,
                f"C_null_p{HIGH_PERCENTILE:g}",
                null["C_high"],
            )

            summary_rows.append(summary)

        if not movement_results:
            continue

        # ----------------------------------------------------
        # FIGURA 2 x 2
        # ----------------------------------------------------
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14.5, 10.0),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        all_positive_values: list[float] = []

        for voice in VOICE_ORDER:
            if voice not in movement_results:
                continue

            result = movement_results[voice]
            d = np.arange(len(result["C_observed"]))
            display_mask = (d >= FIT_D_MIN) & (d <= FIT_D_MAX)

            for curve in (
                result["C_observed"],
                result["null"]["C_center"],
                result["null"]["C_low"],
                result["null"]["C_high"],
            ):
                selected = np.asarray(curve)[display_mask]
                all_positive_values.extend(
                    selected[np.isfinite(selected) & (selected > 0)].tolist()
                )

        y_limits = positive_limits(all_positive_values)

        for ax, voice in zip(axes, VOICE_ORDER):
            if voice not in movement_results:
                ax.set_visible(False)
                continue

            result = movement_results[voice]

            C_observed = result["C_observed"]
            fit_observed = result["fit_observed"]
            null = result["null"]

            d = np.arange(len(C_observed), dtype=float)
            display_mask = (
                (d >= FIT_D_MIN)
                & (d <= FIT_D_MAX)
            )

            d_plot = d[display_mask]
            C_obs_plot = C_observed[display_mask]
            C_null_plot = null["C_center"][display_mask]
            C_low_plot = null["C_low"][display_mask]
            C_high_plot = null["C_high"][display_mask]

            valid_band = (
                np.isfinite(C_low_plot)
                & np.isfinite(C_high_plot)
                & (C_low_plot > 0)
                & (C_high_plot > 0)
            )

            # Banda de dispersión del nulo.
            ax.fill_between(
                d_plot[valid_band],
                C_low_plot[valid_band],
                C_high_plot[valid_band],
                color=NULL_COLOR,
                alpha=0.18,
                linewidth=0,
                label=(
                    f"Shuffle P{LOW_PERCENTILE:g}–P{HIGH_PERCENTILE:g}"
                    if voice == "vln I"
                    else None
                ),
                zorder=1,
            )

            # Puntos observados.
            valid_obs = np.isfinite(C_obs_plot) & (C_obs_plot > 0)
            ax.semilogy(
                d_plot[valid_obs],
                C_obs_plot[valid_obs],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=VOICE_COLORS[voice],
                markeredgecolor="white",
                markeredgewidth=0.6,
                color=VOICE_COLORS[voice],
                label=r"$C_d$ observado" if voice == "vln I" else None,
                zorder=4,
            )

            # Ajuste exponencial observado.
            ax.semilogy(
                fit_observed["d_fit"],
                fit_observed["C_fit"],
                linestyle="-",
                linewidth=2.0,
                color=VOICE_COLORS[voice],
                label="Ajuste observado" if voice == "vln I" else None,
                zorder=3,
            )

            # Curva central del nulo.
            valid_null = np.isfinite(C_null_plot) & (C_null_plot > 0)
            ax.semilogy(
                d_plot[valid_null],
                C_null_plot[valid_null],
                linestyle="none",
                marker="s",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=NULL_COLOR,
                markeredgewidth=1.1,
                color=NULL_COLOR,
                label="Shuffle: media geométrica" if voice == "vln I" else None,
                zorder=4,
            )

            # Ajuste exponencial del nulo.
            fit_null = null["fit_center"]
            ax.semilogy(
                fit_null["d_fit"],
                fit_null["C_fit"],
                linestyle="--",
                linewidth=1.9,
                color=NULL_COLOR,
                label="Ajuste del shuffle" if voice == "vln I" else None,
                zorder=3,
            )

            ax.set_title(VOICE_TITLES[voice])
            ax.set_xlim(0.7, 8.3)
            ax.set_ylim(*y_limits)
            # ax.set_ylim(0.1, 1.0)

            ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

            ax.grid(True, which="major", alpha=0.26)
            ax.grid(True, which="minor", alpha=0.10)

            annotation = (
                rf"$\lambda_{{obs}}={fit_observed['lambda_decay']:.3f}$"
                "\n"
                rf"$\xi_{{obs}}={fit_observed['decay_length']:.3f}$"
                "\n"
                rf"$R^2_{{obs}}={fit_observed['r_squared']:.3f}$"
                "\n"
                rf"$\lambda_{{null}}="
                rf"{null['lambda_mean']:.3f}\pm{null['lambda_std']:.3f}$"
            )

            ax.text(
                0.03,
                0.05,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.3,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.90,
                },
            )

        fig.supxlabel(r"Longitud del fragmento $d$")
        fig.supylabel(r"Integral de correlación $C_d$")

        fig.suptitle(
            f"{composer} | {work_title} | Movimiento {movement_no}\n"
            rf"Ajuste semilog: $C_d=A\,e^{{-\lambda d}}$ sobre {ANALYSIS_MODE}",
            y=0.985,
        )

        visible_axes = [ax for ax in axes if ax.get_visible()]
        if visible_axes:
            handles, labels = visible_axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                ncol=4,
                frameon=False,
            )

        fig.tight_layout(rect=(0.035, 0.045, 0.99, 0.89))

        composer_directory = OUTPUT_DIR / slugify(composer)
        composer_directory.mkdir(parents=True, exist_ok=True)

        file_stem = (
            f"{slugify(row.get('work_id', work_title))}"
            f"__mov_{movement_no}"
            f"__Cd_semilog_exponencial_shuffle"
        )

        if SAVE_PNG:
            fig.savefig(
                composer_directory / f"{file_stem}.png",
                dpi=PNG_DPI,
                bbox_inches="tight",
            )

        if SAVE_PDF:
            fig.savefig(
                composer_directory / f"{file_stem}.pdf",
                bbox_inches="tight",
            )

        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nProceso terminado.")
    print(f"Figuras: {OUTPUT_DIR.resolve()}")
    print(f"Resumen: {SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
