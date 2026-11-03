# -*- coding: utf-8 -*-
"""
Comparación exponencial vs ley de potencia para las integrales de correlación C_d.

Para cada movimiento del corpus:
    - crea una figura 2x2, una subgráfica por voz;
    - calcula C_d observado;
    - ajusta AL OBSERVADO simultáneamente:
          (1) exponencial: C_d = A exp(-lambda d)
          (2) ley de potencia: C_d = A d^(-alpha)
    - selecciona el mejor modelo observado por menor RMSE en log(C_d)
      (equivalentemente, mayor R^2, ya que ambos tienen dos parámetros y
      usan exactamente los mismos puntos);
    - al shuffle se le asigna SIEMPRE un ajuste exponencial para el análisis
      y para las figuras;
    - conserva el CSV legado
          resumen_ajuste_ley_potencia_Cd_observado_y_shuffle.csv
      con exactamente el mismo esquema que Cd_powerlaw_C1null_corpus.py;
    - genera además un segundo CSV con toda la información de comparación
      de modelos y el ajuste exponencial del shuffle.

NOTA DE COMPATIBILIDAD
---------------------
Para reproducir exactamente las columnas del CSV legado se siguen calculando
los diagnósticos de ley de potencia del shuffle que ese archivo contenía.
Esos diagnósticos NO se usan para elegir modelo ni para graficar el nulo.
El ajuste operativo del shuffle es siempre exponencial.

IMPORTANTE:
    C_0 = 1 no se ajusta porque es trivial por definición.
    Los ajustes se realizan sobre d=1,...,8.
    Ambos modelos observados se comparan en el mismo espacio log(C_d).

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

# Se conserva el directorio del script power-law para que el CSV legado
# permanezca en la misma ubicación esperada.
OUTPUT_DIR = Path("figuras_Cd_powerlaw_C1null_shuffle")

# CSV legado: mismo nombre y mismas columnas que Cd_powerlaw_C1null_corpus.py
SUMMARY_CSV = OUTPUT_DIR / "resumen_ajuste_ley_potencia_Cd_observado_y_shuffle.csv"

# CSV nuevo con la comparación exponencial vs power-law del observado y
# el ajuste exponencial del shuffle.
MODEL_SELECTION_CSV = (
    OUTPUT_DIR / "resumen_seleccion_modelo_Cd_observado_y_shuffle_exponencial.csv"
)

# "midi" o "intervals"
ANALYSIS_MODE = "intervals"

MAX_GAMMA = 7

# Probabilidad nula de cercanía objetivo común a todas las voces.
C1_NULL_TARGET =  0.5807

# C = [C_0, C_1, ..., C_8]. C_0=1 se excluye del ajuste.
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
        "legend.fontsize": 9.2,
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
    """Lee voices_pitches y roles y devuelve nombre de voz -> secuencia MIDI."""
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
    """Devuelve alturas MIDI e intervalos melódicos consecutivos."""
    pitches = np.asarray(pitches, dtype=float).ravel()
    pitches = pitches[np.isfinite(pitches)]
    return pitches, np.diff(pitches)


def exact_C1_null_from_counts(counts: np.ndarray, K: int) -> float:
    """
    Valor esperado exacto de C_{1,null} para un shuffle.

    Si n_r es la multiplicidad del rango r,

        C1_null(K) = [sum_r C(n_r,2)
                      + sum_{h=1}^K sum_r n_r n_{r+h}] / C(N,2).
    """
    counts = np.asarray(counts, dtype=np.int64).ravel()
    N = int(np.sum(counts))

    if N < 2:
        return np.nan

    q = len(counts)
    K = int(np.clip(K, 0, max(q - 1, 0)))

    close_pairs = np.sum(counts * (counts - 1) // 2, dtype=np.int64)

    for h in range(1, K + 1):
        close_pairs += np.sum(counts[:-h] * counts[h:], dtype=np.int64)

    total_pairs = N * (N - 1) / 2.0
    return float(close_pairs / total_pairs)


def determine_K_from_C1_null(
    array: np.ndarray,
    target: float = C1_NULL_TARGET,
) -> tuple[int, int, float]:
    """
    Elige el mayor K que todavía cumple C1_null(K) <= target.

    Si ni K=0 cumple la cota, se utiliza K=0.
    """
    array = np.asarray(array, dtype=float).ravel()
    array = array[np.isfinite(array)]

    if array.size < 2:
        raise ValueError("Se necesitan al menos dos datos para determinar K.")

    if not (0.0 <= target <= 1.0):
        raise ValueError("C1_NULL_TARGET debe estar entre 0 y 1.")

    _, counts = np.unique(array, return_counts=True)
    counts = counts.astype(np.int64)
    n_unique = int(len(counts))

    candidates = np.arange(n_unique, dtype=int)
    c1_values = np.array(
        [exact_C1_null_from_counts(counts, K) for K in candidates],
        dtype=float,
    )

    admissible = candidates[c1_values <= target]

    if admissible.size == 0:
        K = 0
    else:
        K = int(admissible[-1])

    return K, n_unique, float(c1_values[K])


def compute_C(array: np.ndarray, K: int) -> np.ndarray:
    """Calcula C = [C_0, ..., C_8] usando un K explícito."""
    C, _ = ml.gamma_index_rank_ties(
        np.asarray(array, dtype=float),
        max_gamma=MAX_GAMMA,
        K=int(0.0),
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
    Ajusta
        C_d = A exp(-lambda d)
    mediante regresión lineal de log(C_d) frente a d.
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


def fit_power_law(
    C: np.ndarray,
    d_min: int = FIT_D_MIN,
    d_max: int = FIT_D_MAX,
) -> dict[str, Any]:
    """
    Ajusta
        C_d = A d^(-alpha)
    mediante regresión lineal de log(C_d) frente a log(d).
    """
    C = np.asarray(C, dtype=float).ravel()
    d_all = np.arange(len(C), dtype=float)

    mask = (
        (d_all >= d_min)
        & (d_all <= d_max)
        & (d_all > 0)
        & np.isfinite(C)
        & (C > 0)
    )

    d = d_all[mask]
    C_valid = C[mask]
    d_fit = np.arange(d_min, d_max + 1, dtype=float)

    if len(d) < 2:
        return {
            "alpha": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "A": np.nan,
            "r_squared": np.nan,
            "rmse_log": np.nan,
            "d_fit": d_fit,
            "C_fit": np.full(d_fit.shape, np.nan, dtype=float),
            "n_fit": len(d),
        }

    x = np.log(d)
    y = np.log(C_valid)

    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = intercept + slope * x

    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_log = float(np.sqrt(np.mean(residuals**2)))

    alpha = -float(slope)
    A = float(np.exp(intercept))
    C_fit = A * d_fit ** (-alpha)

    return {
        "alpha": alpha,
        "slope": float(slope),
        "intercept": float(intercept),
        "A": A,
        "r_squared": r_squared,
        "rmse_log": rmse_log,
        "d_fit": d_fit,
        "C_fit": C_fit,
        "n_fit": len(d),
    }


def choose_best_observed_fit(
    fit_exp: dict[str, Any],
    fit_power: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Selecciona el mejor ajuste por menor RMSE en log(C_d).

    Ambos modelos tienen dos parámetros (A y lambda/alpha), se ajustan sobre
    los mismos d y sus errores se evalúan en el mismo espacio log(C_d).
    En caso de empate numérico se favorece el exponencial.
    """
    rmse_exp = fit_exp["rmse_log"]
    rmse_power = fit_power["rmse_log"]

    exp_ok = np.isfinite(rmse_exp)
    power_ok = np.isfinite(rmse_power)

    if exp_ok and power_ok:
        if rmse_exp <= rmse_power:
            return "exponential", fit_exp
        return "powerlaw", fit_power

    if exp_ok:
        return "exponential", fit_exp
    if power_ok:
        return "powerlaw", fit_power

    return "none", fit_exp


def geometric_mean_positive(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Media geométrica ignorando valores no positivos o no finitos."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    log_values = np.where(valid, np.log(values), np.nan)

    with np.errstate(invalid="ignore"):
        result = np.exp(np.nanmean(log_values, axis=axis))

    return result


def compute_shuffle_statistics(
    array: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Genera N_SHUFFLES nulos usando el mismo K de la voz.

    AJUSTE OPERATIVO DEL SHUFFLE:
        siempre exponencial.

    Por compatibilidad con el CSV legado también se calculan alpha y R^2
    power-law del shuffle, pero no se utilizan en figuras ni selección.
    """
    array = np.asarray(array, dtype=float).ravel()
    C_shuffles = np.empty((N_SHUFFLES, FIT_D_MAX + 1), dtype=float)

    # Exponencial: ajuste que sí se usa para el nulo.
    lambda_values = np.full(N_SHUFFLES, np.nan, dtype=float)
    r2_exp_values = np.full(N_SHUFFLES, np.nan, dtype=float)
    rmse_exp_values = np.full(N_SHUFFLES, np.nan, dtype=float)

    # Power-law: sólo para mantener idéntico el CSV legado.
    alpha_values_legacy = np.full(N_SHUFFLES, np.nan, dtype=float)
    r2_power_values_legacy = np.full(N_SHUFFLES, np.nan, dtype=float)

    for iteration in range(N_SHUFFLES):
        shuffled = rng.permutation(array)
        C_null = compute_C(shuffled, K)
        C_shuffles[iteration] = C_null

        fit_exp = fit_exponential_decay(C_null)
        lambda_values[iteration] = fit_exp["lambda_decay"]
        r2_exp_values[iteration] = fit_exp["r_squared"]
        rmse_exp_values[iteration] = fit_exp["rmse_log"]

        # Compatibilidad con el CSV antiguo.
        fit_power = fit_power_law(C_null)
        alpha_values_legacy[iteration] = fit_power["alpha"]
        r2_power_values_legacy[iteration] = fit_power["r_squared"]

    C_center = geometric_mean_positive(C_shuffles, axis=0)
    C_low = np.nanpercentile(C_shuffles, LOW_PERCENTILE, axis=0)
    C_high = np.nanpercentile(C_shuffles, HIGH_PERCENTILE, axis=0)

    fit_center_exp = fit_exponential_decay(C_center)
    fit_center_power_legacy = fit_power_law(C_center)

    return {
        "C_all": C_shuffles,
        "C_center": C_center,
        "C_low": C_low,
        "C_high": C_high,

        # Ajuste exponencial usado.
        "fit_center_exp": fit_center_exp,
        "lambda_mean": float(np.nanmean(lambda_values)),
        "lambda_std": float(np.nanstd(lambda_values, ddof=1)),
        "lambda_low": float(np.nanpercentile(lambda_values, LOW_PERCENTILE)),
        "lambda_high": float(np.nanpercentile(lambda_values, HIGH_PERCENTILE)),
        "r2_exp_mean": float(np.nanmean(r2_exp_values)),
        "rmse_exp_mean": float(np.nanmean(rmse_exp_values)),

        # Sólo compatibilidad con SUMMARY_CSV legado.
        "fit_center_power_legacy": fit_center_power_legacy,
        "alpha_mean_legacy": float(np.nanmean(alpha_values_legacy)),
        "alpha_std_legacy": float(np.nanstd(alpha_values_legacy, ddof=1)),
        "alpha_low_legacy": float(
            np.nanpercentile(alpha_values_legacy, LOW_PERCENTILE)
        ),
        "alpha_high_legacy": float(
            np.nanpercentile(alpha_values_legacy, HIGH_PERCENTILE)
        ),
        "r2_power_mean_legacy": float(np.nanmean(r2_power_values_legacy)),
    }


def positive_limits(values: list[float]) -> tuple[float, float]:
    """Límites comunes del eje vertical para los cuatro subplots."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]

    if arr.size == 0:
        return 1e-4, 1.0

    # Se conserva el comportamiento de los scripts originales.
    return 0.01, 1.0


def add_summary_values(
    summary: dict[str, Any],
    prefix: str,
    values: np.ndarray,
) -> None:
    """Añade C_0, C_1, ... al registro."""
    for d, value in enumerate(np.asarray(values, dtype=float)):
        summary[f"{prefix}_{d}"] = value


def evaluate_fit_curve(model: str, fit: dict[str, Any], d: np.ndarray) -> np.ndarray:
    """Evalúa un ajuste sobre una malla d para dibujarlo suavemente."""
    d = np.asarray(d, dtype=float)

    if model == "exponential":
        return fit["A"] * np.exp(-fit["lambda_decay"] * d)
    if model == "powerlaw":
        return fit["A"] * d ** (-fit["alpha"])

    return np.full_like(d, np.nan, dtype=float)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el corpus:\n{CSV_PATH.resolve()}")

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

    # Debe conservar exactamente el esquema del CSV power-law original.
    legacy_summary_rows: list[dict[str, Any]] = []

    # Nuevo resumen completo.
    model_summary_rows: list[dict[str, Any]] = []

    total_movements = len(corpus)

    for row_index, row in corpus.iterrows():
        composer = (
            str(row["composer_canonical"])
            if (
                "composer_canonical" in corpus.columns
                and pd.notna(row["composer_canonical"])
            )
            else str(row["composer"])
        )

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

        for voice in VOICE_ORDER:
            midi, intervals = preprocess_sequence(voices[voice])

            if ANALYSIS_MODE == "intervals":
                sequence = intervals
            elif ANALYSIS_MODE == "midi":
                sequence = midi
            else:
                raise ValueError(
                    "ANALYSIS_MODE debe ser 'midi' o 'intervals'. "
                    f"Valor recibido: {ANALYSIS_MODE!r}"
                )

            if sequence.size < FIT_D_MAX + 2:
                print(
                    f"    ADVERTENCIA: {voice} tiene solo "
                    f"{sequence.size} datos útiles."
                )
                continue

            try:
                K, n_unique, C1_null_exact = determine_K_from_C1_null(
                    sequence,
                    C1_NULL_TARGET,
                )

                C_observed = compute_C(sequence, K)

                # Ambos ajustes del observado.
                fit_exp_observed = fit_exponential_decay(C_observed)
                fit_power_observed = fit_power_law(C_observed)

                best_model, best_fit = choose_best_observed_fit(
                    fit_exp_observed,
                    fit_power_observed,
                )

                local_seed = int(rng_master.integers(0, 2**32 - 1))
                local_rng = np.random.default_rng(local_seed)

                # El shuffle operativo es siempre exponencial.
                null = compute_shuffle_statistics(sequence, K, local_rng)

            except Exception as exc:
                print(f"    ERROR en {voice}: {exc}")
                continue

            movement_results[voice] = {
                "sequence": sequence,
                "C_observed": C_observed,
                "fit_exp_observed": fit_exp_observed,
                "fit_power_observed": fit_power_observed,
                "best_model": best_model,
                "best_fit": best_fit,
                "null": null,
                "K": K,
                "n_unique": n_unique,
                "C1_null_exact": C1_null_exact,
            }

            # --------------------------------------------------------
            # CSV LEGADO: MISMO ESQUEMA QUE Cd_powerlaw_C1null_corpus.py
            # --------------------------------------------------------
            legacy_summary = {
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
                "alpha_observed": fit_power_observed["alpha"],
                "slope_loglog_observed": fit_power_observed["slope"],
                "intercept_loglog_observed": fit_power_observed["intercept"],
                "A_observed": fit_power_observed["A"],
                "r_squared_observed": fit_power_observed["r_squared"],
                "rmse_log_observed": fit_power_observed["rmse_log"],
                "alpha_null_mean": null["alpha_mean_legacy"],
                "alpha_null_std": null["alpha_std_legacy"],
                f"alpha_null_p{LOW_PERCENTILE:g}": null["alpha_low_legacy"],
                f"alpha_null_p{HIGH_PERCENTILE:g}": null["alpha_high_legacy"],
                "alpha_fit_null_center": null["fit_center_power_legacy"]["alpha"],
                "r_squared_fit_null_center": null["fit_center_power_legacy"][
                    "r_squared"
                ],
                "r_squared_null_mean": null["r2_power_mean_legacy"],
            }

            add_summary_values(legacy_summary, "C_observed", C_observed)
            add_summary_values(
                legacy_summary,
                "C_null_geometric_mean",
                null["C_center"],
            )
            add_summary_values(
                legacy_summary,
                f"C_null_p{LOW_PERCENTILE:g}",
                null["C_low"],
            )
            add_summary_values(
                legacy_summary,
                f"C_null_p{HIGH_PERCENTILE:g}",
                null["C_high"],
            )

            legacy_summary_rows.append(legacy_summary)

            # --------------------------------------------------------
            # CSV NUEVO: INFORMACIÓN COMPLETA DE SELECCIÓN
            # --------------------------------------------------------
            rmse_exp = fit_exp_observed["rmse_log"]
            rmse_power = fit_power_observed["rmse_log"]
            r2_exp = fit_exp_observed["r_squared"]
            r2_power = fit_power_observed["r_squared"]

            model_summary = {
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
                "K": K,
                "n_unique": n_unique,
                "C1_null_target": C1_NULL_TARGET,
                "C1_null_exact": C1_null_exact,
                "best_model_observed": best_model,
                "selection_criterion": "minimum_rmse_log",
                "rmse_log_exponential_observed": rmse_exp,
                "rmse_log_powerlaw_observed": rmse_power,
                "delta_rmse_exp_minus_power_observed": (
                    rmse_exp - rmse_power
                    if np.isfinite(rmse_exp) and np.isfinite(rmse_power)
                    else np.nan
                ),
                "r_squared_exponential_observed": r2_exp,
                "r_squared_powerlaw_observed": r2_power,
                "delta_r2_exp_minus_power_observed": (
                    r2_exp - r2_power
                    if np.isfinite(r2_exp) and np.isfinite(r2_power)
                    else np.nan
                ),

                # Exponencial observado
                "lambda_observed": fit_exp_observed["lambda_decay"],
                "decay_length_observed": fit_exp_observed["decay_length"],
                "A_exponential_observed": fit_exp_observed["A"],
                "slope_semilog_observed": fit_exp_observed["slope"],
                "intercept_semilog_observed": fit_exp_observed["intercept"],

                # Power-law observado
                "alpha_observed": fit_power_observed["alpha"],
                "A_powerlaw_observed": fit_power_observed["A"],
                "slope_loglog_observed": fit_power_observed["slope"],
                "intercept_loglog_observed": fit_power_observed["intercept"],

                # Shuffle: SIEMPRE exponencial
                "shuffle_model": "exponential",
                "lambda_null_mean": null["lambda_mean"],
                "lambda_null_std": null["lambda_std"],
                f"lambda_null_p{LOW_PERCENTILE:g}": null["lambda_low"],
                f"lambda_null_p{HIGH_PERCENTILE:g}": null["lambda_high"],
                "lambda_fit_null_center": null["fit_center_exp"]["lambda_decay"],
                "decay_length_fit_null_center": null["fit_center_exp"][
                    "decay_length"
                ],
                "A_fit_null_center_exponential": null["fit_center_exp"]["A"],
                "r_squared_fit_null_center_exponential": null["fit_center_exp"][
                    "r_squared"
                ],
                "rmse_log_fit_null_center_exponential": null["fit_center_exp"][
                    "rmse_log"
                ],
                "r_squared_null_exponential_mean": null["r2_exp_mean"],
                "rmse_log_null_exponential_mean": null["rmse_exp_mean"],
            }

            add_summary_values(model_summary, "C_observed", C_observed)
            add_summary_values(
                model_summary,
                "C_null_geometric_mean",
                null["C_center"],
            )
            add_summary_values(
                model_summary,
                f"C_null_p{LOW_PERCENTILE:g}",
                null["C_low"],
            )
            add_summary_values(
                model_summary,
                f"C_null_p{HIGH_PERCENTILE:g}",
                null["C_high"],
            )

            model_summary_rows.append(model_summary)

        if not movement_results:
            continue

        # ----------------------------------------------------
        # FIGURA 2 x 2
        # ----------------------------------------------------
        # Se usa eje y logarítmico y x lineal para todos los paneles.
        # Así el shuffle exponencial sigue siendo una recta, mientras que
        # un mejor ajuste power-law observado aparece con su curvatura natural.
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
            fit_exp_observed = result["fit_exp_observed"]
            fit_power_observed = result["fit_power_observed"]
            best_model = result["best_model"]
            best_fit = result["best_fit"]
            null = result["null"]

            d = np.arange(len(C_observed), dtype=float)
            display_mask = (d >= FIT_D_MIN) & (d <= FIT_D_MAX)

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

            # Banda del nulo.
            ax.fill_between(
                d_plot[valid_band],
                C_low_plot[valid_band],
                C_high_plot[valid_band],
                color=NULL_COLOR,
                alpha=0.18,
                linewidth=0,
                label=f"Shuffle P{LOW_PERCENTILE:g}–P{HIGH_PERCENTILE:g}",
                zorder=1,
            )

            # Observado.
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
                label=r"$C_d$ observado",
                zorder=4,
            )

            # Sólo se grafica el MEJOR ajuste del observado.
            d_dense = np.linspace(FIT_D_MIN, FIT_D_MAX, 250)
            C_best_dense = evaluate_fit_curve(best_model, best_fit, d_dense)

            best_label = (
                "Mejor ajuste observado: exponencial"
                if best_model == "exponential"
                else "Mejor ajuste observado: ley de potencia"
            )

            ax.semilogy(
                d_dense,
                C_best_dense,
                linestyle="-",
                linewidth=2.1,
                color=VOICE_COLORS[voice],
                label=best_label,
                zorder=3,
            )

            # Curva central del shuffle.
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
                label="Shuffle: media geométrica",
                zorder=4,
            )

            # Shuffle: SIEMPRE ajuste exponencial.
            fit_null_exp = null["fit_center_exp"]
            C_null_fit_dense = evaluate_fit_curve(
                "exponential",
                fit_null_exp,
                d_dense,
            )

            ax.semilogy(
                d_dense,
                C_null_fit_dense,
                linestyle="--",
                linewidth=1.9,
                color=NULL_COLOR,
                label="Shuffle: ajuste exponencial",
                zorder=3,
            )

            ax.set_title(VOICE_TITLES[voice])
            ax.legend(
                loc="upper right",
                frameon=False,
                title=(
                    rf"$K=0$, $n_{{\rm unique}}={result['n_unique']}$"
                ),
            )

            ax.set_xlim(0.7, 8.3)
            ax.set_ylim(*y_limits)
            ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

            ax.grid(True, which="major", alpha=0.26)
            ax.grid(True, which="minor", alpha=0.10)

            # Información de los dos ajustes, aunque sólo se dibuje el mejor.
            annotation = (
                rf"Mejor: {best_model}"
                "\n"
                rf"$RMSE_{{exp}}={fit_exp_observed['rmse_log']:.4f}$, $RMSE_{{pow}}={fit_power_observed['rmse_log']:.4f}$"
                "\n"
                rf"$R^2_{{exp}}={fit_exp_observed['r_squared']:.3f}$,$R^2_{{pow}}={fit_power_observed['r_squared']:.3f}$"
                "\n"
                rf"$\lambda_{{null}}={null['lambda_mean']:.3f}"
                rf"\pm{null['lambda_std']:.3f}$"
            )

            ax.text(
                0.03,
                0.05,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.8,
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
            rf"Observado: mejor entre exponencial y ley de potencia; "
            rf"shuffle: exponencial; $C_{{1,\mathrm{{null}}}}^*="
            rf"{C1_NULL_TARGET:.5f}$",
            y=0.985,
        )

        fig.tight_layout(rect=(0.035, 0.045, 0.99, 0.93))

        composer_directory = OUTPUT_DIR / slugify(composer)
        composer_directory.mkdir(parents=True, exist_ok=True)

        file_stem = (
            f"{slugify(row.get('work_id', work_title))}"
            f"__mov_{movement_no}"
            f"__Cd_bestfit_observed_shuffle_exponential"
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

    # CSV legado: mismo nombre y esquema.
    legacy_summary_df = pd.DataFrame(legacy_summary_rows)
    legacy_summary_df.to_csv(SUMMARY_CSV, index=False)

    # CSV nuevo.
    model_summary_df = pd.DataFrame(model_summary_rows)
    model_summary_df.to_csv(MODEL_SELECTION_CSV, index=False)

    print("\nProceso terminado.")
    print(f"Figuras: {OUTPUT_DIR.resolve()}")
    print(f"Resumen legado: {SUMMARY_CSV.resolve()}")
    print(f"Resumen selección de modelo: {MODEL_SELECTION_CSV.resolve()}")

    if not model_summary_df.empty and "best_model_observed" in model_summary_df:
        counts = model_summary_df["best_model_observed"].value_counts(dropna=False)
        print("\nModelos seleccionados en el observado:")
        for model, count in counts.items():
            print(f"    {model}: {count}")


if __name__ == "__main__":
    main()
