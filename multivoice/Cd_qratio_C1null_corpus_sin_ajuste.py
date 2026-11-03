# -*- coding: utf-8 -*-
"""
Gráficas de q_d = C_{d+1}/C_d para el corpus, sin ningún ajuste.

Para cada movimiento:
    - crea una figura 2x2, una subgráfica por voz;
    - calcula C = [C_0, ..., C_8] para la serie observada;
    - calcula

          q_d = C_{d+1}/C_d,      d = 1, ..., 7;

    - genera N_SHUFFLES versiones barajadas de la misma voz, usando el mismo K;
    - calcula q_d para cada shuffle;
    - grafica únicamente:
          * los puntos q_d observados;
          * los puntos centrales del ensemble shuffle;
          * la banda percentil del ensemble shuffle.

NO se hace ningún ajuste exponencial, power-law, lineal ni de máxima
verosimilitud. Tampoco se selecciona ningún modelo.

Interpretación útil:
    q_d = C_{d+1}/C_d

es la fracción de recurrencias de longitud d que continúan siendo recurrentes
una dimensión adicional. Si la caída de C_d es exponencial regular, q_d será
aproximadamente constante y los puntos aparecerán aproximadamente horizontales.

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

OUTPUT_DIR = Path("figuras_qratio_C1null_shuffle_sin_ajuste")
SUMMARY_CSV = OUTPUT_DIR / "resumen_qratio_C1null_sin_ajuste.csv"

# "midi" o "intervals"
ANALYSIS_MODE = "intervals"

MAX_GAMMA = 9

# Probabilidad nula de cercanía objetivo común a todas las voces.
C1_NULL_TARGET = 0.5807

# C = [C_0, C_1, ..., C_8].
D_MIN = 1
D_MAX = MAX_GAMMA + 1

# q_d = C_{d+1}/C_d queda definido para d=1,...,7.
Q_D_MIN = D_MIN
Q_D_MAX = D_MAX - 1

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
        "legend.fontsize": 9.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
    }
)


# ============================================================
# LECTURA Y PREPROCESAMIENTO
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


# ============================================================
# K A PARTIR DE C_{1,NULL}
# ============================================================


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
    K = 0 if admissible.size == 0 else int(admissible[-1])

    return K, n_unique, float(c1_values[K])


# ============================================================
# C_d Y q_d
# ============================================================


def compute_C(array: np.ndarray, K: int) -> np.ndarray:
    """Calcula C = [C_0, ..., C_8] usando un K explícito."""
    C, _ = ml.gamma_index_rank_ties(
        np.asarray(array, dtype=float),
        max_gamma=MAX_GAMMA,
        K=int(K),
    )

    C = np.asarray(C, dtype=float).ravel()

    expected_length = D_MAX + 1
    if len(C) < expected_length:
        raise ValueError(
            f"Se esperaban al menos {expected_length} valores de C, "
            f"pero se obtuvieron {len(C)}."
        )

    return C[:expected_length]


def compute_q_from_C(
    C: np.ndarray,
    d_min: int = Q_D_MIN,
    d_max: int = Q_D_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula q_d = C_{d+1}/C_d para d=d_min,...,d_max."""
    C = np.asarray(C, dtype=float).ravel()
    d_values = np.arange(d_min, d_max + 1, dtype=int)
    q_values = np.full(d_values.shape, np.nan, dtype=float)

    for i, d in enumerate(d_values):
        if d + 1 >= len(C):
            continue

        denominator = C[d]
        numerator = C[d + 1]

        if (
            np.isfinite(denominator)
            and np.isfinite(numerator)
            and denominator > 0
        ):
            q_values[i] = numerator / denominator

    return d_values.astype(float), q_values


# ============================================================
# SHUFFLE
# ============================================================


def geometric_mean_positive(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Media geométrica ignorando valores no positivos o no finitos."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    log_values = np.where(valid, np.log(values), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.exp(np.nanmean(log_values, axis=axis))

    return result


def compute_shuffle_statistics(
    array: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Genera N_SHUFFLES nulos usando el mismo K de la voz.

    No se ajusta ningún modelo. Sólo se guardan C_d y q_d de cada shuffle
    y sus estadísticas descriptivas punto a punto.
    """
    array = np.asarray(array, dtype=float).ravel()

    C_shuffles = np.full(
        (N_SHUFFLES, D_MAX + 1),
        np.nan,
        dtype=float,
    )
    q_shuffles = np.full(
        (N_SHUFFLES, Q_D_MAX - Q_D_MIN + 1),
        np.nan,
        dtype=float,
    )

    for iteration in range(N_SHUFFLES):
        shuffled = rng.permutation(array)
        C_null = compute_C(shuffled, K)
        _, q_null = compute_q_from_C(C_null)

        C_shuffles[iteration] = C_null
        q_shuffles[iteration] = q_null

    # Se conserva la media geométrica usada en los scripts anteriores.
    C_center = geometric_mean_positive(C_shuffles, axis=0)
    q_center = geometric_mean_positive(q_shuffles, axis=0)

    q_low = np.nanpercentile(q_shuffles, LOW_PERCENTILE, axis=0)
    q_high = np.nanpercentile(q_shuffles, HIGH_PERCENTILE, axis=0)

    return {
        "C_all": C_shuffles,
        "C_center": C_center,
        "q_all": q_shuffles,
        "q_center": q_center,
        "q_low": q_low,
        "q_high": q_high,
    }


# ============================================================
# CSV
# ============================================================


def add_indexed_values(
    summary: dict[str, Any],
    prefix: str,
    values: np.ndarray,
    start_index: int,
) -> None:
    """Añade valores indexados al registro."""
    for offset, value in enumerate(np.asarray(values)):
        summary[f"{prefix}_{start_index + offset}"] = value


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
    summary_rows: list[dict[str, Any]] = []

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
                # Se conserva la convención del script de partida.
                sequence = intervals
            elif ANALYSIS_MODE == "midi":
                sequence = midi
            else:
                raise ValueError(
                    "ANALYSIS_MODE debe ser 'midi' o 'intervals'. "
                    f"Valor recibido: {ANALYSIS_MODE!r}"
                )

            if sequence.size < D_MAX + 2:
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
                d_q, q_observed = compute_q_from_C(C_observed)

                local_seed = int(rng_master.integers(0, 2**32 - 1))
                local_rng = np.random.default_rng(local_seed)
                null = compute_shuffle_statistics(sequence, K, local_rng)

            except Exception as exc:
                print(f"    ERROR en {voice}: {exc}")
                continue

            movement_results[voice] = {
                "C_observed": C_observed,
                "d_q": d_q,
                "q_observed": q_observed,
                "null": null,
                "K": K,
                "n_unique": n_unique,
                "C1_null_exact": C1_null_exact,
            }

            summary: dict[str, Any] = {
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
                "n_sequence": len(sequence),
                "d_min": D_MIN,
                "d_max": D_MAX,
                "q_d_min": Q_D_MIN,
                "q_d_max": Q_D_MAX,
                "n_shuffles": N_SHUFFLES,
                "K": K,
                "n_unique": n_unique,
                "C1_null_target": C1_NULL_TARGET,
                "C1_null_exact": C1_null_exact,
            }

            add_indexed_values(summary, "C_observed", C_observed, 0)
            add_indexed_values(summary, "q_observed", q_observed, Q_D_MIN)
            add_indexed_values(
                summary,
                "q_null_geometric_mean",
                null["q_center"],
                Q_D_MIN,
            )
            add_indexed_values(
                summary,
                f"q_null_p{LOW_PERCENTILE:g}",
                null["q_low"],
                Q_D_MIN,
            )
            add_indexed_values(
                summary,
                f"q_null_p{HIGH_PERCENTILE:g}",
                null["q_high"],
                Q_D_MIN,
            )

            summary_rows.append(summary)

        if not movement_results:
            continue

        # ----------------------------------------------------
        # FIGURA 2 x 2: SÓLO PUNTOS OBSERVADOS + SHUFFLE
        # ----------------------------------------------------
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14.5, 10.0),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, voice in zip(axes, VOICE_ORDER):
            if voice not in movement_results:
                ax.set_visible(False)
                continue

            result = movement_results[voice]
            d_q = result["d_q"]
            q_observed = result["q_observed"]
            null = result["null"]

            valid_band = (
                np.isfinite(null["q_low"])
                & np.isfinite(null["q_high"])
            )

            # Banda del ensemble shuffle.
            ax.fill_between(
                d_q[valid_band],
                null["q_low"][valid_band],
                null["q_high"][valid_band],
                color=NULL_COLOR,
                alpha=0.18,
                linewidth=0,
                label=f"Shuffle P{LOW_PERCENTILE:g}–P{HIGH_PERCENTILE:g}",
                zorder=1,
            )

            # Puntos observados.
            valid_obs = np.isfinite(q_observed)
            ax.plot(
                d_q[valid_obs],
                q_observed[valid_obs],
                linestyle="none",
                marker="o",
                markersize=7,
                markerfacecolor=VOICE_COLORS[voice],
                markeredgecolor="white",
                markeredgewidth=0.7,
                color=VOICE_COLORS[voice],
                label=r"Observado: $q_d=C_{d+1}/C_d$",
                zorder=4,
            )

            # Puntos centrales del shuffle: SIN línea y SIN ajuste.
            valid_null = np.isfinite(null["q_center"])
            ax.plot(
                d_q[valid_null],
                null["q_center"][valid_null],
                linestyle="none",
                marker="s",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=NULL_COLOR,
                markeredgewidth=1.2,
                color=NULL_COLOR,
                label="Shuffle: media geométrica",
                zorder=4,
            )

            ax.set_title(VOICE_TITLES[voice])
            ax.legend(
                loc="best",
                frameon=False,
                title=(
                    rf"$K={result['K']}$"
                    + "\n"
                    + rf"$n_{{\rm unique}}={result['n_unique']}$"
                ),
            )

            ax.set_xlim(Q_D_MIN - 0.25, Q_D_MAX + 0.25)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks(np.arange(Q_D_MIN, Q_D_MAX + 1))
            ax.grid(True, which="major", alpha=0.26)

        fig.supxlabel(r"Longitud previa $d$")
        fig.supylabel(r"$q_d=C_{d+1}/C_d$")

        fig.suptitle(
            f"{composer} | {work_title} | Movimiento {movement_no}\n"
            rf"Sin ajustes: observado y shuffle; "
            rf"$C_{{1,\mathrm{{null}}}}^*={C1_NULL_TARGET:.5f}$",
            y=0.985,
        )

        fig.tight_layout(rect=(0.035, 0.045, 0.99, 0.93))

        composer_directory = OUTPUT_DIR / slugify(composer)
        composer_directory.mkdir(parents=True, exist_ok=True)

        file_stem = (
            f"{slugify(row.get('work_id', work_title))}"
            f"__mov_{movement_no}"
            f"__qratio_sin_ajuste"
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
