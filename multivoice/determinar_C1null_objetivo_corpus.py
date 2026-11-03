# -*- coding: utf-8 -*-
"""
Calibración global de C_{1,null} para las integrales de correlación por rangos
con empates.

Objetivo
--------
En lugar de usar el mismo mu para todas las voces, elegir para cada voz un K
propio de modo que la probabilidad nula de cercanía C_{1,null} sea lo más
parecida posible a un valor objetivo común.

Punto clave
-----------
Para d=1 no hace falta generar shuffles. Si el shuffle conserva exactamente
la distribución marginal, el valor esperado de C_{1,null}(K) se obtiene de
forma exacta a partir de las multiplicidades de los rangos:

    C1_null(K) = (# pares de posiciones con |r_i-r_j| <= K) / C(N, 2).

Con empates, K=0 todavía cuenta pares con el mismo rango, por lo que cada voz
tiene un mínimo distinto:

    C1_min = C1_null(0).

Por ello, el valor común más bajo que es alcanzable por TODAS las voces es:

    C1_target_all = max_voces C1_min.

Como K es entero, no siempre se puede alcanzar exactamente el target; el
script elige el K cuyo C1_null queda más cerca y cuantifica el error.

Además se calcula una alternativa robusta (por defecto percentil 95 de los
C1_min). Esa alternativa cubre aproximadamente el 95 % de las voces y evita
que una única voz con una distribución marginal extremadamente concentrada
obligue a usar un C1_target demasiado grande en todo el corpus.

Salidas
-------
- calibracion_C1null_por_voz.csv
- resumen_C1null_objetivos.csv
- diagnostico_C1null_signed.png
- diagnostico_C1null_absolute.png

El script evalúa AMBAS convenciones de intervalos:
    signed   : Delta p_n = p_{n+1} - p_n
    absolute : |Delta p_n|
Así se puede comparar directamente con scripts antiguos que usen np.abs().
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

CSV_PATH = Path("quartet_subcorpus_all_movements.csv")
OUTPUT_DIR = Path("calibracion_C1null")

VOICE_ORDER = ["vln I", "vln II", "viola", "cello"]

# Fracción de voces que se desea cubrir con la alternativa robusta.
# 0.95 significa que ~95 % de las voces tendrán C1_min <= target_robust.
ROBUST_COVERAGE = 0.95

# Se analizan ambas para que el diagnóstico revele el efecto de np.abs().
INTERVAL_MODES = ("signed", "absolute")

# Cuántas voces extremas imprimir para diagnosticar qué fija el target global.
N_TOP_LIMITING = 10


# ============================================================
# LECTURA DEL CORPUS
# ============================================================

def parse_literal(value: Any) -> Any:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_voices(row: pd.Series) -> dict[str, np.ndarray]:
    voices = parse_literal(row["voices_pitches"])
    roles = [str(role).strip() for role in str(row["roles"]).split("|")]

    if len(voices) != len(roles):
        raise ValueError(
            f"Número de voces ({len(voices)}) distinto del número de roles "
            f"({len(roles)})."
        )

    voice_map: dict[str, np.ndarray] = {}
    for role, pitches in zip(roles, voices):
        x = np.asarray(pitches, dtype=float).ravel()
        x = x[np.isfinite(x)]
        voice_map[role] = x

    missing = [voice for voice in VOICE_ORDER if voice not in voice_map]
    if missing:
        raise ValueError(f"Faltan voces esperadas: {missing}")

    return {voice: voice_map[voice] for voice in VOICE_ORDER}


def make_intervals(pitches: np.ndarray, mode: str) -> np.ndarray:
    intervals = np.diff(np.asarray(pitches, dtype=float))

    if mode == "signed":
        return intervals
    if mode == "absolute":
        return np.abs(intervals)

    raise ValueError("mode debe ser 'signed' o 'absolute'.")


# ============================================================
# C1 NULL EXACTO
# ============================================================

def c1_null_curve_exact(sequence: np.ndarray) -> dict[str, Any]:
    """
    Calcula C_{1,null}(K) exactamente para K = 0, ..., n_unique-1.

    Los valores distintos se ordenan y se les asignan rangos consecutivos
    0, 1, ..., q-1, exactamente como en ranks_with_ties.

    Sea n_r la multiplicidad del rango r. Para K dado, cuentan:
      - pares dentro del mismo rango: C(n_r, 2),
      - pares entre rangos r y s con 1 <= |r-s| <= K: n_r n_s.

    Esta es la probabilidad exacta para dos POSICIONES distintas elegidas sin
    reemplazo, que corresponde a la normalización por pares de tu C_1.
    """
    x = np.asarray(sequence, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size < 2:
        raise ValueError("Se necesitan al menos dos datos para calcular C1.")

    unique_values, counts = np.unique(x, return_counts=True)
    counts = counts.astype(np.int64)

    N = int(counts.sum())
    q = int(len(counts))
    denominator = N * (N - 1) / 2.0

    C1 = np.empty(q, dtype=float)

    # K = 0: solamente pares empatados (mismo rango).
    close_pairs = float(np.sum(counts * (counts - 1) / 2.0))
    C1[0] = close_pairs / denominator

    # Al pasar de K-1 a K se añaden los pares separados exactamente K rangos.
    for K in range(1, q):
        close_pairs += float(np.sum(counts[:-K] * counts[K:]))
        C1[K] = close_pairs / denominator

    # Por construcción, K=q-1 acepta todos los pares.
    # Una pequeña corrección numérica evita 0.9999999999999999.
    C1[-1] = 1.0

    mode_index = int(np.argmax(counts))

    return {
        "N": N,
        "n_unique": q,
        "unique_values": unique_values,
        "counts": counts,
        "C1_curve": C1,
        "C1_min": float(C1[0]),
        "modal_value": float(unique_values[mode_index]),
        "modal_fraction": float(counts[mode_index] / N),
    }


def choose_K_nearest(C1_curve: np.ndarray, target: float) -> tuple[int, float, float]:
    """Elige K que minimiza |C1_null(K) - target|."""
    C1_curve = np.asarray(C1_curve, dtype=float)
    K = int(np.argmin(np.abs(C1_curve - target)))
    achieved = float(C1_curve[K])
    error = float(achieved - target)
    return K, achieved, error


# ============================================================
# OBJETIVOS GLOBAL Y ROBUSTO
# ============================================================

def calibrate_mode(records: list[dict[str, Any]], mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode_records = [r for r in records if r["interval_mode"] == mode]
    c1_min = np.asarray([r["C1_min"] for r in mode_records], dtype=float)

    # Estricto: es el MENOR target que puede ser alcanzado (en sentido continuo)
    # por todas las voces, porque ninguna voz puede bajar de su C1_null(0).
    target_all = float(np.max(c1_min))

    # Robusto: permite diagnosticar si unos pocos outliers fuerzan target_all
    # a un valor demasiado grande.
    target_robust = float(np.quantile(c1_min, ROBUST_COVERAGE))

    rows: list[dict[str, Any]] = []

    for rec in mode_records:
        curve = rec["C1_curve"]

        K_all, c_all, e_all = choose_K_nearest(curve, target_all)

        robust_feasible = bool(rec["C1_min"] <= target_robust)
        if robust_feasible:
            K_rob, c_rob, e_rob = choose_K_nearest(curve, target_robust)
        else:
            K_rob, c_rob, e_rob = 0, float(curve[0]), float(curve[0] - target_robust)

        row = {k: v for k, v in rec.items() if k != "C1_curve"}
        row.update(
            {
                "target_all": target_all,
                "K_target_all": K_all,
                "C1_target_all_achieved": c_all,
                "error_target_all_signed": e_all,
                "error_target_all_abs": abs(e_all),
                "target_robust": target_robust,
                "robust_feasible": robust_feasible,
                "K_target_robust": K_rob,
                "C1_target_robust_achieved": c_rob,
                "error_target_robust_signed": e_rob,
                "error_target_robust_abs": abs(e_rob),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)

    limiting_idx = int(np.argmax(c1_min))
    limiting = mode_records[limiting_idx]

    summary = {
        "interval_mode": mode,
        "n_voices": len(mode_records),
        "target_all": target_all,
        "target_robust": target_robust,
        "robust_coverage_requested": ROBUST_COVERAGE,
        "robust_n_feasible": int(np.sum(c1_min <= target_robust)),
        "robust_fraction_feasible": float(np.mean(c1_min <= target_robust)),
        "C1_min_min": float(np.min(c1_min)),
        "C1_min_median": float(np.median(c1_min)),
        "C1_min_p90": float(np.quantile(c1_min, 0.90)),
        "C1_min_p95": float(np.quantile(c1_min, 0.95)),
        "C1_min_p99": float(np.quantile(c1_min, 0.99)),
        "C1_min_max": float(np.max(c1_min)),
        "all_mean_abs_matching_error": float(out["error_target_all_abs"].mean()),
        "all_p95_abs_matching_error": float(out["error_target_all_abs"].quantile(0.95)),
        "all_max_abs_matching_error": float(out["error_target_all_abs"].max()),
        "limiting_composer": limiting["composer"],
        "limiting_work_title": limiting["work_title"],
        "limiting_movement_no": limiting["movement_no"],
        "limiting_voice": limiting["voice"],
        "limiting_n_unique": limiting["n_unique"],
        "limiting_modal_value": limiting["modal_value"],
        "limiting_modal_fraction": limiting["modal_fraction"],
    }

    return out, summary


# ============================================================
# FIGURA DIAGNÓSTICA
# ============================================================

def plot_diagnostic(df_mode: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    mode = summary["interval_mode"]
    target_all = summary["target_all"]
    target_robust = summary["target_robust"]

    fig, ax = plt.subplots(figsize=(9.2, 5.7))

    ax.hist(df_mode["C1_min"], bins=35, alpha=0.75, edgecolor="white")
    ax.axvline(
        target_all,
        linestyle="-",
        linewidth=2.0,
        label=rf"Todos: $C_{{1,\min}}^{{\max}}={target_all:.3f}$",
    )
    ax.axvline(
        target_robust,
        linestyle="--",
        linewidth=2.0,
        label=(
            rf"Robusto {100*ROBUST_COVERAGE:.0f}%: "
            rf"$C_{{1}}^*={target_robust:.3f}$"
        ),
    )

    ax.set_xlabel(r"$C_1 (K=0)$")
    ax.set_ylabel("Histograma de series por voz")
    ax.set_title(f"Distribución de $C_1 (K=0)$ — intervalos {mode}")
    ax.grid(True, alpha=0.20)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el corpus: {CSV_PATH.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(CSV_PATH)
    records: list[dict[str, Any]] = []

    for row_index, row in corpus.iterrows():
        voices = parse_voices(row)

        composer = (
            str(row["composer_canonical"])
            if "composer_canonical" in corpus.columns and pd.notna(row["composer_canonical"])
            else str(row["composer"])
        )

        for voice in VOICE_ORDER:
            pitches = voices[voice]

            for mode in INTERVAL_MODES:
                sequence = make_intervals(pitches, mode)
                info = c1_null_curve_exact(sequence)

                records.append(
                    {
                        "row_index": row_index,
                        "composer": composer,
                        "composer_id": row.get("composer", np.nan),
                        "work_id": row.get("work_id", np.nan),
                        "work_title": row.get("work_title", np.nan),
                        "movement_no": row.get("movement_no", np.nan),
                        "filename": row.get("filename", np.nan),
                        "voice": voice,
                        "interval_mode": mode,
                        "N": info["N"],
                        "n_unique": info["n_unique"],
                        "C1_min": info["C1_min"],
                        "modal_value": info["modal_value"],
                        "modal_fraction": info["modal_fraction"],
                        "C1_curve": info["C1_curve"],
                    }
                )

    detail_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for mode in INTERVAL_MODES:
        df_mode, summary = calibrate_mode(records, mode)
        detail_frames.append(df_mode)
        summaries.append(summary)

        plot_diagnostic(
            df_mode,
            summary,
            OUTPUT_DIR / f"diagnostico_C1null_{mode}.png",
        )

        print("\n" + "=" * 78)
        print(f"INTERVALOS: {mode}")
        print("=" * 78)
        print(f"Voces analizadas: {summary['n_voices']}")
        print(f"Target estricto para TODAS las voces: {summary['target_all']:.9f}")
        print(
            f"Target robusto P{100*ROBUST_COVERAGE:.0f}: "
            f"{summary['target_robust']:.9f} "
            f"({summary['robust_n_feasible']}/{summary['n_voices']} voces alcanzables)"
        )
        print(
            "Error de discretización con target estricto: "
            f"media={summary['all_mean_abs_matching_error']:.4f}, "
            f"P95={summary['all_p95_abs_matching_error']:.4f}, "
            f"máximo={summary['all_max_abs_matching_error']:.4f}"
        )
        print(
            "Voz que fija el target estricto: "
            f"{summary['limiting_composer']} | "
            f"{summary['limiting_work_title']} | "
            f"mov. {summary['limiting_movement_no']} | "
            f"{summary['limiting_voice']}"
        )
        print(
            f"  C1_min={summary['target_all']:.6f}, "
            f"n_unique={summary['limiting_n_unique']}, "
            f"valor modal={summary['limiting_modal_value']:.3g}, "
            f"fracción modal={summary['limiting_modal_fraction']:.3f}"
        )

        print(f"\nTop {N_TOP_LIMITING} voces con mayor C1_min:")
        cols = [
            "composer",
            "work_title",
            "movement_no",
            "voice",
            "N",
            "n_unique",
            "C1_min",
            "modal_value",
            "modal_fraction",
        ]
        print(
            df_mode.sort_values("C1_min", ascending=False)
            .head(N_TOP_LIMITING)[cols]
            .to_string(index=False)
        )

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_path = OUTPUT_DIR / "calibracion_C1null_por_voz.csv"
    summary_path = OUTPUT_DIR / "resumen_C1null_objetivos.csv"

    detail.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nArchivos guardados:")
    print(f"  {detail_path.resolve()}")
    print(f"  {summary_path.resolve()}")
    for mode in INTERVAL_MODES:
        print(f"  {(OUTPUT_DIR / f'diagnostico_C1null_{mode}.png').resolve()}")


if __name__ == "__main__":
    main()
