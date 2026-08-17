#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de PE y mPE del Cuarteto Op. 10 de Debussy por movimiento y voz.

Calcula, para alturas MIDI e intervalos dirigidos:
    - entropía observada sin normalizar;
    - distribución nula por shuffle de la representación analizada;
    - Z = (H_obs - media_null) / std_null;
    - organization_z = -Z, para que valores positivos altos indiquen
      entropía menor que el shuffle y, por tanto, mayor organización ordinal;
    - p-valores empíricos de cola inferior y superior;
    - cruces de curvas entre voces;
    - rankings por movimiento y resumen entre movimientos;
    - acuerdo entre rankings obtenidos con alturas e intervalos.

Uso recomendado inicial:
    python analisis_mpe_debussy_op10.py \
        --csv "quartet_subcorpus_all_movements(3).csv" \
        --output resultados_debussy_op10 \
        --m-min 2 --m-max 7 --n-shuffles 500

Para evaluar sensibilidad a las longitudes diferentes de las voces:
    python analisis_mpe_debussy_op10.py \
        --csv "quartet_subcorpus_all_movements(3).csv" \
        --output resultados_debussy_op10_longitud_igualada \
        --length-mode movement_min_center \
        --m-min 2 --m-max 7 --n-shuffles 500

Notas metodológicas:
1. Para alturas se barajan directamente las alturas, preservando su histograma.
2. Para intervalos se calcula diff(alturas) y se barajan directamente los
   intervalos, preservando la distribución marginal de intervalos.
3. m representa número de eventos de nota consecutivos, no duración musical.
4. Un cruce de curvas cerca de m=k indica un cambio de organización ordinal
   relativa en esa escala; no demuestra por sí solo un motivo de longitud k-1.
5. A m grande existe submuestreo del espacio de patrones. El archivo largo
   incluye n_vectors y windows_per_factorial como diagnóstico heurístico.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mi_libreria as ml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "No se encontró 'mi_libreria'. Coloca mi_libreria.py en la misma "
        "carpeta del script o instala el módulo en el entorno activo."
    ) from exc

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


VOICE_LABELS_FALLBACK = ["Violín I", "Violín II", "Viola", "Violonchelo"]
REPRESENTATION_LABELS = {
    "pitch": "Alturas MIDI",
    "interval": "Intervalos dirigidos",
}
ENTROPY_LABELS = {
    "mpe": "mPE",
    "pe": "PE",
}


@dataclass(frozen=True)
class Config:
    csv_path: Path
    output_dir: Path
    work_id: str
    m_values: tuple[int, ...]
    tau: int
    n_shuffles: int
    seed: int
    length_mode: str
    entropy_types: tuple[str, ...]
    representations: tuple[str, ...]
    make_plots: bool


def default_csv_path() -> Path:
    """Busca el CSV primero en la carpeta actual y luego junto al script."""
    filename = "quartet_subcorpus_all_movements.csv"
    candidates = [
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Se devuelve una ruta informativa; load_work producirá un error claro.
    return candidates[-1]


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="PE/mPE por voz y movimiento para el Cuarteto Op. 10 de Debussy."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "CSV del subcorpus. Si se omite, se busca "
            "'quartet_subcorpus_all_movements(3).csv' en la carpeta actual "
            "y junto al script."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("resultados_debussy_op10"),
        help="Directorio de salida."
    )
    parser.add_argument("--work-id", default="debussy_op10")
    parser.add_argument("--m-min", type=int, default=2)
    parser.add_argument("--m-max", type=int, default=7)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--length-mode",
        choices=["full", "movement_min_center"],
        default="full",
        help=(
            "full: usa cada voz completa; movement_min_center: recorta cada voz "
            "al centro hasta la longitud mínima del movimiento y representación."
        ),
    )
    parser.add_argument(
        "--entropy-types", nargs="+", choices=["mpe", "pe"],
        default=["mpe", "pe"]
    )
    parser.add_argument(
        "--representations", nargs="+", choices=["pitch", "interval"],
        default=["pitch", "interval"]
    )
    parser.add_argument("--no-plots", action="store_true")

    # parse_known_args evita que los argumentos internos de Jupyter/IPython
    # (por ejemplo, -f kernel.json) provoquen SystemExit: 2. Fuera de un
    # kernel se conserva el comportamiento estricto para detectar errores.
    if "ipykernel" in sys.modules:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            print(
                "Aviso: se ignoraron argumentos internos de Jupyter/IPython: "
                + " ".join(map(str, unknown))
            )
    else:
        args = parser.parse_args(argv)

    if args.m_min < 2:
        parser.error("m-min debe ser al menos 2.")
    if args.m_max < args.m_min:
        parser.error("m-max debe ser mayor o igual que m-min.")
    if args.tau < 1:
        parser.error("tau debe ser al menos 1.")
    if args.n_shuffles < 2:
        parser.error("n-shuffles debe ser al menos 2 para estimar la desviación estándar.")

    csv_path = args.csv if args.csv is not None else default_csv_path()

    return Config(
        csv_path=csv_path,
        output_dir=args.output,
        work_id=args.work_id,
        m_values=tuple(range(args.m_min, args.m_max + 1)),
        tau=args.tau,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        length_mode=args.length_mode,
        entropy_types=tuple(args.entropy_types),
        representations=tuple(args.representations),
        make_plots=not args.no_plots,
    )


def safe_name(text: object) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9áéíóúüñ]+", "_", value)
    return value.strip("_")


def movement_label(value: object) -> str:
    try:
        number = int(float(value))
        return f"Movimiento {number}"
    except (TypeError, ValueError):
        return f"Movimiento {value}"


def parse_roles(raw_roles: object, n_voices: int) -> list[str]:
    if pd.isna(raw_roles):
        return VOICE_LABELS_FALLBACK[:n_voices]

    raw = [item.strip() for item in str(raw_roles).split("|")]
    replacements = {
        "vln i": "Violín I",
        "vln ii": "Violín II",
        "violin i": "Violín I",
        "violin ii": "Violín II",
        "viola": "Viola",
        "cello": "Violonchelo",
        "violoncello": "Violonchelo",
    }
    parsed = [replacements.get(item.lower(), item) for item in raw]
    if len(parsed) != n_voices:
        return VOICE_LABELS_FALLBACK[:n_voices]
    return parsed


def load_work(config: Config) -> pd.DataFrame:
    if not config.csv_path.exists():
        raise FileNotFoundError(
            f"No existe el CSV: {config.csv_path}\n"
            "Coloca 'quartet_subcorpus_all_movements(3).csv' en la misma "
            "carpeta que el script, o ejecuta el programa indicando la ruta:\n"
            "python analisis_mpe_debussy_op10_corregido.py "
            "--csv RUTA_AL_CSV"
        )

    df = pd.read_csv(config.csv_path)
    required = {"work_id", "movement_no", "voices_pitches", "roles"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    work = df[df["work_id"].astype(str).str.lower() == config.work_id.lower()].copy()
    if work.empty:
        candidates = sorted(df["work_id"].dropna().astype(str).unique())
        raise ValueError(
            f"No se encontró work_id={config.work_id!r}. "
            f"Ejemplos disponibles: {candidates[:12]}"
        )

    work = work.sort_values("movement_no").reset_index(drop=True)
    return work


def parse_voices(row: pd.Series) -> tuple[list[str], list[np.ndarray]]:
    try:
        data = ast.literal_eval(str(row["voices_pitches"]))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"No se pudo interpretar voices_pitches en movimiento {row['movement_no']}."
        ) from exc

    if not isinstance(data, list) or not all(isinstance(v, list) for v in data):
        raise TypeError("voices_pitches debe ser una lista de listas.")

    voices = [np.asarray(v, dtype=float) for v in data]
    roles = parse_roles(row.get("roles"), len(voices))

    for role, arr in zip(roles, voices):
        if arr.ndim != 1 or len(arr) == 0:
            raise ValueError(
                f"La voz {role} del movimiento {row['movement_no']} no es una serie 1D válida."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"La voz {role} del movimiento {row['movement_no']} contiene NaN o infinitos."
            )

    return roles, voices


def represent(pitches: np.ndarray, representation: str) -> np.ndarray:
    if representation == "pitch":
        return pitches.copy()
    if representation == "interval":
        return np.diff(pitches)
    raise ValueError(f"Representación desconocida: {representation}")


def center_crop(arr: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length debe ser positivo.")
    if len(arr) < target_length:
        raise ValueError("No se puede recortar a una longitud mayor que la serie.")
    start = (len(arr) - target_length) // 2
    return arr[start:start + target_length].copy()


def prepare_representations(
    voices: Sequence[np.ndarray],
    representation: str,
    length_mode: str,
) -> list[np.ndarray]:
    arrays = [represent(v, representation) for v in voices]
    if any(len(arr) < 2 for arr in arrays):
        raise ValueError(f"Alguna voz queda demasiado corta para {representation}.")

    if length_mode == "movement_min_center":
        target = min(map(len, arrays))
        arrays = [center_crop(arr, target) for arr in arrays]
    return arrays


def entropy_function(kind: str, tau: int) -> Callable[[np.ndarray, int], float]:
    if kind == "mpe":
        def func(arr: np.ndarray, m: int) -> float:
            return float(
                ml.modified_permutation_entropy(arr, m, tau=tau, norm=False)
            )
        return func

    if kind == "pe":
        def func(arr: np.ndarray, m: int) -> float:
            return float(ml.permutation_entropy(arr, m, tau=tau, norm=False))
        return func

    raise ValueError(f"Tipo de entropía desconocido: {kind}")


def validate_embedding_length(arr: np.ndarray, m: int, tau: int) -> int:
    n_vectors = len(arr) - (m - 1) * tau
    if n_vectors <= 0:
        raise ValueError(
            f"Serie de longitud {len(arr)} insuficiente para m={m}, tau={tau}."
        )
    return n_vectors


def sequence_metadata(arr: np.ndarray) -> dict[str, float | int]:
    adjacent_equal = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else np.nan
    zero_fraction = float(np.mean(arr == 0))
    return {
        "sequence_length": int(len(arr)),
        "n_unique_values": int(np.unique(arr).size),
        "adjacent_equal_fraction": adjacent_equal,
        "zero_fraction": zero_fraction,
        "series_mean": float(np.mean(arr)),
        "series_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        "series_min": float(np.min(arr)),
        "series_max": float(np.max(arr)),
    }


def compute_series_results(
    arr: np.ndarray,
    movement_no: object,
    voice_index: int,
    voice: str,
    representation: str,
    config: Config,
) -> list[dict[str, object]]:
    """Calcula observado y nulo. Las mismas permutaciones se usan para PE y mPE."""
    entropy_funcs = {
        kind: entropy_function(kind, config.tau) for kind in config.entropy_types
    }
    meta = sequence_metadata(arr)

    observed: dict[str, np.ndarray] = {}
    null_values: dict[str, np.ndarray] = {}

    for kind, func in entropy_funcs.items():
        values = []
        for m in config.m_values:
            validate_embedding_length(arr, m, config.tau)
            value = func(arr, m)
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"Resultado no finito para {kind}, movimiento={movement_no}, "
                    f"voz={voice}, representación={representation}, m={m}."
                )
            values.append(value)
        observed[kind] = np.asarray(values, dtype=float)
        null_values[kind] = np.empty(
            (config.n_shuffles, len(config.m_values)), dtype=float
        )

    rep_code = 0 if representation == "pitch" else 1
    seed_sequence = np.random.SeedSequence(
        [config.seed, int(float(movement_no)), voice_index, rep_code]
    )
    rng = np.random.default_rng(seed_sequence)

    iterator = tqdm(
        range(config.n_shuffles),
        desc=f"Mov. {movement_no} | {voice} | {representation}",
        leave=False,
    )
    for b in iterator:
        shuffled = rng.permutation(arr)
        for kind, func in entropy_funcs.items():
            for j, m in enumerate(config.m_values):
                value = func(shuffled, m)
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Nulo no finito para {kind}, movimiento={movement_no}, "
                        f"voz={voice}, representación={representation}, m={m}, shuffle={b}."
                    )
                null_values[kind][b, j] = value

    rows: list[dict[str, object]] = []
    for kind in config.entropy_types:
        null_mean = null_values[kind].mean(axis=0)
        null_std = null_values[kind].std(axis=0, ddof=1)

        for j, m in enumerate(config.m_values):
            obs = observed[kind][j]
            mu = null_mean[j]
            sigma = null_std[j]
            z_score = (obs - mu) / sigma if sigma > 0 else np.nan
            organization_z = -z_score if np.isfinite(z_score) else np.nan

            lower_count = int(np.count_nonzero(null_values[kind][:, j] <= obs))
            upper_count = int(np.count_nonzero(null_values[kind][:, j] >= obs))
            p_lower = (lower_count + 1) / (config.n_shuffles + 1)
            p_upper = (upper_count + 1) / (config.n_shuffles + 1)

            n_vectors = validate_embedding_length(arr, m, config.tau)
            factorial_m = math.factorial(m)
            windows_per_factorial = n_vectors / factorial_m

            row = {
                "work_id": config.work_id,
                "movement_no": movement_no,
                "movement": movement_label(movement_no),
                "voice_index": voice_index,
                "voice": voice,
                "representation": representation,
                "representation_label": REPRESENTATION_LABELS[representation],
                "entropy_type": kind,
                "entropy_label": ENTROPY_LABELS[kind],
                "m": m,
                "tau": config.tau,
                "observed_entropy": obs,
                "null_mean": mu,
                "null_std": sigma,
                "z_score": z_score,
                "organization_z": organization_z,
                "empirical_p_lower": p_lower,
                "empirical_p_upper": p_upper,
                "n_shuffles": config.n_shuffles,
                "n_vectors": n_vectors,
                "factorial_m": factorial_m,
                "windows_per_factorial": windows_per_factorial,
                "sparse_ordinal_space_heuristic": windows_per_factorial < 1.0,
                "length_mode": config.length_mode,
                **meta,
            }
            rows.append(row)

    return rows


def compute_all(work: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, movement_row in work.iterrows():
        roles, voices = parse_voices(movement_row)
        movement_no = movement_row["movement_no"]

        for representation in config.representations:
            arrays = prepare_representations(
                voices=voices,
                representation=representation,
                length_mode=config.length_mode,
            )
            for voice_index, (voice, arr) in enumerate(zip(roles, arrays)):
                rows.extend(
                    compute_series_results(
                        arr=arr,
                        movement_no=movement_no,
                        voice_index=voice_index,
                        voice=voice,
                        representation=representation,
                        config=config,
                    )
                )

    results = pd.DataFrame(rows)
    sort_cols = [
        "movement_no", "representation", "entropy_type", "voice_index", "m"
    ]
    return results.sort_values(sort_cols).reset_index(drop=True)


def detect_curve_crossings(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["movement_no", "movement", "representation", "entropy_type"]

    for keys, group in results.groupby(group_cols, sort=True):
        movement_no, movement, representation, entropy_type = keys
        voices = list(
            group[["voice_index", "voice"]]
            .drop_duplicates()
            .sort_values("voice_index")["voice"]
        )

        for voice_a, voice_b in itertools.combinations(voices, 2):
            a = group[group["voice"] == voice_a][["m", "organization_z"]]
            b = group[group["voice"] == voice_b][["m", "organization_z"]]
            merged = a.merge(b, on="m", suffixes=("_a", "_b")).sort_values("m")
            if len(merged) < 2:
                continue

            m_values = merged["m"].to_numpy(dtype=float)
            diff = (
                merged["organization_z_a"] - merged["organization_z_b"]
            ).to_numpy(dtype=float)

            for i in range(len(diff) - 1):
                d0, d1 = diff[i], diff[i + 1]
                m0, m1 = m_values[i], m_values[i + 1]
                if not (np.isfinite(d0) and np.isfinite(d1)):
                    continue

                crossing = False
                if d0 == 0:
                    crossing_m = m0
                    crossing = True
                elif d0 * d1 < 0:
                    crossing_m = m0 - d0 * (m1 - m0) / (d1 - d0)
                    crossing = True

                if crossing:
                    leader_before = voice_a if d0 > 0 else voice_b if d0 < 0 else "Empate"
                    leader_after = voice_a if d1 > 0 else voice_b if d1 < 0 else "Empate"
                    rows.append({
                        "movement_no": movement_no,
                        "movement": movement,
                        "representation": representation,
                        "entropy_type": entropy_type,
                        "voice_a": voice_a,
                        "voice_b": voice_b,
                        "m_left": m0,
                        "m_right": m1,
                        "crossing_m_estimate": crossing_m,
                        "difference_left_organization_z": d0,
                        "difference_right_organization_z": d1,
                        "more_organized_before": leader_before,
                        "more_organized_after": leader_after,
                    })

    columns = [
        "movement_no", "movement", "representation", "entropy_type",
        "voice_a", "voice_b", "m_left", "m_right", "crossing_m_estimate",
        "difference_left_organization_z", "difference_right_organization_z",
        "more_organized_before", "more_organized_after",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_rankings(results: pd.DataFrame) -> pd.DataFrame:
    rankings = results.copy()
    rankings["organization_rank_within_movement_m"] = rankings.groupby(
        ["movement_no", "representation", "entropy_type", "m"]
    )["organization_z"].rank(method="min", ascending=False)
    return rankings[[
        "movement_no", "movement", "representation", "entropy_type", "m",
        "voice_index", "voice", "observed_entropy", "z_score", "organization_z",
        "organization_rank_within_movement_m", "empirical_p_lower",
        "sequence_length", "n_vectors", "windows_per_factorial",
        "sparse_ordinal_space_heuristic",
    ]].sort_values([
        "movement_no", "representation", "entropy_type", "m",
        "organization_rank_within_movement_m", "voice_index"
    ])


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def movement_summary(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "movement_no", "movement", "representation", "entropy_type",
        "voice_index", "voice"
    ]
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(group_cols, sort=True):
        (
            movement_no, movement, representation, entropy_type,
            voice_index, voice
        ) = keys
        ordered = group.sort_values("m")
        x = ordered["m"].to_numpy(dtype=float)
        y = ordered["organization_z"].to_numpy(dtype=float)
        rows.append({
            "movement_no": movement_no,
            "movement": movement,
            "representation": representation,
            "entropy_type": entropy_type,
            "voice_index": voice_index,
            "voice": voice,
            "mean_organization_z_across_m": float(np.nanmean(y)),
            "median_organization_z_across_m": float(np.nanmedian(y)),
            "min_organization_z_across_m": float(np.nanmin(y)),
            "max_organization_z_across_m": float(np.nanmax(y)),
            "organization_z_slope_vs_m": linear_slope(x, y),
            "mean_observed_entropy": float(np.nanmean(ordered["observed_entropy"])),
            "mean_empirical_p_lower": float(np.nanmean(ordered["empirical_p_lower"])),
            "sequence_length": int(ordered["sequence_length"].iloc[0]),
            "n_m_values": int(np.count_nonzero(np.isfinite(y))),
        })

    summary = pd.DataFrame(rows)
    summary["mean_organization_rank_within_movement"] = summary.groupby(
        ["movement_no", "representation", "entropy_type"]
    )["mean_organization_z_across_m"].rank(method="min", ascending=False)
    return summary.sort_values([
        "representation", "entropy_type", "movement_no",
        "mean_organization_rank_within_movement", "voice_index"
    ]).reset_index(drop=True)


def spearman_from_four_values(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.nan
    x_rank = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def pitch_interval_comparison(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"pitch", "interval"}.issubset(set(results["representation"])):
        return pd.DataFrame(), pd.DataFrame()

    base_cols = [
        "movement_no", "movement", "entropy_type", "m", "voice_index", "voice"
    ]
    pitch = results[results["representation"] == "pitch"][
        base_cols + ["organization_z", "z_score"]
    ].rename(columns={
        "organization_z": "organization_z_pitch",
        "z_score": "z_score_pitch",
    })
    interval = results[results["representation"] == "interval"][
        base_cols + ["organization_z", "z_score"]
    ].rename(columns={
        "organization_z": "organization_z_interval",
        "z_score": "z_score_interval",
    })

    by_voice = pitch.merge(interval, on=base_cols, how="inner")
    by_voice["organization_rank_pitch"] = by_voice.groupby(
        ["movement_no", "entropy_type", "m"]
    )["organization_z_pitch"].rank(method="average", ascending=False)
    by_voice["organization_rank_interval"] = by_voice.groupby(
        ["movement_no", "entropy_type", "m"]
    )["organization_z_interval"].rank(method="average", ascending=False)
    by_voice["rank_interval_minus_pitch"] = (
        by_voice["organization_rank_interval"]
        - by_voice["organization_rank_pitch"]
    )
    by_voice["organization_z_interval_minus_pitch"] = (
        by_voice["organization_z_interval"]
        - by_voice["organization_z_pitch"]
    )

    agreement_rows: list[dict[str, object]] = []
    group_cols = ["movement_no", "movement", "entropy_type", "m"]
    for keys, group in by_voice.groupby(group_cols, sort=True):
        movement_no, movement, entropy_type, m = keys
        rho = spearman_from_four_values(
            group["organization_z_pitch"].to_numpy(dtype=float),
            group["organization_z_interval"].to_numpy(dtype=float),
        )
        agreement_rows.append({
            "movement_no": movement_no,
            "movement": movement,
            "entropy_type": entropy_type,
            "m": m,
            "spearman_voice_ranking_pitch_vs_interval": rho,
            "mean_absolute_rank_change": float(
                np.mean(np.abs(group["rank_interval_minus_pitch"]))
            ),
            "mean_absolute_organization_z_change": float(
                np.mean(np.abs(group["organization_z_interval_minus_pitch"]))
            ),
        })

    agreement = pd.DataFrame(agreement_rows)
    return (
        by_voice.sort_values(["movement_no", "entropy_type", "m", "voice_index"]),
        agreement.sort_values(["entropy_type", "movement_no", "m"]),
    )


def make_line_plot(
    group: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    horizontal_zero: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for _, voice_group in group.groupby(["voice_index", "voice"], sort=True):
        voice_group = voice_group.sort_values("m")
        line, = ax.plot(
            voice_group["m"], voice_group[y_col], marker="o",
            linewidth=1.8, label=voice_group["voice"].iloc[0]
        )
        sparse = voice_group["sparse_ordinal_space_heuristic"].astype(bool)
        if sparse.any():
            ax.scatter(
                voice_group.loc[sparse, "m"],
                voice_group.loc[sparse, y_col],
                marker="x", s=55, color=line.get_color(), zorder=3,
            )

    if horizontal_zero:
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Dimensión de patrón m")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(group["m"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_movement(results: pd.DataFrame, output_dir: Path) -> None:
    group_cols = [
        "movement_no", "movement", "representation", "representation_label",
        "entropy_type", "entropy_label"
    ]
    for keys, group in results.groupby(group_cols, sort=True):
        (
            movement_no, movement, representation, representation_label,
            entropy_type, entropy_label
        ) = keys
        stem = (
            f"mov_{int(float(movement_no)):02d}__{representation}__{entropy_type}"
        )

        make_line_plot(
            group=group,
            y_col="observed_entropy",
            ylabel=f"{entropy_label} sin normalizar",
            title=f"{movement}: {entropy_label} | {representation_label}",
            output_path=output_dir / "figures" / "raw_entropy" / f"{stem}.png",
        )
        make_line_plot(
            group=group,
            y_col="z_score",
            ylabel=r"$Z=(H_{obs}-\mu_{shuffle})/\sigma_{shuffle}$",
            title=(
                f"{movement}: Z de {entropy_label} | {representation_label}\n"
                "Z < 0: entropía menor que el shuffle"
            ),
            output_path=output_dir / "figures" / "z_score" / f"{stem}.png",
            horizontal_zero=True,
        )
        make_line_plot(
            group=group,
            y_col="organization_z",
            ylabel=r"$Z_{org}=-Z$",
            title=(
                f"{movement}: organización ordinal ({entropy_label}) | "
                f"{representation_label}\nZ_org > 0: entropía menor que el shuffle"
            ),
            output_path=(
                output_dir / "figures" / "organization_z" / f"{stem}.png"
            ),
            horizontal_zero=True,
        )


def plot_movement_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    group_cols = ["representation", "entropy_type"]
    for (representation, entropy_type), group in summary.groupby(group_cols, sort=True):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for _, voice_group in group.groupby(["voice_index", "voice"], sort=True):
            voice_group = voice_group.sort_values("movement_no")
            ax.plot(
                voice_group["movement_no"],
                voice_group["mean_organization_z_across_m"],
                marker="o", linewidth=1.8, label=voice_group["voice"].iloc[0]
            )
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.set_xlabel("Movimiento")
        ax.set_ylabel(r"Media de $Z_{org}$ sobre m")
        ax.set_title(
            f"Cambio entre movimientos | {ENTROPY_LABELS[entropy_type]} | "
            f"{REPRESENTATION_LABELS[representation]}"
        )
        ax.set_xticks(sorted(group["movement_no"].unique()))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = (
            output_dir / "figures" / "movement_summary" /
            f"movement_summary__{representation}__{entropy_type}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_pitch_interval_agreement(agreement: pd.DataFrame, output_dir: Path) -> None:
    if agreement.empty:
        return
    for entropy_type, group in agreement.groupby("entropy_type", sort=True):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for movement_no, movement_group in group.groupby("movement_no", sort=True):
            movement_group = movement_group.sort_values("m")
            ax.plot(
                movement_group["m"],
                movement_group["spearman_voice_ranking_pitch_vs_interval"],
                marker="o", linewidth=1.8,
                label=f"Movimiento {int(float(movement_no))}"
            )
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Dimensión de patrón m")
        ax.set_ylabel("Spearman entre rankings de las cuatro voces")
        ax.set_title(
            f"Acuerdo alturas–intervalos | {ENTROPY_LABELS[entropy_type]}"
        )
        ax.set_xticks(sorted(group["m"].unique()))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = (
            output_dir / "figures" / "pitch_interval_agreement" /
            f"pitch_interval_agreement__{entropy_type}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_outputs(
    results: pd.DataFrame,
    crossings: pd.DataFrame,
    rankings: pd.DataFrame,
    summary: pd.DataFrame,
    pitch_interval_by_voice: pd.DataFrame,
    pitch_interval_agreement: pd.DataFrame,
    config: Config,
) -> None:
    data_dir = config.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(data_dir / "debussy_entropy_zscores_long.csv", index=False)
    crossings.to_csv(data_dir / "curve_crossings_organization_z.csv", index=False)
    rankings.to_csv(data_dir / "voice_rankings_by_m.csv", index=False)
    summary.to_csv(data_dir / "movement_voice_summary.csv", index=False)
    pitch_interval_by_voice.to_csv(
        data_dir / "pitch_interval_voice_comparison.csv", index=False
    )
    pitch_interval_agreement.to_csv(
        data_dir / "pitch_interval_rank_agreement.csv", index=False
    )

    config_text = (
        f"csv={config.csv_path}\n"
        f"work_id={config.work_id}\n"
        f"m_values={list(config.m_values)}\n"
        f"tau={config.tau}\n"
        f"n_shuffles={config.n_shuffles}\n"
        f"seed={config.seed}\n"
        f"length_mode={config.length_mode}\n"
        f"entropy_types={list(config.entropy_types)}\n"
        f"representations={list(config.representations)}\n"
        f"z_score=(observed_entropy-null_mean)/null_std\n"
        f"organization_z=-z_score\n"
    )
    (config.output_dir / "analysis_config.txt").write_text(
        config_text, encoding="utf-8"
    )


def print_summary(summary: pd.DataFrame, crossings: pd.DataFrame) -> None:
    print("\n=== Ranking medio de organización por movimiento ===")
    display_cols = [
        "movement", "representation", "entropy_type", "voice",
        "mean_organization_z_across_m", "mean_organization_rank_within_movement"
    ]
    print(summary[display_cols].to_string(index=False))

    print("\n=== Cruces detectados en organization_z ===")
    if crossings.empty:
        print("No se detectaron cruces entre m consecutivos.")
    else:
        crossing_cols = [
            "movement", "representation", "entropy_type", "voice_a", "voice_b",
            "crossing_m_estimate", "more_organized_before", "more_organized_after"
        ]
        print(crossings[crossing_cols].to_string(index=False))


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo: {config.csv_path}")
    work = load_work(config)
    print(
        f"Obra: {config.work_id} | movimientos: {len(work)} | "
        f"m={config.m_values[0]}..{config.m_values[-1]} | "
        f"shuffles={config.n_shuffles} | length_mode={config.length_mode}"
    )

    results = compute_all(work, config)
    crossings = detect_curve_crossings(results)
    rankings = compute_rankings(results)
    summary = movement_summary(results)
    pitch_interval_by_voice, pitch_interval_agreement = (
        pitch_interval_comparison(results)
    )

    save_outputs(
        results=results,
        crossings=crossings,
        rankings=rankings,
        summary=summary,
        pitch_interval_by_voice=pitch_interval_by_voice,
        pitch_interval_agreement=pitch_interval_agreement,
        config=config,
    )

    if config.make_plots:
        plot_per_movement(results, config.output_dir)
        plot_movement_summary(summary, config.output_dir)
        plot_pitch_interval_agreement(pitch_interval_agreement, config.output_dir)

    print_summary(summary, crossings)
    print(f"\nResultados guardados en: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
