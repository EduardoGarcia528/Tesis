# -*- coding: utf-8 -*-
"""
Comparación lambda_MI vs kappa_C en un corpus de cuartetos.

Para cada voz de cada movimiento:
    1) Extrae la serie MIDI o la serie de intervalos melódicos.
    2) Determina K a partir de un C_{1,null} objetivo común.
    3) Calcula C_d observado y ajusta
           C_d = A_C exp(-lambda_C d)
       sobre d = FIT_D_MIN,...,FIT_D_MAX.
    4) Genera shuffles, calcula lambda_C,null para cada uno y define
           kappa_C = lambda_C,obs / <lambda_C,null>.
    5) Calcula la auto-información mutua
           I(tau) = I(X_t ; X_{t+tau}).
    6) Según MI_MODE, ajusta:
           "real": I_used(tau) = I_real(tau)
           "corr": I_used(tau) = I_real(tau) - <I_shuffle(tau)>
       con
           I_used(tau) = I_0 exp(-lambda_MI tau).
    7) Guarda un CSV por voz/movimiento, un CSV de correlaciones y figuras
       de lambda_MI vs kappa_C.

Notas importantes
-----------------
- En modo "intervals" se usan, por defecto, INTERVALOS CON SIGNO.
  Si se desea reproducir el abs() del script antiguo, usar
      INTERVAL_USE_ABS = True
- Para MI_MODE="corr", I_corr puede ser <= 0 en algunos retardos por
  fluctuaciones finitas. El ajuste semilog usa solamente puntos positivos
  dentro del rango de ajuste. Se guarda n_fit_MI para diagnosticarlo.
- El mismo conjunto de shuffles de cada voz se usa para construir el nulo
  de C_d y, cuando MI_MODE="corr", el piso de MI.

Requiere:
    numpy
    pandas
    matplotlib
    scipy
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
from scipy.stats import pearsonr, spearmanr

import mi_libreria as ml


# ============================================================
# CONFIGURACIÓN
# ============================================================

CSV_PATH = Path("quartet_subcorpus_all_movements.csv")
OUTPUT_DIR = Path("lambda_MI_vs_kappaC")
SUMMARY_CSV = OUTPUT_DIR / "resumen_lambda_MI_vs_kappaC.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "correlaciones_lambda_MI_vs_kappaC.csv"

# Serie analizada por AMBAS métricas: "midi" o "intervals".
ANALYSIS_MODE = "intervals"

# En intervalos, False = intervalos con signo (recomendado para tu análisis).
# True reproduce el np.abs(intervals) del script anterior.
INTERVAL_USE_ABS = False

# ----- C_d -----
MAX_GAMMA = 7
C1_NULL_TARGET = 0.5807
FIT_D_MIN = 1
FIT_D_MAX = 8

# ----- Auto-MI -----
# "real" -> ajusta I_real(tau)
# "corr" -> ajusta I_real(tau) - <I_shuffle(tau)>
MI_MODE = "real"

TAU_MIN = 1
TAU_MAX = 10
FIT_TAU_MIN = 1
FIT_TAU_MAX = 10

# Mínimo de puntos positivos necesario para aceptar el ajuste de MI.
MIN_MI_FIT_POINTS = 3

# ----- Nulo -----
N_SHUFFLES = 200
RNG_SEED = 12345

# ----- Salida -----
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

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 13,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)


# ============================================================
# LECTURA Y PREPROCESAMIENTO DEL CORPUS
# ============================================================


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s/]+", "_", text)
    return text.strip("_")


def parse_literal(value: Any) -> Any:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_voices(row: pd.Series) -> dict[str, np.ndarray]:
    """Lee voices_pitches y roles del CSV."""
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


def build_analysis_sequence(pitches: np.ndarray) -> np.ndarray:
    """Construye la serie usada tanto por C_d como por la auto-MI."""
    midi = np.asarray(pitches, dtype=float).ravel()
    midi = midi[np.isfinite(midi)]

    if ANALYSIS_MODE == "midi":
        return midi

    if ANALYSIS_MODE == "intervals":
        intervals = np.diff(midi)
        if INTERVAL_USE_ABS:
            intervals = np.abs(intervals)
        return intervals
    if ANALYSIS_MODE == "mod12":
        return np.mod(midi, 12)

    raise ValueError("ANALYSIS_MODE debe ser 'midi' o 'intervals'.")


# ============================================================
# C1_NULL Y C_d
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
    """Elige el mayor K tal que C1_null(K) <= target."""
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


def compute_C(array: np.ndarray, K: int) -> np.ndarray:
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


def fit_C_exponential(C: np.ndarray) -> dict[str, Any]:
    """Ajusta log(C_d) = log(A) - lambda_C d."""
    C = np.asarray(C, dtype=float).ravel()
    d_all = np.arange(len(C), dtype=float)

    mask = (
        (d_all >= FIT_D_MIN)
        & (d_all <= FIT_D_MAX)
        & np.isfinite(C)
        & (C > 0)
    )

    x = d_all[mask]
    y = np.log(C[mask])

    if len(x) < 2:
        return {
            "lambda": np.nan,
            "A": np.nan,
            "r2": np.nan,
            "rmse_log": np.nan,
            "n_fit": len(x),
        }

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "lambda": -float(slope),
        "A": float(np.exp(intercept)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse_log": float(np.sqrt(np.mean((y - y_hat) ** 2))),
        "n_fit": len(x),
    }


# ============================================================
# AUTO-MUTUAL INFORMATION DISCRETA
# ============================================================


def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """MI empírica discreta I(X;Y), en bits."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("x e y deben tener la misma longitud.")

    n = x.size
    if n == 0:
        return np.nan

    _, x_inv = np.unique(x, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)

    nx = int(x_inv.max()) + 1
    ny = int(y_inv.max()) + 1

    joint = np.zeros((nx, ny), dtype=np.int64)
    np.add.at(joint, (x_inv, y_inv), 1)

    pxy = joint.astype(float) / float(n)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    denom = px[:, None] * py[None, :]

    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))


def compute_auto_mi(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula I(tau)=I(X_t;X_{t+tau}) para tau=TAU_MIN,...,TAU_MAX."""
    array = np.asarray(array, dtype=float).ravel()
    taus = np.arange(TAU_MIN, TAU_MAX + 1, dtype=int)
    values = np.full(len(taus), np.nan, dtype=float)

    for i, tau in enumerate(taus):
        if array.size <= tau:
            continue
        values[i] = mutual_information_discrete(array[:-tau], array[tau:])

    return taus, values


def fit_MI_exponential(
    taus: np.ndarray,
    I_values: np.ndarray,
) -> dict[str, Any]:
    """
    Ajusta
        I(tau) = I0 exp(-lambda_MI tau)
    en semilog.

    Para I_corr se descartan automáticamente valores <= 0.
    """
    taus = np.asarray(taus, dtype=float).ravel()
    I_values = np.asarray(I_values, dtype=float).ravel()

    mask = (
        (taus >= FIT_TAU_MIN)
        & (taus <= FIT_TAU_MAX)
        & np.isfinite(I_values)
        & (I_values > 0)
    )

    x = taus[mask]
    y_raw = I_values[mask]

    if len(x) < MIN_MI_FIT_POINTS:
        return {
            "lambda": np.nan,
            "I0": np.nan,
            "r2": np.nan,
            "rmse_log": np.nan,
            "n_fit": len(x),
            "taus_used": x.astype(int),
        }

    y = np.log(y_raw)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "lambda": -float(slope),
        "I0": float(np.exp(intercept)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse_log": float(np.sqrt(np.mean((y - y_hat) ** 2))),
        "n_fit": len(x),
        "taus_used": x.astype(int),
    }


# ============================================================
# SHUFFLE: NULO DE C_d Y, OPCIONALMENTE, DE MI
# ============================================================


def compute_shuffle_statistics(
    array: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Usa los mismos shuffles para el nulo de C_d y para el sesgo de MI.
    """
    array = np.asarray(array, dtype=float).ravel()

    lambda_C_null = np.full(N_SHUFFLES, np.nan, dtype=float)

    need_mi_null = MI_MODE == "corr"
    n_taus = TAU_MAX - TAU_MIN + 1
    I_null_all = (
        np.full((N_SHUFFLES, n_taus), np.nan, dtype=float)
        if need_mi_null
        else None
    )

    for iteration in range(N_SHUFFLES):
        shuffled = rng.permutation(array)

        C_null = compute_C(shuffled, K)
        fit_null = fit_C_exponential(C_null)
        lambda_C_null[iteration] = fit_null["lambda"]

        if need_mi_null:
            _, I_null = compute_auto_mi(shuffled)
            I_null_all[iteration] = I_null

    result: dict[str, Any] = {
        "lambda_C_null_mean": float(np.nanmean(lambda_C_null)),
        "lambda_C_null_std": float(np.nanstd(lambda_C_null, ddof=1)),
    }

    if need_mi_null:
        result["I_null_mean"] = np.nanmean(I_null_all, axis=0)
        result["I_null_std"] = np.nanstd(I_null_all, axis=0, ddof=1)
    else:
        result["I_null_mean"] = np.full(n_taus, np.nan)
        result["I_null_std"] = np.full(n_taus, np.nan)

    return result


# ============================================================
# CORRELACIONES Y GRÁFICAS
# ============================================================


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {
            "n": len(x),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pr = pearsonr(x, y)
    sr = spearmanr(x, y)

    return {
        "n": len(x),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Global
    corr = safe_correlations(df["lambda_MI"], df["kappa_C"])
    rows.append({"group_type": "all", "group": "all", **corr})

    # Por voz
    for voice, sub in df.groupby("voice"):
        corr = safe_correlations(sub["lambda_MI"], sub["kappa_C"])
        rows.append({"group_type": "voice", "group": voice, **corr})

    # Por compositor
    for composer, sub in df.groupby("composer"):
        corr = safe_correlations(sub["lambda_MI"], sub["kappa_C"])
        rows.append({"group_type": "composer", "group": composer, **corr})

    return pd.DataFrame(rows)


def plot_overall_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    for voice in VOICE_ORDER:
        sub = df[df["voice"] == voice]
        if sub.empty:
            continue
        ax.scatter(
            sub["lambda_MI"],
            sub["kappa_C"],
            s=36,
            alpha=0.72,
            label=VOICE_TITLES[voice],
            color=VOICE_COLORS[voice],
            edgecolors="none",
        )

    valid = df[["lambda_MI", "kappa_C"]].dropna()
    corr = safe_correlations(valid["lambda_MI"], valid["kappa_C"])

    if len(valid) >= 2:
        x = valid["lambda_MI"].to_numpy(float)
        y = valid["kappa_C"].to_numpy(float)
        if np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(xx, intercept + slope * xx, "k--", lw=1.4, alpha=0.8)

    ax.text(
        0.03,
        0.97,
        (
            f"N = {corr['n']}\n"
            f"Pearson r = {corr['pearson_r']:.3f}\n"
            f"Spearman ρ = {corr['spearman_rho']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel(r"Exponente de decaimiento de auto-MI, $\lambda_{MI}$")
    ax.set_ylabel(r"$\kappa_C = \lambda_C / \langle\lambda_{C,null}\rangle$")
    ax.set_title(
        rf"$\lambda_{{MI}}$ vs $\kappa_C$ | {ANALYSIS_MODE} | MI={MI_MODE}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / "lambda_MI_vs_kappaC_scatter_global.png",
            dpi=PNG_DPI,
            bbox_inches="tight",
        )
    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / "lambda_MI_vs_kappaC_scatter_global.pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_scatter_by_voice(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, voice in zip(axes, VOICE_ORDER):
        sub = df[df["voice"] == voice].copy()
        valid = sub[["lambda_MI", "kappa_C"]].dropna()

        ax.scatter(
            valid["lambda_MI"],
            valid["kappa_C"],
            s=34,
            alpha=0.72,
            color=VOICE_COLORS[voice],
            edgecolors="none",
        )

        corr = safe_correlations(valid["lambda_MI"], valid["kappa_C"])

        if len(valid) >= 2:
            x = valid["lambda_MI"].to_numpy(float)
            y = valid["kappa_C"].to_numpy(float)
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xx = np.linspace(np.min(x), np.max(x), 200)
                ax.plot(xx, intercept + slope * xx, "k--", lw=1.2, alpha=0.8)

        ax.set_title(VOICE_TITLES[voice])
        ax.text(
            0.04,
            0.96,
            (
                f"N={corr['n']}\n"
                f"r={corr['pearson_r']:.3f}\n"
                f"ρ={corr['spearman_rho']:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
        )
        ax.grid(alpha=0.25)

    fig.supxlabel(r"Exponente de decaimiento de auto-MI, $\lambda_{MI}$")
    fig.supylabel(r"$\kappa_C = \lambda_C / \langle\lambda_{C,null}\rangle$")
    fig.suptitle(
        rf"$\lambda_{{MI}}$ vs $\kappa_C$ por voz | {ANALYSIS_MODE} | MI={MI_MODE}",
        y=0.98,
    )
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / "lambda_MI_vs_kappaC_scatter_por_voz.png",
            dpi=PNG_DPI,
            bbox_inches="tight",
        )
    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / "lambda_MI_vs_kappaC_scatter_por_voz.pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================


def main() -> None:
    if MI_MODE not in {"real", "corr"}:
        raise ValueError("MI_MODE debe ser 'real' o 'corr'.")

    if ANALYSIS_MODE not in {"midi", "intervals","mod12"}:
        raise ValueError("ANALYSIS_MODE debe ser 'midi' o 'intervals' o 'mod12'.")

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
    missing = required_columns.difference(corpus.columns)
    if missing:
        raise KeyError(f"Faltan columnas necesarias: {sorted(missing)}")

    rng_master = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []

    total = len(corpus)

    for row_index, row in corpus.iterrows():
        composer = (
            str(row["composer_canonical"])
            if "composer_canonical" in corpus.columns
            and pd.notna(row["composer_canonical"])
            else str(row["composer"])
        )
        work_title = str(row["work_title"])
        movement_no = row["movement_no"]

        print(
            f"[{row_index + 1:03d}/{total:03d}] "
            f"{composer} | {work_title} | movimiento {movement_no}"
        )

        try:
            voices = parse_voices(row)
        except Exception as exc:
            print(f"    ERROR al leer voces: {exc}")
            continue

        for voice in VOICE_ORDER:
            sequence = build_analysis_sequence(voices[voice])

            min_required = max(FIT_D_MAX + 2, TAU_MAX + 2)
            if sequence.size < min_required:
                print(
                    f"    ADVERTENCIA: {voice} tiene solo {sequence.size} datos; "
                    "se omite."
                )
                continue

            try:
                # ---------------- C_d observado ----------------
                K, n_unique, C1_null_exact = determine_K_from_C1_null(
                    sequence,
                    C1_NULL_TARGET,
                )

                C_observed = compute_C(sequence, K)
                fit_C_obs = fit_C_exponential(C_observed)

                # ---------------- MI real ----------------
                taus, I_real = compute_auto_mi(sequence)

                # ---------------- Shuffles ----------------
                local_seed = int(rng_master.integers(0, 2**32 - 1))
                local_rng = np.random.default_rng(local_seed)
                null = compute_shuffle_statistics(sequence, K, local_rng)

                lambda_C_null_mean = null["lambda_C_null_mean"]
                kappa_C = (
                    fit_C_obs["lambda"] / lambda_C_null_mean
                    if np.isfinite(fit_C_obs["lambda"])
                    and np.isfinite(lambda_C_null_mean)
                    and lambda_C_null_mean != 0
                    else np.nan
                )

                # ---------------- MI usada ----------------
                I_null_mean = null["I_null_mean"]
                I_null_std = null["I_null_std"]
                I_corr = I_real - I_null_mean if MI_MODE == "corr" else np.full_like(I_real, np.nan)
                I_used = I_real if MI_MODE == "real" else I_corr

                fit_MI = fit_MI_exponential(taus, I_used)

            except Exception as exc:
                print(f"    ERROR en {voice}: {exc}")
                continue

            result: dict[str, Any] = {
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
                "interval_use_abs": INTERVAL_USE_ABS,
                "sequence_length": len(sequence),
                "n_notes_original": len(voices[voice]),
                "n_unique": n_unique,
                "K": K,
                "C1_null_target": C1_NULL_TARGET,
                "C1_null_exact": C1_null_exact,
                "fit_d_min": FIT_D_MIN,
                "fit_d_max": FIT_D_MAX,
                "lambda_C_observed": fit_C_obs["lambda"],
                "A_C_observed": fit_C_obs["A"],
                "r2_C_observed": fit_C_obs["r2"],
                "rmse_log_C_observed": fit_C_obs["rmse_log"],
                "lambda_C_null_mean": lambda_C_null_mean,
                "lambda_C_null_std": null["lambda_C_null_std"],
                "kappa_C": kappa_C,
                "MI_mode": MI_MODE,
                "tau_min": TAU_MIN,
                "tau_max": TAU_MAX,
                "fit_tau_min": FIT_TAU_MIN,
                "fit_tau_max": FIT_TAU_MAX,
                "lambda_MI": fit_MI["lambda"],
                "I0_fit": fit_MI["I0"],
                "r2_MI": fit_MI["r2"],
                "rmse_log_MI": fit_MI["rmse_log"],
                "n_fit_MI": fit_MI["n_fit"],
                "taus_used_MI": "|".join(map(str, fit_MI["taus_used"])),
                "n_shuffles": N_SHUFFLES,
                "rng_local_seed": local_seed,
            }

            # Guardar C_d observado.
            for d, value in enumerate(C_observed):
                result[f"C_observed_{d}"] = value

            # Guardar las curvas de MI lag por lag.
            for tau, real, null_mean, null_std, corr, used in zip(
                taus,
                I_real,
                I_null_mean,
                I_null_std,
                I_corr,
                I_used,
            ):
                result[f"I_real_{tau}"] = real
                result[f"I_null_mean_{tau}"] = null_mean
                result[f"I_null_std_{tau}"] = null_std
                result[f"I_corr_{tau}"] = corr
                result[f"I_used_{tau}"] = used

            rows.append(result)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    if summary_df.empty:
        print("\nNo se obtuvieron resultados válidos.")
        return

    correlations_df = build_correlation_table(summary_df)
    correlations_df.to_csv(CORRELATIONS_CSV, index=False)

    plot_overall_scatter(summary_df)
    plot_scatter_by_voice(summary_df)

    print("\nProceso terminado.")
    print(f"Resumen: {SUMMARY_CSV.resolve()}")
    print(f"Correlaciones: {CORRELATIONS_CSV.resolve()}")
    print(f"Figuras: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
