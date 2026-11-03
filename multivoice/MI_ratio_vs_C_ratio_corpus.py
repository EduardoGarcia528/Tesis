# -*- coding: utf-8 -*-
"""
Comparación entre el cociente de auto-información mutua I(1)/I(2)
y el cociente C_1/C_2 en un corpus de cuartetos.

Para cada voz de cada movimiento:
    1) Extrae la serie MIDI o la serie de intervalos melódicos.
    2) Determina K a partir de un C_{1,null} objetivo común.
    3) Calcula C_1 y C_2 observados y define

           R_C = C_1 / C_2.

    4) Genera shuffles con el mismo K y calcula, para cada shuffle,

           R_C,null = C_{1,null} / C_{2,null}.

       Como el análisis natural es semilogarítmico, se usa la media
       geométrica del cociente nulo:

           R_C,null^g = exp(<log R_C,null>).

       Se define entonces el cociente corregido por el nulo

           R_C^* = R_C / R_C,null^g,

       de modo que exactamente

           log R_C^* = log R_C - <log R_C,null>.

    5) Calcula la auto-MI discreta en tau=1 y tau=2:

           I(tau) = I(X_t ; X_{t+tau}),

       y el cociente del artículo

           R_MI = I(1) / I(2).

       Por defecto MI_RATIO_MODE="real", que reproduce la definición
       del artículo. Opcionalmente puede usarse "corr":

           I_corr(tau) = I_real(tau) - <I_shuffle(tau)>,

       y construir R_MI con I_corr(1)/I_corr(2). Esta opción es más
       delicada porque I_corr puede ser <= 0.

    6) Compara cuatro pares:

           R_MI              vs R_C
           R_MI              vs R_C^*
           log(R_MI)         vs log(R_C)
           log(R_MI)         vs log(R_C^*)

       calculando Pearson y Spearman globalmente, por voz y por compositor.

Salidas:
    - resumen_MIratio_vs_Cratio.csv
    - correlaciones_MIratio_vs_Cratio.csv
    - MIratio_vs_Cratio_global_4paneles.png
    - log_MIratio_vs_log_Cratio_corregido_por_voz.png

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
OUTPUT_DIR = Path("MI_ratio_vs_C_ratio")
SUMMARY_CSV = OUTPUT_DIR / "resumen_MIratio_vs_Cratio.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "correlaciones_MIratio_vs_Cratio.csv"

# Serie usada por AMBAS métricas: "midi" o "intervals".
ANALYSIS_MODE = "intervals"

# Solo afecta ANALYSIS_MODE="intervals".
# False = intervalos con signo.
# True  = valores absolutos de los intervalos.
INTERVAL_USE_ABS = False

# ------------------------------------------------------------
# Parámetro de C_d
# ------------------------------------------------------------
# Cada voz determina el K entero más grande cuyo C_{1,null} exacto
# no rebasa este objetivo.
C1_NULL_TARGET = 0.5807

# Para obtener C_0, C_1 y C_2 basta max_gamma=1 en
# gamma_index_rank_ties, de acuerdo con la convención usada en
# Cd_semilog_C1null_corpus.py.
MAX_GAMMA_RATIO = 1

# ------------------------------------------------------------
# Auto-MI
# ------------------------------------------------------------
# "real" = I(1)/I(2), exactamente la razón utilizada en el artículo.
# "corr" = I_corr(1)/I_corr(2), con corrección de shuffle.
MI_RATIO_MODE = "real"

# ------------------------------------------------------------
# Nulo por shuffle
# ------------------------------------------------------------
N_SHUFFLES = 200
RNG_SEED = 12345

# ------------------------------------------------------------
# Salida gráfica
# ------------------------------------------------------------
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
        "axes.titlesize": 12.0,
        "axes.labelsize": 12.5,
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
    """Convierte texto en una cadena segura para nombres de archivo."""
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
    Lee voices_pitches y roles, y devuelve
        nombre de voz -> secuencia MIDI.
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


def build_analysis_sequence(pitches: np.ndarray) -> np.ndarray:
    """Construye la misma serie que será usada por C_d y por la auto-MI."""
    midi = np.asarray(pitches, dtype=float).ravel()
    midi = midi[np.isfinite(midi)]

    if ANALYSIS_MODE == "midi":
        return midi

    if ANALYSIS_MODE == "intervals":
        intervals = np.diff(midi)
        if INTERVAL_USE_ABS:
            intervals = np.abs(intervals)
        return intervals

    raise ValueError("ANALYSIS_MODE debe ser 'midi' o 'intervals'.")


# ============================================================
# C1_NULL, K Y C_1/C_2
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
    Determina K con el mismo criterio de Cd_semilog_C1null_corpus.py:
    el mayor K entero tal que C1_null(K) <= target.
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


def compute_C1_C2(array: np.ndarray, K: int) -> tuple[float, float]:
    """Calcula únicamente C_1 y C_2 mediante gamma_index_rank_ties."""
    C, _ = ml.gamma_index_rank_ties(
        np.asarray(array, dtype=float),
        max_gamma=MAX_GAMMA_RATIO,
        K=int(K),
    )

    C = np.asarray(C, dtype=float).ravel()

    if len(C) < 3:
        raise ValueError(
            "gamma_index_rank_ties no devolvió C_0, C_1 y C_2. "
            f"Longitud recibida: {len(C)}"
        )

    return float(C[1]), float(C[2])


def safe_ratio(numerator: float, denominator: float) -> float:
    """Cociente finito si el denominador es positivo y finito."""
    if (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0
    ):
        return float(numerator / denominator)
    return np.nan


def safe_log(value: float) -> float:
    """Logaritmo natural de un valor estrictamente positivo."""
    if np.isfinite(value) and value > 0:
        return float(np.log(value))
    return np.nan


# ============================================================
# AUTO-MUTUAL INFORMATION DISCRETA
# ============================================================


def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimador empírico discreto (plug-in / máxima verosimilitud) de MI,
    en bits:

        I(X;Y) = sum p(x,y) log2[p(x,y)/(p(x)p(y))].
    """
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


def compute_I1_I2(array: np.ndarray) -> tuple[float, float]:
    """Calcula I(1) e I(2) de la misma serie."""
    array = np.asarray(array, dtype=float).ravel()

    if array.size <= 2:
        return np.nan, np.nan

    I1 = mutual_information_discrete(array[:-1], array[1:])
    I2 = mutual_information_discrete(array[:-2], array[2:])
    return float(I1), float(I2)


# ============================================================
# NULO POR SHUFFLE
# ============================================================


def compute_shuffle_statistics(
    array: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Usa los mismos shuffles para:
      - el nulo del cociente C_1/C_2;
      - el piso de I(1) e I(2), necesario si MI_RATIO_MODE="corr".
    """
    array = np.asarray(array, dtype=float).ravel()

    C_ratio_null = np.full(N_SHUFFLES, np.nan, dtype=float)
    I1_null = np.full(N_SHUFFLES, np.nan, dtype=float)
    I2_null = np.full(N_SHUFFLES, np.nan, dtype=float)

    for iteration in range(N_SHUFFLES):
        shuffled = rng.permutation(array)

        C1_null, C2_null = compute_C1_C2(shuffled, K)
        C_ratio_null[iteration] = safe_ratio(C1_null, C2_null)

        I1_s, I2_s = compute_I1_I2(shuffled)
        I1_null[iteration] = I1_s
        I2_null[iteration] = I2_s

    valid_C_ratio = (
        np.isfinite(C_ratio_null)
        & (C_ratio_null > 0)
    )

    if np.any(valid_C_ratio):
        log_C_ratio_null = np.log(C_ratio_null[valid_C_ratio])
        C_ratio_null_geomean = float(np.exp(np.mean(log_C_ratio_null)))
        log_C_ratio_null_mean = float(np.mean(log_C_ratio_null))
    else:
        C_ratio_null_geomean = np.nan
        log_C_ratio_null_mean = np.nan

    return {
        "C_ratio_null_arithmetic_mean": float(np.nanmean(C_ratio_null)),
        "C_ratio_null_geometric_mean": C_ratio_null_geomean,
        "C_ratio_null_std": float(np.nanstd(C_ratio_null, ddof=1)),
        "log_C_ratio_null_mean": log_C_ratio_null_mean,
        "I1_null_mean": float(np.nanmean(I1_null)),
        "I1_null_std": float(np.nanstd(I1_null, ddof=1)),
        "I2_null_mean": float(np.nanmean(I2_null)),
        "I2_null_std": float(np.nanstd(I2_null, ddof=1)),
    }


# ============================================================
# CORRELACIONES
# ============================================================


COMPARISONS = {
    "ratio_raw": {
        "x": "MI_ratio",
        "y": "C_ratio",
        "x_label": r"$I(1)/I(2)$",
        "y_label": r"$C_1/C_2$",
        "title": "Cocientes directos",
    },
    "ratio_C_corrected": {
        "x": "MI_ratio",
        "y": "C_ratio_corrected",
        "x_label": r"$I(1)/I(2)$",
        "y_label": r"$(C_1/C_2)/R_{C,\mathrm{null}}^{g}$",
        "title": "Cociente de C corregido por shuffle",
    },
    "log_raw": {
        "x": "log_MI_ratio",
        "y": "log_C_ratio",
        "x_label": r"$\ln[I(1)/I(2)]$",
        "y_label": r"$\ln(C_1/C_2)$",
        "title": "Pendientes locales",
    },
    "log_C_corrected": {
        "x": "log_MI_ratio",
        "y": "log_C_ratio_corrected",
        "x_label": r"$\ln[I(1)/I(2)]$",
        "y_label": r"$\ln(C_1/C_2)-\langle\ln R_{C,\mathrm{null}}\rangle$",
        "title": "Pendiente local de C corregida por shuffle",
    },
}


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "n": int(len(x)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pr = pearsonr(x, y)
    sr = spearmanr(x, y)

    return {
        "n": int(len(x)),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Correlaciones para las cuatro comparaciones, global/voz/compositor."""
    rows: list[dict[str, Any]] = []

    group_specs: list[tuple[str, str, pd.DataFrame]] = [
        ("all", "all", df),
    ]

    for voice, sub in df.groupby("voice"):
        group_specs.append(("voice", str(voice), sub))

    for composer, sub in df.groupby("composer"):
        group_specs.append(("composer", str(composer), sub))

    for comparison_name, spec in COMPARISONS.items():
        for group_type, group, sub in group_specs:
            corr = safe_correlations(sub[spec["x"]], sub[spec["y"]])
            rows.append(
                {
                    "comparison": comparison_name,
                    "x_variable": spec["x"],
                    "y_variable": spec["y"],
                    "group_type": group_type,
                    "group": group,
                    **corr,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# GRÁFICAS
# ============================================================


def add_scatter_and_fit(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by_voice: bool = True,
) -> dict[str, float]:
    """Scatter, recta OLS visual y anotación de correlaciones."""
    if color_by_voice:
        for voice in VOICE_ORDER:
            sub = df[df["voice"] == voice]
            valid = sub[[x_col, y_col]].dropna()
            if valid.empty:
                continue

            ax.scatter(
                valid[x_col],
                valid[y_col],
                s=30,
                alpha=0.68,
                color=VOICE_COLORS[voice],
                edgecolors="none",
                label=VOICE_TITLES[voice],
            )
    else:
        valid = df[[x_col, y_col]].dropna()
        ax.scatter(
            valid[x_col],
            valid[y_col],
            s=30,
            alpha=0.70,
            edgecolors="none",
        )

    valid = df[[x_col, y_col]].dropna()
    corr = safe_correlations(valid[x_col], valid[y_col])

    if len(valid) >= 2:
        x = valid[x_col].to_numpy(float)
        y = valid[y_col].to_numpy(float)
        if np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(xx, intercept + slope * xx, "k--", lw=1.25, alpha=0.80)

    ax.text(
        0.03,
        0.97,
        (
            f"N={corr['n']}\n"
            f"Pearson r={corr['pearson_r']:.3f}\n"
            f"Spearman ρ={corr['spearman_rho']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.88,
        },
    )

    ax.grid(alpha=0.25)
    return corr


def plot_global_four_panel(df: pd.DataFrame) -> None:
    """Figura global con las cuatro comparaciones propuestas."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axes = axes.ravel()

    for ax, (_, spec) in zip(axes, COMPARISONS.items()):
        add_scatter_and_fit(ax, df, spec["x"], spec["y"], color_by_voice=True)
        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.set_title(spec["title"])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )

    fig.suptitle(
        f"Auto-MI vs persistencia local de $C_d$ | {ANALYSIS_MODE} | "
        f"MI ratio={MI_RATIO_MODE} | $C_{{1,null}}^*={C1_NULL_TARGET:.5f}$",
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.925))

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / "MIratio_vs_Cratio_global_4paneles.png",
            dpi=PNG_DPI,
            bbox_inches="tight",
        )
    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / "MIratio_vs_Cratio_global_4paneles.pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_corrected_log_by_voice(df: pd.DataFrame) -> None:
    """
    Figura 2x2 por voz para la comparación que considero más interpretable:

        ln[I(1)/I(2)] vs ln(C_1/C_2) - <ln R_C,null>.
    """
    x_col = "log_MI_ratio"
    y_col = "log_C_ratio_corrected"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, voice in zip(axes, VOICE_ORDER):
        sub = df[df["voice"] == voice].copy()
        add_scatter_and_fit(
            ax,
            sub,
            x_col,
            y_col,
            color_by_voice=False,
        )
        ax.set_title(VOICE_TITLES[voice])

    fig.supxlabel(r"$\ln[I(1)/I(2)]$")
    fig.supylabel(
        r"$\ln(C_1/C_2)-\langle\ln(C_{1,null}/C_{2,null})\rangle$"
    )
    fig.suptitle(
        f"Comparación local corregida por shuffle | {ANALYSIS_MODE} | "
        f"MI ratio={MI_RATIO_MODE}",
        y=0.98,
    )
    fig.tight_layout(rect=(0.035, 0.035, 1.0, 0.95))

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / "log_MIratio_vs_log_Cratio_corregido_por_voz.png",
            dpi=PNG_DPI,
            bbox_inches="tight",
        )
    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / "log_MIratio_vs_log_Cratio_corregido_por_voz.pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================


def main() -> None:
    if ANALYSIS_MODE not in {"midi", "intervals"}:
        raise ValueError("ANALYSIS_MODE debe ser 'midi' o 'intervals'.")

    if MI_RATIO_MODE not in {"real", "corr"}:
        raise ValueError("MI_RATIO_MODE debe ser 'real' o 'corr'.")

    if N_SHUFFLES < 1:
        raise ValueError("N_SHUFFLES debe ser al menos 1.")

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

            if sequence.size < 4:
                print(
                    f"    ADVERTENCIA: {voice} tiene solo {sequence.size} datos; "
                    "se omite."
                )
                continue

            try:
                # ------------------------------------------------
                # C_1/C_2 observado
                # ------------------------------------------------
                K, n_unique, C1_null_exact = determine_K_from_C1_null(
                    sequence,
                    C1_NULL_TARGET,
                )

                C1, C2 = compute_C1_C2(sequence, K)
                C_ratio = safe_ratio(C1, C2)
                log_C_ratio = safe_log(C_ratio)

                # ------------------------------------------------
                # I(1)/I(2) observado
                # ------------------------------------------------
                I1_real, I2_real = compute_I1_I2(sequence)

                # ------------------------------------------------
                # Shuffles comunes para C y MI
                # ------------------------------------------------
                local_seed = int(rng_master.integers(0, 2**32 - 1))
                local_rng = np.random.default_rng(local_seed)
                null = compute_shuffle_statistics(sequence, K, local_rng)

                # ------------------------------------------------
                # C corregido por shuffle
                # ------------------------------------------------
                C_ratio_null_g = null["C_ratio_null_geometric_mean"]
                C_ratio_corrected = safe_ratio(C_ratio, C_ratio_null_g)

                # Esta forma es exactamente equivalente al cociente anterior
                # porque C_ratio_null_g = exp(<log R_C,null>).
                log_C_ratio_corrected = (
                    log_C_ratio - null["log_C_ratio_null_mean"]
                    if np.isfinite(log_C_ratio)
                    and np.isfinite(null["log_C_ratio_null_mean"])
                    else np.nan
                )

                # ------------------------------------------------
                # Elegir MI real (paper) o corregida por shuffle
                # ------------------------------------------------
                I1_corr = I1_real - null["I1_null_mean"]
                I2_corr = I2_real - null["I2_null_mean"]

                if MI_RATIO_MODE == "real":
                    I1_used = I1_real
                    I2_used = I2_real
                else:
                    I1_used = I1_corr
                    I2_used = I2_corr

                MI_ratio = safe_ratio(I1_used, I2_used)
                log_MI_ratio = safe_log(MI_ratio)

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
                "C1": C1,
                "C2": C2,
                "C_ratio": C_ratio,
                "log_C_ratio": log_C_ratio,
                "C_ratio_null_arithmetic_mean": null[
                    "C_ratio_null_arithmetic_mean"
                ],
                "C_ratio_null_geometric_mean": C_ratio_null_g,
                "C_ratio_null_std": null["C_ratio_null_std"],
                "log_C_ratio_null_mean": null["log_C_ratio_null_mean"],
                "C_ratio_corrected": C_ratio_corrected,
                "log_C_ratio_corrected": log_C_ratio_corrected,
                "MI_ratio_mode": MI_RATIO_MODE,
                "I1_real": I1_real,
                "I2_real": I2_real,
                "I1_null_mean": null["I1_null_mean"],
                "I1_null_std": null["I1_null_std"],
                "I2_null_mean": null["I2_null_mean"],
                "I2_null_std": null["I2_null_std"],
                "I1_corr": I1_corr,
                "I2_corr": I2_corr,
                "I1_used": I1_used,
                "I2_used": I2_used,
                "MI_ratio": MI_ratio,
                "log_MI_ratio": log_MI_ratio,
                "n_shuffles": N_SHUFFLES,
                "rng_local_seed": local_seed,
            }

            rows.append(result)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    if summary_df.empty:
        print("\nNo se obtuvieron resultados válidos.")
        return

    correlations_df = build_correlation_table(summary_df)
    correlations_df.to_csv(CORRELATIONS_CSV, index=False)

    plot_global_four_panel(summary_df)
    plot_corrected_log_by_voice(summary_df)

    print("\nProceso terminado.")
    print(f"Resumen: {SUMMARY_CSV.resolve()}")
    print(f"Correlaciones: {CORRELATIONS_CSV.resolve()}")
    print(f"Figuras: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
