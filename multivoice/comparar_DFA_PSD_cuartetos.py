# -*- coding: utf-8 -*-
"""
ETAPA 1: comparacion de exponentes DFA y espectro de potencias (PSD).

Se extraen las cuatro voces del mismo CSV y con el mismo formato que en
Cd_semilog_C1null_corpus.py. Para cada secuencia se estima:

    F(s) ~ s**alpha                [DFA estandar]
    S(f) ~ f**(-beta)              [periodograma, promediado logaritmico]
    delta = beta - (2*alpha - 1)

La relacion beta=2*alpha-1 se espera bajo hipotesis adicionales sobre el
proceso y el rango de escalas: NO es una identidad para toda melodia.

Se prueban dos ventanas de escalamiento:
    principal:   20 <= s <= floor(N/4)  (como la referencia DFA proporcionada)
    conservadora: 20 <= s <= floor(N/8)
Para la PSD se usan las frecuencias recíprocas: 1/s_max <= f <= 1/s_min.
Se guardan los DOS analisis, sin sustituir un resultado por otro.

No calcula el indice xi, MDFA ni surrogados IAAFT (no son esta etapa).
El DFA usa fathon si esta instalado, como el archivo de referencia, y si
no esta disponible utiliza un algoritmo NumPy equivalente (revSeg=False).

Requisitos: numpy, pandas, scipy, matplotlib. fathon opcional.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram, welch
from scipy.stats import linregress, pearsonr, spearmanr

try:
    import fathon
    from fathon import fathonUtils as fu
    FATHON_AVAILABLE = True
except ImportError:
    FATHON_AVAILABLE = False


# ============================================================================
# CONFIGURACION: personalizar aqui
# ============================================================================
CSV_PATH = Path("quartet_subcorpus_all_movements.csv")
OUTPUT_DIR = Path("resultados_DFA_vs_PSD")

ANALYSIS_MODE = "midi"  # "intervals" o "midi"
# False: intervalos CON signo, recomendado para examinar correlaciones.
# True: reproduce el np.abs(intervals) del codigo Cd proporcionado.
USE_ABSOLUTE_INTERVALS = False

DFA_ORDER = 2             # DFA2: detrending polinomial de orden 2
DFA_S_MIN = 20            # igual al archivo indice_no_linealidad_mdfa(2).py
DFA_S_MAX_CAP = 500
DFA_SCALE_POINTS = 25     # numero de tamanos log-espaciados (no consecutivos)
DFA_ENGINE = "auto"      # "auto", "fathon" o "numpy"

PSD_N_BINS = 12           # bins geometricos para no sobreponderar f grandes
PSD_METHOD = "welch"     # "welch" (menor varianza) o "periodogram"
PSD_WELCH_FRACTION = 0.50  # longitud de cada segmento de Welch relativa a N
PSD_WINDOW = "hann"
PSD_DETREND = "linear"

MIN_SCALES = 8            # minima cantidad de puntos para ajustar el DFA
MIN_PSD_BINS = 8         # minimo numero de bins ocupados para ajustar PSD
MIN_PSD_RAW = 16         # minimo numero de frecuencias sin agrupar
MIN_DECADES = 0.80       # criterio EXPLORATORIO de amplitud log10 en N/8
MAX_ALPHA_SENSITIVITY = 0.15  # |alpha_N/4 - alpha_N/8|
MAX_BETA_SENSITIVITY = 0.25   # |beta_N/4 - beta_N/8|

SAVE_MOVEMENT_FIGURES = True    # 1 PNG por movimiento, 4 filas x 2 columnas
SAVE_SUMMARY_FIGURES = True
PNG_DPI = 180

VOICE_ORDER = ["vln I", "vln II", "viola", "cello"]
VOICE_LABELS = {
    "vln I": "Violin I", "vln II": "Violin II",
    "viola": "Viola", "cello": "Violonchelo",
}
VOICE_COLORS = {
    "vln I": "#1f77b4", "vln II": "#ff7f0e",
    "viola": "#2ca02c", "cello": "#d62728",
}


# ============================================================================
# LECTURA DEL CORPUS: misma logica del script Cd proporcionado
# ============================================================================
def slugify(value) -> str:
    txt = re.sub(r"[^\w\s-]", "", str(value).strip().lower(), flags=re.UNICODE)
    return re.sub(r"[\s/]+", "_", txt).strip("_") or "sin_nombre"


def parse_literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def parse_voices(row: pd.Series) -> dict[str, np.ndarray]:
    voices = parse_literal(row["voices_pitches"])
    if not isinstance(voices, (list, tuple)):
        raise TypeError("voices_pitches no es una lista de voces")
    roles = [r.strip() for r in str(row.get("roles", "vln I|vln II|viola|cello")).split("|")]
    if len(voices) != len(roles):
        raise ValueError("Numero de roles distinto del numero de voces")
    voice_map = {}
    for role, pitches in zip(roles, voices):
        arr = np.asarray(pitches, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        voice_map[role] = arr
    missing = [v for v in VOICE_ORDER if v not in voice_map]
    if missing:
        raise ValueError(f"Faltan voces: {missing}")
    return {voice: voice_map[voice] for voice in VOICE_ORDER}


def prepare_sequence(pitches):
    midi = np.asarray(pitches, dtype=float).ravel()
    midi = midi[np.isfinite(midi)]
    if ANALYSIS_MODE == "midi":
        return midi
    if ANALYSIS_MODE == "intervals":
        intervals = np.diff(midi)
        return np.abs(intervals) if USE_ABSOLUTE_INTERVALS else intervals
    raise ValueError("ANALYSIS_MODE debe ser 'intervals' o 'midi'")


# ============================================================================
# DFA: integracion estandar y eliminacion de tendencias polinomiales
# ============================================================================
def dfa_numpy(x: np.ndarray, sizes: np.ndarray, degree: int):
    """DFA estandar, sin ventanas invertidas (revSeg=False)."""
    x = np.asarray(x, dtype=float)
    profile = np.cumsum(x - x.mean())
    n_all = len(profile)
    good_s, good_F = [], []
    for s in sizes:
        s = int(s)
        nseg = n_all // s
        if s <= degree + 2 or nseg < 2:
            continue
        segments = profile[:nseg * s].reshape(nseg, s)
        t = np.linspace(-1., 1., s)
        design = np.polynomial.polynomial.polyvander(t, degree)
        coeff, _, _, _ = np.linalg.lstsq(design, segments.T, rcond=None)
        residual = segments.T - design @ coeff
        fluc = float(np.sqrt(np.mean(residual ** 2)))
        if np.isfinite(fluc) and fluc > 0:
            good_s.append(float(s))
            good_F.append(fluc)
    return np.asarray(good_s), np.asarray(good_F)


def dfa_fathon(x: np.ndarray, sizes: np.ndarray, degree: int):
    """Mismo nucleo DFA que _dfa_fluctuation del archivo proporcionado."""
    if not FATHON_AVAILABLE:
        raise ImportError("DFA_ENGINE='fathon' requiere instalar fathon")
    profile = fu.toAggregated(np.asarray(x, dtype=float))
    dfa = fathon.DFA(profile)
    scales, fluc = dfa.computeFlucVec(
        sizes.astype(np.int64), revSeg=False, polOrd=degree
    )
    scales = np.asarray(scales, dtype=float)
    fluc = np.asarray(fluc, dtype=float)
    ok = np.isfinite(scales) & np.isfinite(fluc) & (fluc > 0)
    return scales[ok], fluc[ok]


def get_dfa_curve(x: np.ndarray, scale_divisor: int):
    n = len(x)
    smax = min(DFA_S_MAX_CAP, n // scale_divisor)
    if smax <= DFA_S_MIN:
        return np.empty(0), np.empty(0), smax
    sizes = np.unique(np.rint(np.geomspace(DFA_S_MIN, smax, DFA_SCALE_POINTS)).astype(int))
    sizes = sizes[sizes > DFA_ORDER + 2]
    if DFA_ENGINE not in {"auto", "fathon", "numpy"}:
        raise ValueError("DFA_ENGINE invalido")
    engine = ("fathon" if FATHON_AVAILABLE else "numpy") if DFA_ENGINE == "auto" else DFA_ENGINE
    if engine == "fathon":
        s, F = dfa_fathon(x, sizes, DFA_ORDER)
    else:
        s, F = dfa_numpy(x, sizes, DFA_ORDER)
    return s, F, smax


# ============================================================================
# PSD: periodograma y promediado por bins logaritmicos
# ============================================================================
def get_psd_curve(x: np.ndarray, smax: int):
    if smax <= DFA_S_MIN:
        return np.empty(0), np.empty(0), 0
    if PSD_METHOD == "welch":
        nseg = max(16, int(len(x) * PSD_WELCH_FRACTION))
        nseg = min(nseg, len(x))
        freq, power = welch(
            x, fs=1., window=PSD_WINDOW, detrend=PSD_DETREND,
            scaling="density", nperseg=nseg, noverlap=nseg // 2,
            nfft=len(x), return_onesided=True,
        )
    elif PSD_METHOD == "periodogram":
        freq, power = periodogram(
            x, fs=1., window=PSD_WINDOW, detrend=PSD_DETREND,
            scaling="density", return_onesided=True,
        )
    else:
        raise ValueError("PSD_METHOD debe ser 'welch' o 'periodogram'")
    f_low, f_high = 1. / smax, 1. / DFA_S_MIN
    ok = (np.isfinite(freq) & np.isfinite(power) & (power > 0)
          & (freq >= f_low) & (freq <= f_high))
    f, P = freq[ok], power[ok]
    n_raw = len(f)
    if n_raw == 0:
        return np.empty(0), np.empty(0), 0

    # Cada punto del ajuste representa un intervalo logaritmico de frecuencia;
    # se hace media ARITMETICA de la potencia dentro de cada bin.
    edges = np.geomspace(f_low, f_high, PSD_N_BINS + 1)
    bins = np.searchsorted(edges, f, side="right") - 1
    bins = np.clip(bins, 0, PSD_N_BINS - 1)  # incluye f_high
    f_binned, P_binned = [], []
    for b in range(PSD_N_BINS):
        m = bins == b
        if np.any(m):
            f_binned.append(float(np.exp(np.mean(np.log(f[m])))))
            P_binned.append(float(np.mean(P[m])))
    return np.asarray(f_binned), np.asarray(P_binned), n_raw


def linear_log_fit(x: np.ndarray, y: np.ndarray, min_points: int):
    """Estima y = 10**intercept * x**slope con regresion log10-log10."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    out = dict(slope=np.nan, intercept=np.nan, r2=np.nan, rmse_log10=np.nan,
               slope_se=np.nan, n_fit=int(len(x)), decades=np.nan)
    if len(x) < min_points or np.unique(x).size < 3:
        return out
    lx, ly = np.log10(x), np.log10(y)
    if np.ptp(ly) < 1e-12:
        # Pendiente nominal cero, pero R² no se define si Y es constante.
        out.update(slope=0., intercept=float(np.mean(ly)), rmse_log10=0.,
                   decades=float(np.ptp(lx)))
        return out
    reg = linregress(lx, ly)
    res = ly - (reg.intercept + reg.slope * lx)
    out.update(
        slope=float(reg.slope), intercept=float(reg.intercept),
        r2=float(reg.rvalue**2),
        rmse_log10=float(np.sqrt(np.mean(res ** 2))),
        slope_se=float(reg.stderr),  # OLS nominal: no es un IC formal para PSD
        decades=float(np.ptp(lx)),
    )
    return out


def analyze_one_range(x: np.ndarray, divisor: int):
    """Calcula ambos ajustes para un rango de escalas: N/4 o N/8."""
    s, F, smax = get_dfa_curve(x, divisor)
    df = linear_log_fit(s, F, MIN_SCALES)
    f, P, n_raw = get_psd_curve(x, smax)
    pf = linear_log_fit(f, P, MIN_PSD_BINS)
    alpha = df["slope"]
    beta = -pf["slope"] if np.isfinite(pf["slope"]) else np.nan
    delta = beta - (2. * alpha - 1.) if np.isfinite(alpha) and np.isfinite(beta) else np.nan
    return {
        "alpha": alpha, "beta": beta, "beta_pred": (2*alpha-1) if np.isfinite(alpha) else np.nan,
        "delta": delta, "abs_delta": abs(delta) if np.isfinite(delta) else np.nan,
        "dfa": df, "psd": pf, "s": s, "F": F, "f": f, "P": P,
        "psd_n_raw": n_raw, "smax": smax,
        "dfa_nseg_smax": len(x) // int(max(s)) if len(s) else 0,
        "enough_spectral_points": n_raw >= MIN_PSD_RAW,
    }


def result_to_flat(result, prefix):
    df, pf = result["dfa"], result["psd"]
    return {
        f"alpha_{prefix}": result["alpha"],
        f"beta_{prefix}": result["beta"],
        f"beta_pred_{prefix}": result["beta_pred"],
        f"delta_{prefix}": result["delta"],
        f"abs_delta_{prefix}": result["abs_delta"],
        f"dfa_r2_{prefix}": df["r2"],
        f"dfa_rmse_log10_{prefix}": df["rmse_log10"],
        f"dfa_alpha_se_ols_{prefix}": df["slope_se"],
        f"dfa_n_scales_{prefix}": df["n_fit"],
        f"dfa_decades_{prefix}": df["decades"],
        f"dfa_s_max_{prefix}": result["smax"],
        f"dfa_nseg_smax_{prefix}": result["dfa_nseg_smax"],
        f"psd_r2_{prefix}": pf["r2"],
        f"psd_rmse_log10_{prefix}": pf["rmse_log10"],
        f"psd_beta_se_ols_{prefix}": pf["slope_se"],
        f"psd_n_bins_{prefix}": pf["n_fit"],
        f"psd_n_frequencies_{prefix}": result["psd_n_raw"],
        f"psd_decades_{prefix}": pf["decades"],
    }


def is_estimable(result, require_range=False):
    dfa, psd = result["dfa"], result["psd"]
    ok = (np.isfinite(result["alpha"]) and np.isfinite(result["beta"])
          and dfa["n_fit"] >= MIN_SCALES and psd["n_fit"] >= MIN_PSD_BINS
          and result["psd_n_raw"] >= MIN_PSD_RAW)
    if require_range:
        ok = ok and np.isfinite(dfa["decades"]) and dfa["decades"] >= MIN_DECADES
        ok = ok and np.isfinite(psd["decades"]) and psd["decades"] >= MIN_DECADES * .80
    return bool(ok)


# ============================================================================
# FIGURAS
# ============================================================================
def draw_fit(ax, xx, yy, fit, title, exponent_name, exponent):
    if len(xx):
        ax.loglog(xx, yy, "o", ms=4, alpha=.8, label="Datos")
    if np.isfinite(fit["slope"]):
        xline = np.geomspace(np.min(xx), np.max(xx), 100)
        yline = 10 ** (fit["intercept"] + fit["slope"] * np.log10(xline))
        ax.loglog(xline, yline, "-", lw=1.6, label="Ajuste global")
    if np.isfinite(exponent):
        ax.set_title(f"{title} | {exponent_name}={exponent:.3f}, R²={fit['r2']:.3f}", fontsize=10)
    else:
        ax.set_title(f"{title} | sin ajuste suficiente", fontsize=10)
    ax.grid(True, which="both", alpha=.2)
    ax.legend(loc="best", fontsize=8) if len(xx) else None


def save_movement_plot(composer, work, movement, row_index, results):
    fig, axs = plt.subplots(4, 2, figsize=(13.2, 13.8))
    for i, voice in enumerate(VOICE_ORDER):
        left, right = axs[i, 0], axs[i, 1]
        if voice not in results:
            left.text(.5, .5, "Voz no disponible", ha="center", va="center", transform=left.transAxes)
            right.axis("off")
            continue
        r = results[voice]["principal"]
        draw_fit(left, r["s"], r["F"], r["dfa"], VOICE_LABELS[voice], "alpha", r["alpha"])
        draw_fit(right, r["f"], r["P"], r["psd"], VOICE_LABELS[voice], "beta", r["beta"])
        left.set_xlabel("Escala s (notas/intervalos)")
        left.set_ylabel("Fluctuacion F(s)")
        right.set_xlabel("Frecuencia f (ciclos por nota/intervalo)")
        right.set_ylabel("Densidad espectral")
        if np.isfinite(r["delta"]):
            right.text(.03, .04, f"beta-(2alpha-1)={r['delta']:+.3f}",
                       transform=right.transAxes, fontsize=9,
                       bbox={"facecolor": "white", "alpha": .85, "edgecolor": ".7"})
    fig.suptitle(f"DFA frente a PSD | {composer} | {work} | Movimiento {movement}\n"
                 f"{ANALYSIS_MODE}, abs(intervalos)={USE_ABSOLUTE_INTERVALS}; rango principal N/4",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    folder = OUTPUT_DIR / "por_movimiento" / slugify(composer)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{row_index:03d}_{slugify(work)[:70]}_mov_{slugify(movement)}"
    fig.savefig(folder / f"{stem}.png", dpi=PNG_DPI)
    plt.close(fig)


def save_summary_plots(df):
    valid = df[np.isfinite(df["alpha_principal"]) & np.isfinite(df["beta_principal"])].copy()
    qualified = valid[valid["criterio_conservador"]].copy()
    # beta observado vs beta predicho: la referencia es identidad, no regresion
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    for voice in VOICE_ORDER:
        sub = valid[valid["voice"] == voice]
        if len(sub):
            ax.scatter(sub["beta_pred_principal"], sub["beta_principal"], s=21,
                       color=VOICE_COLORS[voice], alpha=.25, label=VOICE_LABELS[voice])
        sub_good = qualified[qualified["voice"] == voice]
        if len(sub_good):
            ax.scatter(sub_good["beta_pred_principal"], sub_good["beta_principal"],
                       s=32, facecolors=VOICE_COLORS[voice], edgecolors="black",
                       linewidths=.45, alpha=.88)
    if len(valid):
        finite = np.r_[valid["beta_principal"].to_numpy(), valid["beta_pred_principal"].to_numpy()]
        lim = [np.min(finite) - .2, np.max(finite) + .2]
        ax.plot(lim, lim, "k--", lw=1.5, label="beta = 2alpha - 1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel("2 alpha_DFA - 1 (rango N/4)")
    ax.set_ylabel("beta_PSD (rango N/4)")
    ax.set_title(f"Consistencia DFA/PSD | puntos: {len(valid)}; perfil conservador: {len(qualified)}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_beta_PSD_vs_2alpha_menos_1.png", dpi=PNG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.7))
    for voice in VOICE_ORDER:
        sub = valid[valid["voice"] == voice]
        ax.scatter(sub["n_series"], sub["delta_principal"], s=20,
                   color=VOICE_COLORS[voice], alpha=.45, label=VOICE_LABELS[voice])
    ax.axhline(0., ls="--", lw=1.3, color="black")
    ax.set_xlabel("Longitud N de la serie")
    ax.set_ylabel("delta = beta - (2alpha - 1)")
    ax.set_title("Discrepancia entre exponentes frente a longitud")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_delta_vs_longitud.png", dpi=PNG_DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    for voice in VOICE_ORDER:
        sub = df[df["voice"] == voice]
        axes[0].scatter(sub["n_series"], sub["dfa_decades_conservador"],
                        color=VOICE_COLORS[voice], alpha=.4, s=16, label=VOICE_LABELS[voice])
    axes[0].axhline(MIN_DECADES, ls="--", color="black", lw=1.1)
    axes[0].set_xlabel("Longitud N")
    axes[0].set_ylabel("Decadas DFA (N/8)")
    axes[0].set_title("Rango de escalas disponible")
    axes[0].legend(fontsize=9)
    counts = df["estado"].value_counts().sort_index()
    axes[1].bar(range(len(counts)), counts.values, color="0.55")
    axes[1].set_xticks(range(len(counts)), counts.index, rotation=15, ha="right")
    axes[1].set_ylabel("Cantidad de voces")
    axes[1].set_title("Clasificacion diagnostica")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_longitud_y_diagnosticos.png", dpi=PNG_DPI)
    plt.close(fig)


def diagnostics_report(df):
    total = len(df)
    valid = df[df["estimable_principal"]]
    good = df[df["criterio_conservador"]]
    lines = [
        "ETAPA 1 - COMPARACION DFA / PSD", "=================================", "",
        f"Numero total de voces: {total}",
        f"Ajustes definidos y con suficientes puntos (N/4): {len(valid)}",
        f"Criterio conservador de rango y estabilidad (N/8): {len(good)}", "",
        "Interpretacion de las etiquetas:",
        "  CONSERVADOR: rango log suficiente en N/8, puntos minimos y",
        "               alpha/beta estables al variar la escala maxima.",
        "  EXPLORATORIO: ajuste N/4 disponible pero no supera los criterios.",
        "  INSUFICIENTE: faltan puntos para ajustar DFA y/o PSD.",
        "Esto NO certifica una ley de potencia ni estacionariedad.", "",
        f"Criterios: min decadas N/8={MIN_DECADES}; min escalas DFA={MIN_SCALES};",
        f"min bins PSD={MIN_PSD_BINS}; min frecuencias PSD={MIN_PSD_RAW};",
        f"|Delta alpha| <= {MAX_ALPHA_SENSITIVITY};",
        f"|Delta beta| <= {MAX_BETA_SENSITIVITY}", "",
        "El R² y el error son de los ajustes en escala log10-log10.",
        "El error estandar OLS es solo orientativo: las estimaciones",
        "a distintas escalas y bins espectrales no son independientes.", "",
    ]
    for name, subset in [("Estimables N/4", valid), ("Criterio conservador", good)]:
        lines.append(f"{name} (n={len(subset)}):")
        if len(subset):
            delta = subset["delta_principal"].to_numpy(dtype=float)
            lines.append(f"  mediana(delta) = {np.nanmedian(delta):+.4f}")
            lines.append(f"  mediana(|delta|) = {np.nanmedian(np.abs(delta)):.4f}")
            if len(subset) >= 3:
                a = subset["beta_pred_principal"].to_numpy(dtype=float)
                b = subset["beta_principal"].to_numpy(dtype=float)
                if np.std(a) > 0 and np.std(b) > 0:
                    lines.append(f"  Pearson(beta_pred,beta) = {pearsonr(a,b)[0]:.4f}")
                    lines.append(f"  Spearman(beta_pred,beta) = {spearmanr(a,b)[0]:.4f}")
        lines.append("")
    lines.extend([
        "Interpretacion estadistica:",
        f"PSD: metodo={PSD_METHOD}; ventana={PSD_WINDOW}; detrend={PSD_DETREND}.",
        "La coincidencia beta = 2alpha-1 es una hipotesis teorica CON SUPUESTOS,",
        "no una igualdad obligatoria para series cortas ni no estacionarias.",
        "R² alto NO demuestra leyes de escala ni memoria de largo alcance.",
        "Se trabaja en indice de evento (nota), no en tiempo fisico.",
        "La bandera 'conservador' es un filtro exploratorio, NO un test formal.",
    ])
    (OUTPUT_DIR / "README_resultados.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7]))


# ============================================================================
# MAIN
# ============================================================================
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontro el corpus: {CSV_PATH.resolve()}")
    if DFA_ENGINE == "fathon" and not FATHON_AVAILABLE:
        raise ImportError("Instala fathon o usa DFA_ENGINE='auto'/'numpy'")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(CSV_PATH)
    required = {"composer", "work_title", "movement_no", "roles", "voices_pitches"}
    missing = required.difference(corpus.columns)
    if missing:
        raise KeyError(f"Columnas ausentes: {sorted(missing)}")
    summary_rows = []
    for idx, row in corpus.iterrows():
        composer = (str(row["composer_canonical"])
                    if "composer_canonical" in corpus.columns and pd.notna(row["composer_canonical"])
                    else str(row["composer"]))
        work = str(row["work_title"])
        movement = row["movement_no"]
        print(f"[{idx+1}/{len(corpus)}] {composer} | {work} | mov. {movement}")
        try:
            voices = parse_voices(row)
        except Exception as exc:
            warnings.warn(f"Movimiento {idx}: {exc}")
            continue
        movement_results = {}
        for voice in VOICE_ORDER:
            sequence = prepare_sequence(voices[voice])
            n = len(sequence)
            rec = {
                "corpus_row": idx, "composer": composer,
                "composer_id": row.get("composer", np.nan),
                "work_id": row.get("work_id", np.nan),
                "work_title": work, "movement_no": movement,
                "period": row.get("period", np.nan),
                "filename": row.get("filename", np.nan),
                "voice": voice, "analysis_mode": ANALYSIS_MODE,
                "abs_intervals": USE_ABSOLUTE_INTERVALS,
                "psd_method": PSD_METHOD,
                "n_notes": len(voices[voice]), "n_series": n,
                "n_unique": len(np.unique(sequence)),
                "std_series": float(np.std(sequence)) if n else np.nan,
                "dfa_order": DFA_ORDER, "dfa_s_min": DFA_S_MIN,
            }
            if n < 10 or not np.all(np.isfinite(sequence)) or np.std(sequence) < 1e-12:
                rec.update(estado="insuficiente", motivo="longitud, finitud o varianza")
                summary_rows.append(rec)
                continue
            try:
                r4 = analyze_one_range(sequence, divisor=4)
                r8 = analyze_one_range(sequence, divisor=8)
                rec.update(result_to_flat(r4, "principal"))
                rec.update(result_to_flat(r8, "conservador"))
                rec["estimable_principal"] = is_estimable(r4)
                rec["estimable_conservador"] = is_estimable(r8, require_range=True)
                da = abs(r4["alpha"] - r8["alpha"]) if np.isfinite(r4["alpha"]) and np.isfinite(r8["alpha"]) else np.nan
                db = abs(r4["beta"] - r8["beta"]) if np.isfinite(r4["beta"]) and np.isfinite(r8["beta"]) else np.nan
                rec["sensitivity_alpha"] = da
                rec["sensitivity_beta"] = db
                rec["criterio_conservador"] = bool(
                    rec["estimable_principal"] and rec["estimable_conservador"]
                    and np.isfinite(da) and da <= MAX_ALPHA_SENSITIVITY
                    and np.isfinite(db) and db <= MAX_BETA_SENSITIVITY
                )
                rec["estado"] = ("conservador" if rec["criterio_conservador"] else
                                 "exploratorio" if rec["estimable_principal"] else "insuficiente")
                rec["motivo"] = ""
                movement_results[voice] = {"principal": r4, "conservador": r8}
            except Exception as exc:
                rec.update(estado="error", motivo=str(exc),
                           estimable_principal=False, estimable_conservador=False,
                           criterio_conservador=False)
                print(f"  ERROR voz {voice}: {exc}")
            summary_rows.append(rec)
        if SAVE_MOVEMENT_FIGURES and movement_results:
            save_movement_plot(composer, work, movement, idx, movement_results)

    df = pd.DataFrame(summary_rows)
    if df.empty:
        raise RuntimeError("No se pudieron extraer voces del corpus")
    for col in ["estimable_principal", "estimable_conservador", "criterio_conservador"]:
        if col not in df:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    df.to_csv(OUTPUT_DIR / "resumen_DFA_PSD_por_voz.csv", index=False, encoding="utf-8-sig")
    if SAVE_SUMMARY_FIGURES:
        save_summary_plots(df)
    diagnostics_report(df)
    print(f"\nCSV y figuras guardados en {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
