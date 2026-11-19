# -*- coding: utf-8 -*-
"""
FIGURA COMBINADA: DFA/mDFA + IAAFT, C_d + shuffle y q_d + shuffle
================================================================
Entrada: un archivo .npy que contenga una serie finita unidimensional.

Por defecto:
  * DFA/mDFA: sobre el .npy ORIGINAL, usando fathon y los IAAFT de ml.iaaft.
  * C_d y q_d: sobre intervalos CON SIGNO np.diff(x), asumiendo x=MIDI.
  * Nulo de DFA/mDFA: IAAFT sobre la serie ORIGINAL.
  * Nulo de C_d y q_d: shuffle de la serie usada para C_d.

Definiciones metodológicas (conforme a los scripts de referencia):
  * mDFA: m_t=|x[t+1]-x[t]|, restar media, integrar m_t, aplicar el
    perfil estándar DFA una segunda vez y corregir F_mDFA(s)=F(s)/s.
  * alfa efectivo: ajuste LINEAL global de log10(F) frente a log10(s).
  * xi_NL: promedio de las diferencias absolutas estandarizadas de las
    PENDIENTES de ajustes POLINOMIALES globales a real y surrogados.
  * C_d: ml.gamma_index_rank_ties(x, max_gamma=7, K=K).
  * lambda: ajuste log(C_d)=log(A)-lambda*d, d=1..8.
  * lambda_null: media de lambdas ajustadas por separado en cada shuffle.
  * kappa=lambda_obs/lambda_null.
  * q_d=C_(d+1)/C_d, d=1..7 (SIN ajuste adicional).

REQUISITOS: numpy, matplotlib, fathon y tu modulo mi_libreria.

EJEMPLOS:
    python graficar_DFA_mDFA_Cd_desde_npy.py melodia.npy
    python graficar_DFA_mDFA_Cd_desde_npy.py melodia.npy --method dfa
    python graficar_DFA_mDFA_Cd_desde_npy.py intervalos.npy --cd-mode direct
    python graficar_DFA_mDFA_Cd_desde_npy.py melodia.npy --k-mode target

También puedes cambiar los valores en CONFIGURACION y ejecutar el script.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import fathon
from fathon import fathonUtils as fu

import mi_libreria as ml


# =====================================================================
# CONFIGURACION
# =====================================================================
NPY_PATH = Path('mi_serie.npy')
OUTPUT_DIR = Path('resultados_serie_individual')

DFA_METHOD = 'mdfa'        # 'dfa' o 'mdfa'
CD_MODE = 'signed_intervals'  # 'signed_intervals' o 'direct'

# Parámetros del DFA/mDFA siguiendo indice_no_linealidad_mdfa(2).py.
DFA_ORDER = 2
S_MIN = 20
S_MAX = None               # None -> min(500, N_efectivo//4)
POLY_DEGREE = 4
N_IAAFT = 20
IAAFT_TOL_PC = 0.1
IAAFT_MAXITER = 1_000_000
IAAFT_SORTTYPE = 'quicksort'

# C_d sobre rangos con empates usando mi_libreria.
MAX_GAMMA = 7             # ml devuelve C_0,...,C_8
D_MIN, D_MAX = 1, 8
K_MODE = 'fixed'          # 'fixed' o 'target'
K_FIXED = 2
C1_NULL_TARGET = 0.50026  # usado solo si K_MODE='target'
N_SHUFFLES = 200

# Bandas: min-max como en los códigos de referencia (percentiles 0 y 100).
LOW_PERCENTILE = 0.0
HIGH_PERCENTILE = 100.0
SEED = 12345

SAVE_PNG = True
SAVE_PDF = False
SAVE_NPZ = True
SAVE_JSON = True
SHOW_PLOT = True
DPI = 200


# =====================================================================
# LECTURA / VALIDACIÓN
# =====================================================================
def load_1d_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'No existe el archivo: {path.resolve()}')
    x = np.load(path, allow_pickle=False)
    if x.ndim != 1:
        raise ValueError(f'Se requiere una serie 1D; shape recibido: {x.shape}.')
    x = np.asarray(x, dtype=np.float64)
    if x.size < 20:
        raise ValueError('La serie es demasiado corta (menos de 20 muestras).')
    if not np.all(np.isfinite(x)):
        raise ValueError('La serie contiene NaN/inf. No se eliminan puntos para evitar crear intervalos falsos.')
    if np.std(x) <= 1e-12:
        raise ValueError('La serie es constante; los exponentes no están definidos.')
    return x


def cd_input_series(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'signed_intervals':
        return np.diff(x)       # intervalos CON SIGNO, sin valor absoluto
    if mode == 'direct':
        return x.copy()         # si el .npy ya contiene los intervalos, p.ej.
    raise ValueError("CD_MODE debe ser 'signed_intervals' o 'direct'.")


# =====================================================================
# DFA/mDFA: se mantienen las operaciones de indice_no_linealidad_mdfa(2).py
# =====================================================================
def dfa_fluctuation(series: np.ndarray, win_sizes: np.ndarray, order: int):
    """DFA con fathon: perfil integrado, revSeg=False, polOrd=order."""
    profile = fu.toAggregated(np.asarray(series, dtype=float))
    solver = fathon.DFA(profile)
    s, F = solver.computeFlucVec(win_sizes, revSeg=False, polOrd=order)
    s, F = np.asarray(s, float), np.asarray(F, float)
    good = np.isfinite(s) & np.isfinite(F) & (F > 0)
    return s[good], F[good]


def mdfa_fluctuation(x: np.ndarray, win_sizes: np.ndarray, order: int):
    """magnitud -> centrado -> integración adicional -> DFA -> F(s)/s."""
    magnitudes = np.abs(np.diff(x))
    magnitudes -= np.mean(magnitudes)
    magnitudes_integradas = np.cumsum(magnitudes)
    s, F = dfa_fluctuation(magnitudes_integradas, win_sizes, order)
    F_corr = F / s
    good = np.isfinite(F_corr) & (F_corr > 0)
    return s[good], F_corr[good]


def global_polynomial_slopes(log_s: np.ndarray, log_F: np.ndarray, degree: int):
    """Un polinomio global de grado 'degree'; se derivan sus valores."""
    if log_s.size <= degree:
        raise ValueError(f'No hay suficientes escalas para polinomio de grado {degree}.')
    coeff = np.polyfit(log_s, log_F, deg=degree)
    polynomial = np.poly1d(coeff)
    return polynomial(log_s), np.polyder(polynomial)(log_s)


def generate_iaaft_from_ml(x: np.ndarray, ns: int, seed: int) -> np.ndarray:
    """IMPORTANTE: NO reimplementa IAAFT; llama exclusivamente a ml.iaaft.

    La semilla se aplica a np.random global solo durante esta llamada. Esto
    ofrece reproducibilidad si la función ml.iaaft utiliza el RNG global de
    NumPy; si crea su propio generador, la semilla podría no controlarla.
    """
    previous_state = np.random.get_state()
    try:
        np.random.seed(seed)
        surrogates = ml.iaaft(
            x,
            ns=ns,
            tol_pc=IAAFT_TOL_PC,
            verbose=False,
            maxiter=IAAFT_MAXITER,
            sorttype=IAAFT_SORTTYPE,
        )
    finally:
        np.random.set_state(previous_state)

    surrogates = np.asarray(surrogates, dtype=float)
    if surrogates.shape != (ns, len(x)):
        raise ValueError(
            f'ml.iaaft devolvió shape={surrogates.shape}; se esperaba {(ns, len(x))}.'
        )
    if not np.all(np.isfinite(surrogates)):
        raise ValueError('ml.iaaft devolvió al menos un dato NaN/inf.')
    return surrogates


def compute_dfa_with_iaaft(x: np.ndarray, method: str):
    method = method.lower()
    if method not in {'dfa', 'mdfa'}:
        raise ValueError("El método debe ser 'dfa' o 'mdfa'.")
    if N_IAAFT < 2:
        raise ValueError('N_IAAFT debe ser al menos 2 para calcular xi.')
    if POLY_DEGREE < 1:
        raise ValueError('POLY_DEGREE debe ser al menos 1.')

    effective_n = len(x) if method == 'dfa' else len(x) - 1
    s_max = int(S_MAX) if S_MAX is not None else min(500, effective_n // 4)
    if s_max <= S_MIN or S_MIN <= DFA_ORDER + 1:
        raise ValueError(f'Escalas inválidas: s_min={S_MIN}, s_max={s_max}.')
    scales = fu.linRangeByStep(int(S_MIN), s_max)
    curve_fn = dfa_fluctuation if method == 'dfa' else mdfa_fluctuation

    s, F = curve_fn(x, scales, DFA_ORDER)
    if s.size <= POLY_DEGREE + 1:
        raise ValueError('Insuficientes escalas válidas para calcular xi.')
    log_s = np.log10(s)
    log_F = np.log10(F)

    polyfit_real, slopes_real = global_polynomial_slopes(log_s, log_F, POLY_DEGREE)
    slope, intercept = np.polyfit(log_s, log_F, deg=1)
    fitted_line = intercept + slope * log_s
    ss_res = np.sum((log_F - fitted_line)**2)
    ss_tot = np.sum((log_F - log_F.mean())**2)
    r2_alpha = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    print(f'Calculando {N_IAAFT} surrogados IAAFT con ml.iaaft...', flush=True)
    surrogates = generate_iaaft_from_ml(x, N_IAAFT, SEED)
    surrogate_logF = np.empty((N_IAAFT, len(s)), dtype=float)
    surrogate_slopes = np.empty_like(surrogate_logF)

    for j, ys in enumerate(surrogates):
        sj, Fj = curve_fn(ys, scales, DFA_ORDER)
        if sj.shape != s.shape or not np.allclose(sj, s):
            raise RuntimeError(
                f'El surrogate IAAFT {j} tiene escalas distintas de la original. '
                'No se puede calcular xi sin escalas alineadas.'
            )
        logFj = np.log10(Fj)
        surrogate_logF[j, :] = logFj
        _, surrogate_slopes[j, :] = global_polynomial_slopes(log_s, logFj, POLY_DEGREE)

    mean_slope = np.mean(surrogate_slopes, axis=0)
    std_slope = np.std(surrogate_slopes, axis=0, ddof=0)  # como código de referencia
    valid = (np.isfinite(slopes_real) & np.isfinite(mean_slope) &
             np.isfinite(std_slope) & (std_slope > np.finfo(float).eps))
    if not np.any(valid):
        raise RuntimeError('No hay escalas con dispersión válida para xi.')
    if not np.all(valid):
        warnings.warn(f'Se excluyeron {np.sum(~valid)} escalas al promediar xi.', RuntimeWarning)

    xi_nl = float(np.mean(np.abs(slopes_real[valid] - mean_slope[valid]) / std_slope[valid]))
    return {
        'method': method,
        's': s,
        'F': F,
        'log_s': log_s,
        'log_F': log_F,
        'polyfit_real': polyfit_real,
        'surrogate_logF': surrogate_logF,
        'alpha': float(slope),
        'R2_alpha': r2_alpha,
        'xi_NL': xi_nl,
    }


# =====================================================================
# C_d, ajuste exponencial, q_d y shuffle
# =====================================================================
def c1_null_exact(counts: np.ndarray, K: int) -> float:
    """Probabilidad teórica bajo shuffle para cercanía de rangos <=K."""
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n < 2:
        return np.nan
    K = min(int(K), len(counts) - 1)
    close = int(np.sum(counts * (counts - 1) // 2))
    for step in range(1, K + 1):
        close += int(np.sum(counts[:-step] * counts[step:]))
    return float(close / (n * (n - 1) / 2.0))


def select_K(x_cd: np.ndarray, mode: str, fixed: int, target: float):
    _, counts = np.unique(x_cd, return_counts=True)
    n_unique = int(len(counts))
    if n_unique < 2:
        raise ValueError('Solo hay un valor distinto para C_d; no hay decaimiento útil.')
    if mode == 'fixed':
        K = int(fixed)
        if K < 0 or K >= n_unique - 1:
            raise ValueError(
                f'K fijo={K} no es válido para n_unique={n_unique}: '
                f'usa K entre 0 y {n_unique-2} para evitar C1=1.'
            )
    elif mode == 'target':
        if not 0 <= target <= 1:
            raise ValueError('C1_NULL_TARGET debe estar entre 0 y 1.')
        # Exactamente el criterio de los scripts de referencia: el mayor K
        # cuyo C1_null(K) no supere el objetivo (o 0 si ninguno cumple).
        candidates = [k for k in range(n_unique)
                      if c1_null_exact(counts, k) <= target]
        K = candidates[-1] if candidates else 0
        if K >= n_unique - 1:
            raise ValueError('El K automático dio C1_null=1; reduce el target.')
    else:
        raise ValueError("K_MODE debe ser 'fixed' o 'target'.")
    return int(K), n_unique, c1_null_exact(counts, K)


def compute_C(x_cd: np.ndarray, K: int) -> np.ndarray:
    C, _ = ml.gamma_index_rank_ties(
        np.asarray(x_cd, dtype=float),
        max_gamma=MAX_GAMMA,
        K=int(K),
    )
    C = np.asarray(C, dtype=float).ravel()
    if len(C) < D_MAX + 1:
        raise ValueError(f'ml.gamma_index_rank_ties devolvió {len(C)} elementos; se requieren {D_MAX+1}.')
    return C[:D_MAX+1]  # C_0,...,C_8


def fit_exponential(C: np.ndarray):
    d = np.arange(len(C), dtype=float)
    mask = (d >= D_MIN) & (d <= D_MAX) & np.isfinite(C) & (C > 0)
    n_fit = int(mask.sum())
    out = {'lambda': np.nan, 'A': np.nan, 'R2': np.nan,
           'RMSE_log': np.nan, 'n_fit': n_fit,
           'd_fit': np.arange(D_MIN, D_MAX+1, dtype=float),
           'C_fit': np.full(D_MAX-D_MIN+1, np.nan, dtype=float)}
    if n_fit < 2:
        return out
    xv, yv = d[mask], np.log(C[mask])
    slope, intercept = np.polyfit(xv, yv, 1)
    fitted = intercept + slope * xv
    ss_res = float(np.sum((yv - fitted)**2))
    ss_tot = float(np.sum((yv - yv.mean())**2))
    lam = -float(slope)
    A = float(np.exp(intercept))
    out.update({'lambda': lam, 'A': A,
                'R2': 1-ss_res/ss_tot if ss_tot > 0 else np.nan,
                'RMSE_log': float(np.sqrt(np.mean((yv-fitted)**2))),
                'C_fit': A * np.exp(-lam * out['d_fit'])})
    return out


def compute_q(C: np.ndarray) -> np.ndarray:
    d = np.arange(D_MIN, D_MAX, dtype=int)  # 1,...,7
    denom = C[d]
    numer = C[d+1]
    q = np.full(len(d), np.nan, dtype=float)
    good = np.isfinite(denom) & np.isfinite(numer) & (denom > 0)
    q[good] = numer[good] / denom[good]
    return q


def geometric_mean_positive(array: np.ndarray, axis=0) -> np.ndarray:
    arr = np.asarray(array, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        center = np.exp(np.nanmean(logs, axis=axis))
    return center


def nanpercentile_safe(a: np.ndarray, percentile: float, axis=0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanpercentile(a, percentile, axis=axis)


def compute_cd_with_shuffle(x_cd: np.ndarray, K: int):
    C_obs = compute_C(x_cd, K)
    fit_obs = fit_exponential(C_obs)
    q_obs = compute_q(C_obs)

    rng = np.random.default_rng(SEED + 100)
    C_null = np.full((N_SHUFFLES, D_MAX+1), np.nan)
    q_null = np.full((N_SHUFFLES, D_MAX-D_MIN), np.nan)
    lambdas_null = np.full(N_SHUFFLES, np.nan)

    print(f'Calculando {N_SHUFFLES} shuffles para C_d y q_d...', flush=True)
    for j in range(N_SHUFFLES):
        Cj = compute_C(rng.permutation(x_cd), K)
        C_null[j] = Cj
        q_null[j] = compute_q(Cj)
        lambdas_null[j] = fit_exponential(Cj)['lambda']

    valid_lam = lambdas_null[np.isfinite(lambdas_null)]
    mean_lam = float(valid_lam.mean()) if valid_lam.size else np.nan
    std_lam = float(valid_lam.std(ddof=1)) if valid_lam.size >= 2 else np.nan
    lam_obs = fit_obs['lambda']
    kappa = (float(lam_obs / mean_lam)
             if np.isfinite(lam_obs) and np.isfinite(mean_lam) and
             lam_obs > 0 and mean_lam > 0 else np.nan)

    C_center = geometric_mean_positive(C_null)
    q_center = geometric_mean_positive(q_null)
    return {
        'C_observed': C_obs,
        'q_observed': q_obs,
        'fit_observed': fit_obs,
        'C_null_all': C_null,
        'q_null_all': q_null,
        'C_null_center': C_center,
        'C_null_low': nanpercentile_safe(C_null, LOW_PERCENTILE),
        'C_null_high': nanpercentile_safe(C_null, HIGH_PERCENTILE),
        'q_null_center': q_center,
        'q_null_low': nanpercentile_safe(q_null, LOW_PERCENTILE),
        'q_null_high': nanpercentile_safe(q_null, HIGH_PERCENTILE),
        'fit_null_center': fit_exponential(C_center),
        'lambda_null_values': lambdas_null,
        'lambda_null_mean': mean_lam,
        'lambda_null_std': std_lam,
        'kappa': kappa,
    }


# =====================================================================
# VISUALIZACION (3 paneles)
# =====================================================================
def make_figure(path: Path, x: np.ndarray, dfa: dict, cd: dict,
                x_cd: np.ndarray, K: int, n_unique: int, C1_null: float,
                cd_mode: str) -> plt.Figure:
    fig = plt.figure(figsize=(13.6, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    ax_dfa = fig.add_subplot(gs[0, :])
    ax_cd = fig.add_subplot(gs[1, 0])
    ax_q = fig.add_subplot(gs[1, 1])
    NULL_COLOR = '#62666c'
    OBS_COLOR = '#2864ad'

    # Panel A: se representan los mismos logaritmos que en el script de xi.
    log_s = dfa['log_s']
    sur_logF = dfa['surrogate_logF']
    sur_low = np.min(sur_logF, axis=0)
    sur_high = np.max(sur_logF, axis=0)
    sur_mean = np.mean(sur_logF, axis=0)
    ax_dfa.fill_between(log_s, sur_low, sur_high, color=NULL_COLOR, alpha=.2,
                        label=f'IAAFT: mínimo–máximo ({N_IAAFT} surrogados)')
    ax_dfa.plot(log_s, sur_mean, '--', lw=1.7, color=NULL_COLOR,
                label='IAAFT: media')
    ax_dfa.plot(log_s, dfa['log_F'], '.', ms=4.5, color=OBS_COLOR,
                label=f'{dfa["method"].upper()}: observado')
    ax_dfa.plot(log_s, dfa['polyfit_real'], '-', lw=1.8, color='#c65d20',
                label=f'Ajuste polinomial global (grado {POLY_DEGREE})')
    ax_dfa.set_xlabel(r'$\log_{10}(s)$')
    ax_dfa.set_ylabel(r'$\log_{10}[F(s)]$' if dfa['method']=='dfa'
                      else r'$\log_{10}[F(s)/s]$')
    ax_dfa.set_title(f'A  ·  {dfa["method"].upper()} y modelo nulo IAAFT')
    ax_dfa.text(.02, .97,
                rf'$\alpha_{{\mathrm{{eff}}}}={dfa["alpha"]:.4f}$'
                + '\n' + rf'$\xi_{{\mathrm{{NL}}}}={dfa["xi_NL"]:.4f}$'
                + '\n' + rf'$R^2_{{\alpha}}={dfa["R2_alpha"]:.4f}$',
                transform=ax_dfa.transAxes, ha='left', va='top', fontsize=11,
                bbox=dict(facecolor='white', edgecolor='.8', alpha=.92, boxstyle='round,pad=.35'))
    ax_dfa.legend(loc='lower right', fontsize=9)
    ax_dfa.grid(alpha=.22)

    # Panel B: C_d semilog con ajuste exponencial y shuffles.
    ds = np.arange(D_MIN, D_MAX+1)
    C_obs = cd['C_observed'][ds]
    c_center = cd['C_null_center'][ds]
    c_low = cd['C_null_low'][ds]
    c_high = cd['C_null_high'][ds]
    band_valid = np.isfinite(c_low) & np.isfinite(c_high) & (c_low>0) & (c_high>0)
    ax_cd.fill_between(ds[band_valid], c_low[band_valid], c_high[band_valid],
                       color=NULL_COLOR, alpha=.18,
                       label=f'Shuffle P{LOW_PERCENTILE:g}–P{HIGH_PERCENTILE:g}')
    obs_good = np.isfinite(C_obs) & (C_obs>0)
    null_good = np.isfinite(c_center) & (c_center>0)
    ax_cd.semilogy(ds[obs_good], C_obs[obs_good], 'o', ms=6.5, color=OBS_COLOR,
                   label=r'$C_d$ observado')
    ax_cd.semilogy(ds[null_good], c_center[null_good], 's', ms=5.0,
                   markerfacecolor='white', markeredgecolor=NULL_COLOR,
                   label='Shuffle: media geométrica')
    fit_obs = cd['fit_observed']
    fit_null = cd['fit_null_center']
    if np.isfinite(fit_obs['lambda']):
        ax_cd.semilogy(fit_obs['d_fit'], fit_obs['C_fit'], '-', color=OBS_COLOR,
                       lw=1.9, label='Ajuste exponencial observado')
    if np.isfinite(fit_null['lambda']):
        ax_cd.semilogy(fit_null['d_fit'], fit_null['C_fit'], '--', color=NULL_COLOR,
                       lw=1.7, label='Ajuste del shuffle central')
    ax_cd.set(xlabel=r'Dimensión $d$', ylabel=r'$C_d$', title='B  ·  Decaimiento de $C_d$')
    ax_cd.set_xticks(ds)
    ax_cd.set_xlim(D_MIN-.3, D_MAX+.3)
    ax_cd.grid(alpha=.22, which='both')
    ax_cd.text(.025, .035,
               rf'$\lambda_{{\mathrm{{obs}}}}={fit_obs["lambda"]:.4f}$'
               + '\n' + rf'$\langle\lambda_{{\mathrm{{null}}}}\rangle={cd["lambda_null_mean"]:.4f}$'
               + '\n' + rf'$\kappa={cd["kappa"]:.4f}$'
               + '\n' + rf'$R^2_{{C_d}}={fit_obs["R2"]:.4f}$'
               + '\n' + rf'$K={K},\;n_{{\rm unique}}={n_unique}$',
               transform=ax_cd.transAxes, ha='left', va='bottom', fontsize=9.3,
               bbox=dict(facecolor='white', edgecolor='.8', alpha=.93, boxstyle='round,pad=.3'))
    ax_cd.legend(fontsize=8, loc='upper right')

    # Panel C: los cocientes q_d son puntos, NO se ajusta una curva.
    dq = np.arange(D_MIN, D_MAX)
    qobs = cd['q_observed']
    qmid = cd['q_null_center']
    qlow = cd['q_null_low']
    qhigh = cd['q_null_high']
    band_q = np.isfinite(qlow) & np.isfinite(qhigh)
    ax_q.fill_between(dq[band_q], qlow[band_q], qhigh[band_q],
                      color=NULL_COLOR, alpha=.18,
                      label=f'Shuffle P{LOW_PERCENTILE:g}–P{HIGH_PERCENTILE:g}')
    ax_q.plot(dq[np.isfinite(qobs)], qobs[np.isfinite(qobs)], 'o', ms=7,
              color=OBS_COLOR, label=r'Observado: $q_d$')
    ax_q.plot(dq[np.isfinite(qmid)], qmid[np.isfinite(qmid)], 's', ms=6,
              markerfacecolor='white', markeredgecolor=NULL_COLOR,
              label='Shuffle: media geométrica')
    ax_q.set(xlabel=r'Dimensión $d$', ylabel=r'$q_d=C_{d+1}/C_d$',
             title='C  ·  Cocientes consecutivos (sin ajuste)')
    ax_q.set_xticks(dq)
    ax_q.set_xlim(D_MIN-.35, D_MAX-.65)
    ax_q.set_ylim(0.0, 1.05)
    ax_q.grid(alpha=.22)
    ax_q.legend(loc='best', fontsize=8)
    ax_q.text(.03, .04, rf'$C_{{1,\mathrm{{null}}}}^{{\mathrm{{exacto}}}}={C1_null:.4f}$',
              transform=ax_q.transAxes, ha='left', va='bottom', fontsize=9,
              bbox=dict(facecolor='white', edgecolor='.8', alpha=.85, boxstyle='round,pad=.3'))

    suffix = 'intervalos con signo' if cd_mode == 'signed_intervals' else 'serie directa'
    fig.suptitle(f'{path.name}  |  N={len(x)}  |  DFA/mDFA: serie original  |  '
                 f'C_d: {suffix} (N={len(x_cd)})', fontsize=13)
    return fig


def json_safe(d):
    """Convierte np scalar y NaN a tipos JSON estándar (NaN -> null)."""
    if isinstance(d, dict):
        return {str(k): json_safe(v) for k, v in d.items()}
    if isinstance(d, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return float(d) if np.isfinite(d) else None
    if isinstance(d, (np.integer, int)):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    return d


def main():
    parser = argparse.ArgumentParser(description='Gráfica DFA/mDFA+IAAFT, C_d y q_d+shuffle desde .npy')
    parser.add_argument('npy', nargs='?', default=str(NPY_PATH), help='Archivo .npy 1D')
    parser.add_argument('--method', choices=('dfa','mdfa'), default=DFA_METHOD)
    parser.add_argument('--cd-mode', choices=('signed_intervals','direct'), default=CD_MODE)
    parser.add_argument('--k-mode', choices=('fixed','target'), default=K_MODE)
    parser.add_argument('--k', type=int, default=K_FIXED)
    parser.add_argument('--c1-target', type=float, default=C1_NULL_TARGET)
    args = parser.parse_args()

    if N_SHUFFLES < 1:
        raise ValueError('N_SHUFFLES debe ser >= 1.')
    if D_MIN < 1 or D_MAX != MAX_GAMMA + 1:
        raise ValueError('Se requiere D_MIN>=1 y D_MAX=MAX_GAMMA+1.')
    if not (0 <= LOW_PERCENTILE < HIGH_PERCENTILE <= 100):
        raise ValueError('Percentiles de banda inválidos.')

    path = Path(args.npy).expanduser()
    x = load_1d_npy(path)
    x_cd = cd_input_series(x, args.cd_mode)
    K, n_unique, c1_null = select_K(x_cd, args.k_mode, args.k, args.c1_target)
    print(f'Archivo: {path.resolve()} | N={len(x)}')
    print(f'Método: {args.method.upper()} | Cd: {args.cd_mode} | K={K} | n_unique={n_unique}')

    dfa = compute_dfa_with_iaaft(x, args.method)
    cd = compute_cd_with_shuffle(x_cd, K)
    if cd['fit_observed']['n_fit'] < D_MAX-D_MIN+1:
        warnings.warn('Algunos C_d observados son cero/no finitos; el ajuste excluyó esas dimensiones.')

    fig = make_figure(path, x, dfa, cd, x_cd, K, n_unique, c1_null, args.cd_mode)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_tag = f'K{K}' if args.k_mode == 'fixed' else f'C1target{args.c1_target:g}_K{K}'
    stem = f'{path.stem}_{args.method}_Cd_{args.cd_mode}_{k_tag}'

    if SAVE_PNG:
        png_path = output_dir / f'{stem}.png'
        fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
        print(f'Figura PNG: {png_path.resolve()}')
    if SAVE_PDF:
        pdf_path = output_dir / f'{stem}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f'Figura PDF: {pdf_path.resolve()}')

    summary = {
        'archivo': str(path.resolve()),
        'N': len(x),
        'metodo_DFA': args.method,
        'cd_mode': args.cd_mode,
        'N_CD': len(x_cd),
        'alpha': dfa['alpha'],
        'R2_alpha': dfa['R2_alpha'],
        'xi_NL': dfa['xi_NL'],
        'n_iaaft': N_IAAFT,
        'K': K,
        'n_unique_Cd': n_unique,
        'C1_null_exact': c1_null,
        'lambda_obs': cd['fit_observed']['lambda'],
        'lambda_null_mean': cd['lambda_null_mean'],
        'lambda_null_std': cd['lambda_null_std'],
        'kappa': cd['kappa'],
        'R2_Cd': cd['fit_observed']['R2'],
        'n_fit_Cd': cd['fit_observed']['n_fit'],
        'n_shuffles': N_SHUFFLES,
    }
    if SAVE_JSON:
        json_path = output_dir / f'{stem}_resumen.json'
        json_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False,
                                        allow_nan=False), encoding='utf-8')
        print(f'Resumen JSON: {json_path.resolve()}')
    if SAVE_NPZ:
        npz_path = output_dir / f'{stem}_curvas.npz'
        np.savez_compressed(
            npz_path,
            s=dfa['s'], F_observada=dfa['F'],
            log10_F_IAAFT=dfa['surrogate_logF'],
            C_observada=cd['C_observed'], C_shuffle=cd['C_null_all'],
            q_observada=cd['q_observed'], q_shuffle=cd['q_null_all'],
            lambda_shuffle=cd['lambda_null_values'],
        )
        print(f'Curvas NPZ: {npz_path.resolve()}')

    print('\nRESULTADOS')
    for key in ('alpha','xi_NL','R2_alpha','lambda_obs','lambda_null_mean',
                'lambda_null_std','kappa','R2_Cd','K','n_unique_Cd','C1_null_exact'):
        value = summary[key]
        value_str = f'{value:.6g}' if isinstance(value, float) else str(value)
        print(f'  {key:22s} = {value_str}')
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
