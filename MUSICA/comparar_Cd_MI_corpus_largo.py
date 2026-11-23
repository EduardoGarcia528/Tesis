# -*- coding: utf-8 -*-
"""
COMPARAR Cd / KAPPA VS AUTO-INFORMACION MUTUA (MI) EN CORPUS GRANDE
=================================================================
Corpus de dists(1).py / comparar_Cd_DFA_mDFA_PSD_corpus_largo(1).py:
    data/composers_hybrid.pkl            (necesario si METHODS incluye hybrid)
    data/datos_composers_hybrid.pkl
    data/composers_complexity.pkl        (necesario si METHODS incluye complexity)
    data/datos_composers_complexity.pkl

Se usa SIEMPRE la serie de INTERVALOS CON SIGNO, diff(MIDI), para Cd y MI.
No se recortan ni interpolan series. N_notas > MIN_NOTES_EXCLUSIVE.

1) Cd y kappa:
   C_d = A exp(-lambda_C*d),   d=1,...,8 (configurable)
   kappa = lambda_C / <lambda_C_shuffle>
   K_MODE='fixed' usa K_FIXED; K_MODE='target' selecciona el mayor K entero
   cuyo C1_null teórico NO SUPERA C1_NULL_TARGET (o K=0 si imposible).

2) Auto-MI discreta en bits (frecuencias empíricas):
   I(tau) = I(X_t ; X_(t+tau))
   I_corr(tau) = I_real(tau) - <I_shuffle(tau)>
   I(tau) = I0 exp(-lambda_MI*tau), ajuste semilog de puntos > 0.
   Se guardan exponentes y calidad para MI_real y MI_corr.

3) Cocientes y pendientes locales:
   R_MI = I(1)/I(2),   R_C = C1/C2
   R_C* = R_C / exp(<log(C1_shuffle/C2_shuffle)>)
   Se comparan R_MI vs R_C y R_C*, y sus logaritmos, tanto MI real
   como corregida por shuffle. Para MI_corr la razón solo existe si
   AMBOS I_corr(1) e I_corr(2) son estrictamente positivos.

4) Estadística global y por método/compositor:
   Pearson, Spearman, Spearman parcial por N, vocabulario, K (cuando cambia)
   e indicador de método (al combinar corpus). IC por bootstrap de compositor
   para grupos con >= 3 compositores (opcional); no se pretende que los
   p-valores por melodía sean independientes dentro de compositor.

5) Salidas progresivas y reproducibles, carpeta determinada por hash de
   parámetros: resultados_por_melodia.csv; correlaciones.csv;
   resumen_por_compositor.csv; errores.csv; figuras .png.

Dependencias: numpy, scipy, pandas, matplotlib, mi_libreria.
La función Cd requerida:
    ml.correlation_integrals_rank_ties(x, max_d=MAX_D, K=float(K),
                                      usar_rangos=True)

Windows: para N_WORKERS>1 ejecutar como script .py, no desde una celda
que reimporte el módulo como __main__.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, t as student_t

import mi_libreria as ml


# ========================================================================
# CONFIGURACION: MODIFICAR AQUI
# ========================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

METHODS = ('hybrid', 'complexity')   # ('hybrid',) o ('complexity',)
MIN_NOTES_EXCLUSIVE = 1000         # conservar solamente N_notas > 1000
MAX_SERIES_PER_METHOD = None      # 5 para probar; None para corpus completo

# K de Cd: 'fixed' O 'target'
K_MODE = 'fixed'
K_FIXED = 2
C1_NULL_TARGET = 0.5807           # activo si K_MODE == 'target'

MAX_D = 8
FIT_D_MIN, FIT_D_MAX = 1, 8
MIN_CD_FIT_POINTS = 4
N_SHUFFLES = 200                  # prueba: 5-10, análisis: 100-200
MIN_NULL_VALID_FRACTION = 0.80

TAU_MIN, TAU_MAX = 1, 10
FIT_TAU_MIN, FIT_TAU_MAX = 1, 10
MIN_MI_FIT_POINTS = 3

# Umbrales de análisis secundario con buenos ajustes (NO filtran datos
# del CSV principal ni la tabla estadística general).
MIN_R2_CD_QUALITY = 0.85
MIN_R2_MI_QUALITY = 0.85
MIN_MI_POINTS_QUALITY = 5

MIN_GROUP_SIZE = 5
BOOTSTRAP_BY_COMPOSER = 200       # 0 para desactivar
SEED = 20261120
N_WORKERS = 1                    # Windows: 1 primero; 2-4 si hay memoria
SAVE_PLOTS = True
PLOT_DPI = 160
N_MI_DIAGNOSTICS = 8

# Se usa carpeta exclusiva por configuración para evitar mezclar K, corpus,
# rangos de ajuste, número de shuffles u otros parámetros.
CONFIG = {
    'version': 2,
    'METHODS': list(METHODS),
    'MIN_NOTES_EXCLUSIVE': MIN_NOTES_EXCLUSIVE,
    'MAX_SERIES_PER_METHOD': MAX_SERIES_PER_METHOD,
    'K_MODE': K_MODE,
    'K_FIXED': K_FIXED if K_MODE == 'fixed' else None,
    'C1_NULL_TARGET': C1_NULL_TARGET if K_MODE == 'target' else None,
    'MAX_D': MAX_D,
    'FIT_D_MIN': FIT_D_MIN,
    'FIT_D_MAX': FIT_D_MAX,
    'MIN_CD_FIT_POINTS': MIN_CD_FIT_POINTS,
    'N_SHUFFLES': N_SHUFFLES,
    'MIN_NULL_VALID_FRACTION': MIN_NULL_VALID_FRACTION,
    'TAU_MIN': TAU_MIN,
    'TAU_MAX': TAU_MAX,
    'FIT_TAU_MIN': FIT_TAU_MIN,
    'FIT_TAU_MAX': FIT_TAU_MAX,
    'MIN_MI_FIT_POINTS': MIN_MI_FIT_POINTS,
    'SEED': SEED,
}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, ensure_ascii=False).encode('utf-8')
).hexdigest()[:10]
K_LABEL = f'K{K_FIXED}' if K_MODE == 'fixed' else f'C1_{C1_NULL_TARGET:.4f}'.replace('.', 'p')
OUTPUT_DIR = BASE_DIR / f'resultados_Cd_MI_{"_".join(METHODS)}_{K_LABEL}_{CONFIG_HASH}'


# ========================================================================
# CORPUS Y REPRODUCIBILIDAD
# ========================================================================
def stable_seed(*identifiers: object) -> int:
    payload = json.dumps([SEED, *map(str, identifiers)], ensure_ascii=False)
    return int.from_bytes(hashlib.blake2b(payload.encode('utf-8'), digest_size=8).digest(), 'little')


def valid_midi(raw) -> np.ndarray:
    midi = np.asarray(raw, dtype=float)
    if midi.ndim != 1:
        raise ValueError(f'La melodía no es un array 1D: {midi.shape}')
    if not np.isfinite(midi).all():
        raise ValueError('NaN/Inf en MIDI: no se eliminan notas para no crear intervalos falsos')
    return midi


def load_tasks():
    """Estructura original: composers[compositor][serie] = MIDI 1D."""
    tasks, invalid = [], []
    for method in METHODS:
        path = DATA_DIR / f'composers_{method}.pkl'
        meta_path = DATA_DIR / f'datos_composers_{method}.pkl'
        for required in (path, meta_path):
            if not required.exists():
                raise FileNotFoundError(f'No existe {required}. Coloca el script junto a data/.')
        with path.open('rb') as f:
            composers = pickle.load(f)
        with meta_path.open('rb') as f:
            metadata = pickle.load(f)
        if not isinstance(composers, dict):
            raise TypeError(f'{path} no contiene un dict de compositores')
        if not isinstance(metadata, dict):
            metadata = {}
        total, eligible, broken = 0, 0, 0
        for composer in sorted(composers, key=str):
            series_dict = composers[composer]
            info = metadata.get(composer, {})
            if not isinstance(info, dict):
                info = {}
            for name in sorted(series_dict, key=str):
                total += 1
                try:
                    midi = valid_midi(series_dict[name])
                except Exception as exc:
                    broken += 1
                    invalid.append({'method': method, 'composer': str(composer),
                                    'serie': str(name), 'error': str(exc)})
                    continue
                if midi.size <= MIN_NOTES_EXCLUSIVE:
                    continue
                tasks.append({
                    'method': str(method), 'composer': str(composer), 'serie': str(name),
                    'birth_year': info.get('Birth_year', np.nan),
                    'composer_index': info.get('Indice', np.nan), 'midi': midi,
                })
                eligible += 1
                if MAX_SERIES_PER_METHOD is not None and eligible >= MAX_SERIES_PER_METHOD:
                    break
            if MAX_SERIES_PER_METHOD is not None and eligible >= MAX_SERIES_PER_METHOD:
                break
        print(f'[{method}] total examinadas={total}, N>{MIN_NOTES_EXCLUSIVE}={eligible}, inválidas={broken}', flush=True)
    return tasks, invalid


# ========================================================================
# K Y EL C1 NULL TEORICO (parejas sin reemplazo)
# ========================================================================
def c1_null_exact_from_counts(counts: np.ndarray, k: int) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n < 2:
        return np.nan
    k = int(np.clip(k, 0, len(counts)-1))
    close = int(np.sum(counts*(counts-1)//2))
    for shift in range(1, k+1):
        close += int(np.dot(counts[:-shift], counts[shift:]))
    return float(close / (n*(n-1)/2))


def choose_k(intervals: np.ndarray) -> dict:
    _, counts = np.unique(intervals, return_counts=True)
    q = len(counts)
    if q < 2:
        raise ValueError('Sólo hay un intervalo distinto; no existe decaimiento informativo')
    if K_MODE == 'fixed':
        if not isinstance(K_FIXED, int) or K_FIXED < 0:
            raise ValueError('K_FIXED debe ser entero >= 0')
        if K_FIXED >= q-1:
            raise ValueError(f'K={K_FIXED} cubre todo el vocabulario (n_unique={q})')
        k = K_FIXED
    elif K_MODE == 'target':
        if not (0 < C1_NULL_TARGET < 1):
            raise ValueError('C1_NULL_TARGET debe pertenecer a (0,1)')
        k = 0
        # C1_null es monótona respecto a K; se elige el mayor que no rebasa objetivo.
        for trial in range(q):
            if c1_null_exact_from_counts(counts, trial) <= C1_NULL_TARGET:
                k = trial
            else:
                break
        if k >= q-1:
            raise ValueError('El K objetivo degeneró en C1_null=1')
    else:
        raise ValueError("K_MODE debe ser 'fixed' o 'target'")
    c1 = c1_null_exact_from_counts(counts, k)
    return {
        'K': k, 'n_unique_intervalos': q, 'C1_null_exact': c1,
        'C1_target_reached_below': bool(c1 <= C1_NULL_TARGET) if K_MODE == 'target' else np.nan,
        'C1_target_gap': c1 - C1_NULL_TARGET if K_MODE == 'target' else np.nan,
    }


# ========================================================================
# Cd, AJUSTE EXPONENCIAL
# ========================================================================
def linear_fit(x: np.ndarray, y: np.ndarray, min_points: int) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    result = {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan,
              'rmse': np.nan, 'n_fit': int(x.size)}
    if x.size < min_points or np.unique(x).size < 2:
        return result
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope*x + intercept)
    sst = float(np.sum((y-y.mean())**2))
    result.update(slope=float(slope), intercept=float(intercept),
                  r2=float(1-np.sum(resid**2)/sst) if sst > 0 else np.nan,
                  rmse=float(np.sqrt(np.mean(resid**2))))
    return result


def compute_cd(intervals: np.ndarray, k: int) -> np.ndarray:
    # Usa el K elegido: NO dejar un K=0 fijo por equivocación.
    c = ml.correlation_integrals_rank_ties(
        np.asarray(intervals, dtype=np.float64),
        max_d=MAX_D, K=float(k), usar_rangos=True,
    )
    c = np.asarray(c, dtype=float).ravel()
    if c.size != MAX_D+1:
        raise ValueError(f'mi_libreria devolvió {c.size} Cd; se esperaban {MAX_D+1}')
    return c


def fit_cd(c: np.ndarray) -> dict:
    d = np.arange(len(c), dtype=int)
    fit_window = (d >= FIT_D_MIN) & (d <= FIT_D_MAX)
    good = fit_window & np.isfinite(c) & (c > 0)
    fit = linear_fit(d[good], np.log(c[good]), MIN_CD_FIT_POINTS)
    return {
        'lambda': -fit['slope'] if np.isfinite(fit['slope']) else np.nan,
        'A': float(np.exp(fit['intercept'])) if np.isfinite(fit['intercept']) else np.nan,
        'r2': fit['r2'], 'rmse': fit['rmse'], 'n_fit': fit['n_fit'],
        'full': bool(np.all(good[fit_window])),
    }


# ========================================================================
# MI DISCRETA: códigos ordenados conservan todos los intervalos con signo
# ========================================================================
def mi_discrete_lags(codes: np.ndarray, q: int, taus: np.ndarray) -> np.ndarray:
    """MI empírica, bits, sin discretización adicional ni pseudocuentas."""
    codes = np.asarray(codes, dtype=np.int64).ravel()
    out = np.full(taus.size, np.nan, dtype=float)
    for i, tau in enumerate(taus):
        tau = int(tau)
        if tau < 1 or tau >= codes.size:
            continue
        a, b = codes[:-tau], codes[tau:]
        n = a.size
        px_counts = np.bincount(a, minlength=q)
        py_counts = np.bincount(b, minlength=q)
        if q <= 128:  # histogramas densos eficientes para vocabulario habitual
            ct = np.bincount(a*q + b, minlength=q*q).reshape(q, q)
            ii, jj = np.nonzero(ct)
            nij = ct[ii, jj].astype(float)
        else:  # evita una matriz q*q muy grande
            keys, counts = np.unique(a*q+b, return_counts=True)
            ii, jj = keys//q, keys%q
            nij = counts.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = (nij/n) * np.log2((nij*n)/(px_counts[ii]*py_counts[jj]))
        out[i] = float(np.sum(terms))
    return out


def fit_mi(taus: np.ndarray, values: np.ndarray) -> dict:
    use_window = (taus >= FIT_TAU_MIN) & (taus <= FIT_TAU_MAX)
    good = use_window & np.isfinite(values) & (values > 0)
    x = taus[good]
    fit = linear_fit(x, np.log(values[good]), MIN_MI_FIT_POINTS)
    return {
        'lambda': -fit['slope'] if np.isfinite(fit['slope']) else np.nan,
        'I0': float(np.exp(fit['intercept'])) if np.isfinite(fit['intercept']) else np.nan,
        'r2': fit['r2'], 'rmse': fit['rmse'], 'n_fit': fit['n_fit'],
        'full': bool(np.all(good[use_window])),
        'contiguous': bool(x.size and np.all(np.diff(x) == 1)),
        'taus_used': '|'.join(map(str, x)),
    }


def positive_ratio(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
        return float(a/b)
    return np.nan


def positive_log(value: float) -> float:
    return float(np.log(value)) if np.isfinite(value) and value > 0 else np.nan


# ========================================================================
# NULOS PARA Cd, R_C, MI: SHUFFLE DE INTERVALOS CON SIGNO
# ========================================================================
def calculate_nulls(intervals, codes, q, k, taus, rng):
    null_lambdas = []
    c1_null, c2_null, null_ratios = [], [], []
    mi_null = np.full((N_SHUFFLES, taus.size), np.nan)
    for i in range(N_SHUFFLES):
        idx = rng.permutation(len(intervals))
        perm_intervals = intervals[idx]
        perm_codes = codes[idx]
        cs = compute_cd(perm_intervals, k)
        fitted = fit_cd(cs)
        c1_null.append(float(cs[1]))
        c2_null.append(float(cs[2]))
        null_ratios.append(positive_ratio(cs[1], cs[2]))
        if np.isfinite(fitted['lambda']) and fitted['lambda'] > 0:
            null_lambdas.append(fitted['lambda'])
        mi_null[i, :] = mi_discrete_lags(perm_codes, q, taus)

    valid_c = np.asarray(null_ratios, float)
    valid_c = valid_c[np.isfinite(valid_c) & (valid_c > 0)]
    null_fraction = len(null_lambdas)/N_SHUFFLES
    ok = null_fraction >= MIN_NULL_VALID_FRACTION
    mi_mean = np.nanmean(mi_null, axis=0)
    mi_std = np.nanstd(mi_null, axis=0, ddof=1) if N_SHUFFLES > 1 else np.full(taus.size, np.nan)
    geo = float(np.exp(np.mean(np.log(valid_c)))) if valid_c.size else np.nan
    return {
        'lambda_null_mean': float(np.mean(null_lambdas)) if ok else np.nan,
        'lambda_null_std': float(np.std(null_lambdas, ddof=1)) if ok and len(null_lambdas)>1 else np.nan,
        'n_null_lambda_valid': len(null_lambdas),
        'null_valid_fraction': null_fraction,
        'C1_null_shuffle_mean': float(np.nanmean(c1_null)),
        'C2_null_shuffle_mean': float(np.nanmean(c2_null)),
        'C_ratio_null_geomean': geo,
        'C_ratio_null_mean': float(np.mean(valid_c)) if valid_c.size else np.nan,
        'MI_null_mean': mi_mean, 'MI_null_std': mi_std,
    }


# ========================================================================
# PROCESAMIENTO POR MELODIA
# ========================================================================
def process_task(task: dict) -> dict:
    row = {key: val for key, val in task.items() if key != 'midi'}
    method, composer, name = row['method'], row['composer'], row['serie']
    row.update(estado='error', error='', rng_seed=stable_seed(method, composer, name),
               K_mode=K_MODE, C1_target=C1_NULL_TARGET if K_MODE=='target' else np.nan,
               n_shuffles=N_SHUFFLES)
    try:
        midi = valid_midi(task['midi'])
        intervals = np.diff(midi)  # IMPORTANTE: intervalos CON SIGNO
        if np.std(intervals) <= 1e-12:
            raise ValueError('Todos los intervalos son idénticos')
        _, codes = np.unique(intervals, return_inverse=True)
        q = int(codes.max()+1)
        row.update(N_notas=int(midi.size), N_intervalos=int(intervals.size),
                   n_unique_MIDI=int(np.unique(midi).size),
                   std_intervalos=float(np.std(intervals)))
        row.update(choose_k(intervals))
        k = int(row['K'])
        taus = np.arange(TAU_MIN, TAU_MAX+1, dtype=int)
        cs = compute_cd(intervals, k)
        fc = fit_cd(cs)
        row.update(lambda_obs=fc['lambda'], A_Cd=fc['A'],
                   R2_Cd=fc['r2'], RMSE_log_Cd=fc['rmse'],
                   n_fit_Cd=fc['n_fit'], fit_Cd_full=fc['full'])
        for d, val in enumerate(cs):
            row[f'C{d}_obs'] = float(val)

        mi_real = mi_discrete_lags(codes, q, taus)
        null = calculate_nulls(intervals, codes, q, k, taus,
                               np.random.default_rng(row['rng_seed']))
        null_mean, null_std = null.pop('MI_null_mean'), null.pop('MI_null_std')
        row.update(null)
        row['kappa'] = positive_ratio(row['lambda_obs'], row['lambda_null_mean'])

        c_ratio = positive_ratio(cs[1], cs[2])
        c_ratio_null = row['C_ratio_null_geomean']
        c_star = positive_ratio(c_ratio, c_ratio_null)
        row.update(C_ratio=c_ratio, log_C_ratio=positive_log(c_ratio),
                   C_ratio_corrected=c_star,
                   log_C_ratio_corrected=positive_log(c_star))

        mi_corr = mi_real - null_mean
        for t, observed, null_m, null_s, corrected in zip(taus, mi_real, null_mean, null_std, mi_corr):
            row[f'I_real_{t}'] = float(observed)
            row[f'I_null_mean_{t}'] = float(null_m)
            row[f'I_null_std_{t}'] = float(null_s)
            row[f'I_corr_{t}'] = float(corrected)
        for mode, vals in (('real', mi_real), ('corr', mi_corr)):
            fm = fit_mi(taus, vals)
            row.update({
                f'lambda_MI_{mode}': fm['lambda'],
                f'I0_MI_{mode}': fm['I0'],
                f'R2_MI_{mode}': fm['r2'],
                f'RMSE_log_MI_{mode}': fm['rmse'],
                f'n_fit_MI_{mode}': fm['n_fit'],
                f'fit_MI_{mode}_full': fm['full'],
                f'fit_MI_{mode}_contiguous': fm['contiguous'],
                f'taus_MI_{mode}_used': fm['taus_used'],
            })
            i1 = float(vals[1-TAU_MIN])  # TAU_MIN se valida =1
            i2 = float(vals[2-TAU_MIN])
            r = positive_ratio(i1, i2)
            row[f'MI_ratio_{mode}'] = r
            row[f'log_MI_ratio_{mode}'] = positive_log(r)
            # Para corregida es particularmente relevante evaluar
            # señal respecto a la dispersión de la MI de los shuffles.
            row[f'I1_{mode}'] = i1
            row[f'I2_{mode}'] = i2
        for t in (1, 2):
            j = t-TAU_MIN
            row[f'Z_MI_corr_{t}'] = float(mi_corr[j]/null_std[j]) if null_std[j]>0 else np.nan

        row.update(estado='ok', error='')
    except Exception as exc:
        row.update(estado='error', error=f'{type(exc).__name__}: {exc}')
    return row


# ========================================================================
# CORRELACIONES Y CONTROLES
# ========================================================================
def comparison_specs():
    specs = []
    for mode in ('real', 'corr'):
        for cd_metric in ('lambda_obs', 'kappa'):
            specs.append((f'lambda_MI_{mode}', cd_metric, 'exponente', mode))
        for mi_kind, c_kind in (('MI_ratio', 'C_ratio'),
                                ('log_MI_ratio', 'log_C_ratio')):
            for suffix in ('', '_corrected'):
                specs.append((f'{mi_kind}_{mode}', f'{c_kind}{suffix}', 'cociente', mode))
    return specs


def correlations_basic(x: np.ndarray, y: np.ndarray):
    n = x.size
    out = {'pearson_r': np.nan, 'pearson_p': np.nan,
           'spearman_rho': np.nan, 'spearman_p': np.nan}
    if n < MIN_GROUP_SIZE or np.unique(x).size < 2 or np.unique(y).size < 2:
        return out
    try:
        pr, sr = pearsonr(x, y), spearmanr(x, y)
        out.update(pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
                   spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue))
    except ValueError:
        pass
    return out


def partial_spearman(group: pd.DataFrame, xcol: str, ycol: str):
    """Correlación de rangos tras residualizar covariables (exploratoria)."""
    if len(group) < MIN_GROUP_SIZE+3:
        return np.nan, np.nan
    x, y = rankdata(group[xcol].to_numpy(float)), rankdata(group[ycol].to_numpy(float))
    terms = [rankdata(np.log(group['N_intervalos'].to_numpy(float))),
             rankdata(group['n_unique_intervalos'].to_numpy(float))]
    if group['K'].nunique() > 1:
        terms.append(rankdata(group['K'].to_numpy(float)))
    if group['method'].nunique() > 1:
        dum = pd.get_dummies(group['method'], drop_first=True).to_numpy(dtype=float)
        terms.extend(dum[:, i] for i in range(dum.shape[1]))
    z = np.column_stack([np.ones(len(group))] + terms)
    rx = x - z@np.linalg.lstsq(z, x, rcond=None)[0]
    ry = y - z@np.linalg.lstsq(z, y, rcond=None)[0]
    if np.std(rx)<1e-12 or np.std(ry)<1e-12:
        return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(group) - int(np.linalg.matrix_rank(z)) - 1
    p = float(2*student_t.sf(abs(r)*np.sqrt(dof/max(1-r*r,1e-14)), dof)) if dof>0 else np.nan
    return r, p


def bootstrap_spearman_by_composer(group, xcol, ycol, rng_seed):
    if BOOTSTRAP_BY_COMPOSER<=0 or group['composer'].nunique()<3:
        return np.nan, np.nan
    # La unidad de remuestreo es method/composer: no se parte un compositor.
    blocks = [s[[xcol, ycol]].to_numpy(float)
              for _, s in group.groupby(['method', 'composer'], sort=True)]
    rng = np.random.default_rng(rng_seed)
    rhos = []
    for _ in range(BOOTSTRAP_BY_COMPOSER):
        draw = rng.integers(0, len(blocks), size=len(blocks))
        xy = np.concatenate([blocks[j] for j in draw], axis=0)
        if len(xy)<MIN_GROUP_SIZE or np.ptp(xy[:,0])==0 or np.ptp(xy[:,1])==0:
            continue
        rhos.append(float(spearmanr(xy[:,0], xy[:,1]).statistic))
    if len(rhos)<20:
        return np.nan, np.nan
    return tuple(map(float, np.percentile(rhos, [2.5, 97.5])))


def build_correlations(df: pd.DataFrame) -> pd.DataFrame:
    groupings = [('global', 'todos', df)]
    groupings += [('method', str(m), s) for m, s in df.groupby('method', sort=True)]
    groupings += [('composer', f'{m} / {c}', s) for (m, c), s in df.groupby(['method','composer'], sort=True)]
    rows = []
    for xcol, ycol, kind, mode in comparison_specs():
        for group_type, group_name, group in groupings:
            selections = [('all_valid', group)]
            if kind == 'exponente':
                qual = (pd.to_numeric(group['R2_Cd'], errors='coerce') >= MIN_R2_CD_QUALITY) & (
                    pd.to_numeric(group[f'R2_MI_{mode}'], errors='coerce') >= MIN_R2_MI_QUALITY) & (
                    pd.to_numeric(group[f'n_fit_MI_{mode}'], errors='coerce') >= MIN_MI_POINTS_QUALITY) & (
                    pd.to_numeric(group[xcol], errors='coerce') > 0) & (
                    pd.to_numeric(group['lambda_obs'], errors='coerce') > 0)
                selections.append(('good_exponential_fits', group.loc[qual]))
            for selection, subset in selections:
                columns = ['method','composer','N_intervalos','n_unique_intervalos','K',xcol,ycol]
                data = subset[columns].copy()
                for col in ('N_intervalos','n_unique_intervalos','K',xcol,ycol):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                data.replace([np.inf,-np.inf], np.nan, inplace=True)
                data.dropna(inplace=True)
                data = data[data.N_intervalos > 0]
                x, y = data[xcol].to_numpy(float), data[ycol].to_numpy(float)
                corr = correlations_basic(x, y)
                part_r, part_p = partial_spearman(data, xcol, ycol)
                ci_low, ci_high = bootstrap_spearman_by_composer(
                    data, xcol, ycol,
                    stable_seed('bootstrap', group_type, group_name, selection, xcol, ycol),
                )
                rows.append({
                    'comparison': f'{xcol}_vs_{ycol}',
                    'x': xcol, 'y': ycol, 'tipo': kind, 'MI_mode': mode,
                    'quality_subset': selection, 'group_type': group_type,
                    'group': group_name, 'n': len(data),
                    'n_composers': data.composer.nunique(),
                    **corr, 'rho_partial_N_vocab_K_method': part_r,
                    'p_partial_approx': part_p,
                    'spearman_boot_composer_ci2p5': ci_low,
                    'spearman_boot_composer_ci97p5': ci_high,
                })
    return pd.DataFrame(rows)


# ========================================================================
# FIGURAS
# ========================================================================
METHOD_COLORS = {'hybrid': '#1f77b4', 'complexity': '#d62728'}


def scatter_panel(ax, df, xcol, ycol, xlabel, ylabel):
    for method, sub in df.groupby('method', sort=True):
        sub = sub[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue
        ax.scatter(sub[xcol], sub[ycol], s=14, alpha=0.45,
                   label=method, color=METHOD_COLORS.get(method))
    valid = df[[xcol,ycol]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(valid):
        stats = correlations_basic(valid[xcol].to_numpy(float), valid[ycol].to_numpy(float))
        ax.text(0.02,0.98, f'n={len(valid)}\nSpearman={stats["spearman_rho"]:.3f}\n'
                f'Pearson={stats["pearson_r"]:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.18)


def draw_exponents(df, label, mode):
    fig, axes = plt.subplots(1,2,figsize=(11.8,4.6),constrained_layout=True)
    for ax, ycol, lab in zip(axes, ('lambda_obs','kappa'),
                              (r'$\lambda_{C,obs}$',r'$\kappa_C$')):
        scatter_panel(ax, df, f'lambda_MI_{mode}', ycol,
                      rf'$\lambda_{{MI,{mode}}}$', lab)
    axes[1].legend(fontsize=9, loc='best')
    fig.suptitle(f'{label}: decaimiento de MI ({mode}) vs Cd; K={K_LABEL}')
    fig.savefig(OUTPUT_DIR / f'exponentes_{mode}_{label}.png', dpi=PLOT_DPI)
    plt.close(fig)


def draw_ratios(df, label, mode):
    pairs = [('MI_ratio_'+mode, 'C_ratio', r'$I(1)/I(2)$',r'$C_1/C_2$'),
             ('MI_ratio_'+mode, 'C_ratio_corrected',r'$I(1)/I(2)$',r'$R_C^*$'),
             ('log_MI_ratio_'+mode,'log_C_ratio',r'$\ln[I(1)/I(2)]$',r'$\ln(C_1/C_2)$'),
             ('log_MI_ratio_'+mode,'log_C_ratio_corrected',
              r'$\ln[I(1)/I(2)]$',r'$\ln R_C^*$')]
    fig, axs = plt.subplots(2,2,figsize=(11.6,8.3),constrained_layout=True)
    for ax,(xcol,ycol,xlab,ylab) in zip(axs.flat,pairs):
        scatter_panel(ax,df,xcol,ycol,xlab,ylab)
    axs[0,1].legend(fontsize=8)
    fig.suptitle(f'{label}: cocientes MI({mode}) vs Cd; K={K_LABEL}')
    fig.savefig(OUTPUT_DIR / f'cocientes_{mode}_{label}.png', dpi=PLOT_DPI)
    plt.close(fig)


def draw_confounders(df, label):
    fig, axes = plt.subplots(2,3,figsize=(13,7.8),constrained_layout=True)
    for row, ycol in enumerate(('lambda_obs','kappa')):
        for col, xcol in enumerate(('N_intervalos','n_unique_intervalos','K')):
            ax = axes[row,col]
            scatter_panel(ax, df,xcol,ycol,xcol,ycol)
            if xcol=='N_intervalos':
                ax.set_xscale('log')
    fig.suptitle(f'{label}: dependencia de longitud, vocabulario y K')
    fig.savefig(OUTPUT_DIR / f'controles_{label}.png', dpi=PLOT_DPI)
    plt.close(fig)


def draw_mi_diagnostics(df):
    """Muestras representativas por R2 (mejores y peores) de MI real."""
    sample = df[np.isfinite(df['R2_MI_real'])].copy()
    if sample.empty:
        return
    half = max(1, N_MI_DIAGNOSTICS//2)
    ids = pd.concat([sample.nlargest(half,'R2_MI_real'),
                     sample.nsmallest(half,'R2_MI_real')]).drop_duplicates(
                         subset=['method','composer','serie']).head(N_MI_DIAGNOSTICS)
    n = len(ids)
    fig, axes = plt.subplots(math.ceil(n/2),2,figsize=(12,3.5*math.ceil(n/2)),
                             constrained_layout=True,squeeze=False)
    taus = np.arange(TAU_MIN, TAU_MAX+1)
    for ax, (_, row) in zip(axes.flat, ids.iterrows()):
        real = np.array([row[f'I_real_{t}'] for t in taus],float)
        null = np.array([row[f'I_null_mean_{t}'] for t in taus],float)
        corr = np.array([row[f'I_corr_{t}'] for t in taus],float)
        ax.semilogy(taus,real,'o-',label='I_real',ms=4)
        ax.semilogy(taus,null,'--',label='<I_shuffle>',alpha=.8)
        pos = corr>0
        ax.semilogy(taus[pos],corr[pos],'s',label='I_corr positivo',ms=4)
        if np.isfinite(row['lambda_MI_real']):
            ax.semilogy(taus,row['I0_MI_real']*np.exp(-row['lambda_MI_real']*taus),
                        ':',color='black', label='ajuste I_real')
        ax.set_title(f'{row["method"]}/{str(row["composer"])[:18]} / '
                     f'{str(row["serie"])[:24]} | R²={row["R2_MI_real"]:.2f}',fontsize=9)
        ax.set(xlabel='Retardo τ (notas)',ylabel='MI (bits)')
        ax.grid(alpha=.2)
    for ax in axes.flat[n:]:
        ax.axis('off')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Curvas de MI observada, nulo y corregida: ejemplos extremos de calidad')
    fig.savefig(OUTPUT_DIR/'diagnostico_curvas_MI.png',dpi=PLOT_DPI)
    plt.close(fig)


# ========================================================================
# GUARDADO PROGRESIVO / RESUMEN
# ========================================================================
BASE_COLUMNS = [
    'method','composer','serie','birth_year','composer_index',
    'N_notas','N_intervalos','n_unique_MIDI','n_unique_intervalos','std_intervalos',
    'K_mode','K','C1_target','C1_null_exact','C1_target_reached_below','C1_target_gap',
    'rng_seed','n_shuffles','estado','error',
    'lambda_obs','A_Cd','R2_Cd','RMSE_log_Cd','n_fit_Cd','fit_Cd_full',
    'lambda_null_mean','lambda_null_std','n_null_lambda_valid','null_valid_fraction',
    'kappa','C1_null_shuffle_mean','C2_null_shuffle_mean',
    'C_ratio','log_C_ratio','C_ratio_null_geomean','C_ratio_null_mean',
    'C_ratio_corrected','log_C_ratio_corrected',
]
for mode in ('real','corr'):
    BASE_COLUMNS += [f'lambda_MI_{mode}',f'I0_MI_{mode}',f'R2_MI_{mode}',
                     f'RMSE_log_MI_{mode}',f'n_fit_MI_{mode}',
                     f'fit_MI_{mode}_full',f'fit_MI_{mode}_contiguous',
                     f'taus_MI_{mode}_used',f'I1_{mode}',f'I2_{mode}',
                     f'MI_ratio_{mode}',f'log_MI_ratio_{mode}']
BASE_COLUMNS += ['Z_MI_corr_1','Z_MI_corr_2']
BASE_COLUMNS += [f'C{d}_obs' for d in range(MAX_D+1)]
for t in range(TAU_MIN,TAU_MAX+1):
    BASE_COLUMNS += [f'I_real_{t}',f'I_null_mean_{t}',f'I_null_std_{t}',f'I_corr_{t}']


def append_row(path: Path, row: dict):
    first = not path.exists() or path.stat().st_size==0
    with path.open('a',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=BASE_COLUMNS,extrasaction='ignore')
        if first:
            writer.writeheader()
        writer.writerow({key:row.get(key,'') for key in BASE_COLUMNS})


def ensure_config():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    cfgpath = OUTPUT_DIR/'configuracion.json'
    if cfgpath.exists():
        old = json.loads(cfgpath.read_text(encoding='utf-8'))
        if old != CONFIG:
            raise RuntimeError('La configuración previa difiere: revisa '+str(cfgpath))
    else:
        cfgpath.write_text(json.dumps(CONFIG,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    if not METHODS or any(m not in {'hybrid','complexity'} for m in METHODS):
        raise ValueError("METHODS debe contener 'hybrid' y/o 'complexity'")
    if K_MODE not in ('fixed','target'):
        raise ValueError("K_MODE debe ser 'fixed' o 'target'")
    if not (1 <= FIT_D_MIN < FIT_D_MAX <= MAX_D):
        raise ValueError('Rango FIT_D inválido')
    if not (TAU_MIN == 1 and TAU_MIN < 2 <= TAU_MAX and
            TAU_MIN <= FIT_TAU_MIN < FIT_TAU_MAX <= TAU_MAX):
        raise ValueError('Rangos MI inválidos: se necesita TAU_MIN=1 y TAU_MAX>=2')
    if not (isinstance(N_SHUFFLES,int) and N_SHUFFLES>=2):
        raise ValueError('N_SHUFFLES debe ser al menos 2')
    if not (0 < MIN_NULL_VALID_FRACTION <=1):
        raise ValueError('MIN_NULL_VALID_FRACTION debe pertenecer a (0,1]')
    ensure_config()
    tasks, invalid = load_tasks()
    pd.DataFrame(invalid,columns=['method','composer','serie','error']).to_csv(
        OUTPUT_DIR/'series_invalidas_en_entrada.csv',index=False)
    result_path = OUTPUT_DIR/'resultados_por_melodia.csv'
    if result_path.exists():
        prev = pd.read_csv(result_path,low_memory=False)
        if not {'method','composer','serie'}.issubset(prev.columns):
            raise RuntimeError('CSV previo sin columnas ID esperadas')
        already = set(zip(prev.method.astype(str),prev.composer.astype(str),prev.serie.astype(str)))
    else:
        already = set()
    pending = [t for t in tasks if (t['method'],t['composer'],t['serie']) not in already]
    print(f'Elegibles={len(tasks)}, completadas={len(tasks)-len(pending)}, pendientes={len(pending)}',flush=True)
    if N_WORKERS == 1:
        iterator = map(process_task,pending)
        for i,row in enumerate(iterator,1):
            append_row(result_path,row)
            if row['estado']=='error':
                print(f'  ERROR {row["method"]}/{row["composer"]}/{row["serie"]}: {row["error"]}',flush=True)
            if i==1 or i%10==0:
                print(f'Procesadas {i}/{len(pending)} nuevas melodías',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for i,row in enumerate(pool.map(process_task,pending,chunksize=1),1):
                append_row(result_path,row)
                if row['estado']=='error':
                    print(f'  ERROR {row["method"]}/{row["composer"]}/{row["serie"]}: {row["error"]}',flush=True)
                if i==1 or i%10==0:
                    print(f'Procesadas {i}/{len(pending)} nuevas melodías',flush=True)
    if not result_path.exists():
        print('No se encontró ninguna serie elegible.')
        return
    all_df = pd.read_csv(result_path,low_memory=False)
    all_df.drop_duplicates(subset=['method','composer','serie'],keep='last',inplace=True)
    all_df.to_csv(result_path,index=False,encoding='utf-8')
    all_df[all_df.estado!='ok'].to_csv(OUTPUT_DIR/'errores.csv',index=False)
    df = all_df[all_df.estado=='ok'].copy()
    if df.empty:
        print('No se obtuvieron métricas; consultar errores.csv')
        return
    metrics = ['N_notas','N_intervalos','n_unique_intervalos','K','R2_Cd',
               'lambda_obs','kappa','lambda_MI_real','lambda_MI_corr',
               'R2_MI_real','R2_MI_corr','n_fit_MI_real','n_fit_MI_corr',
               'C_ratio','MI_ratio_real','MI_ratio_corr']
    for col in metrics + [c for c in df if c.startswith(('I_real_','I_null_mean_','I_corr_'))]:
        if col in df:
            df[col] = pd.to_numeric(df[col],errors='coerce')
    summary = df.groupby(['method','composer'],as_index=False).agg(
        n_melodias=('serie','size'),mediana_N=('N_notas','median'),
        mediana_K=('K','median'),mediana_lambda_C=('lambda_obs','median'),
        mediana_kappa=('kappa','median'),mediana_lambda_MI_real=('lambda_MI_real','median'),
        mediana_lambda_MI_corr=('lambda_MI_corr','median'),
        mediana_ratio_MI=('MI_ratio_real','median'),mediana_ratio_C=('C_ratio','median'))
    summary.to_csv(OUTPUT_DIR/'resumen_por_compositor.csv',index=False)
    corr = build_correlations(df)
    corr.to_csv(OUTPUT_DIR/'correlaciones.csv',index=False)
    key = corr[(corr.group_type.isin(['global','method'])) &
               (corr.quality_subset=='all_valid') & (corr.tipo=='exponente')]
    print('\nCORRELACIONES DE EXPONENTES:')
    print(key[['group','comparison','n','spearman_rho','pearson_r']].to_string(index=False))
    print('\nMelodías procesadas por corpus:')
    print(df.groupby('method').size().to_string())
    if SAVE_PLOTS:
        plots = [('global',df)] + [(str(m),s) for m,s in df.groupby('method')]
        for label, sub in plots:
            for mode in ('real','corr'):
                draw_exponents(sub,label,mode)
                draw_ratios(sub,label,mode)
            draw_confounders(sub,label)
        if N_MI_DIAGNOSTICS>0:
            draw_mi_diagnostics(df)
    print(f'\nResultados: {OUTPUT_DIR.resolve()}')
    print('Archivos principales: resultados_por_melodia.csv, correlaciones.csv')


if __name__ == '__main__':
    main()
