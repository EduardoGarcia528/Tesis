# -*- coding: utf-8 -*-
"""
COMPARAR Cd (INTERVALOS CON SIGNO) VS DFA/mDFA/PSD (ALTURAS MIDI)
================================================================
Corpus:
    data/composers_hybrid.pkl
    data/datos_composers_hybrid.pkl
    data/composers_complexity.pkl
    data/datos_composers_complexity.pkl

Las series se cargan igual que en dists(1).py:
    composers[compositor][serie] = array 1D de alturas MIDI

Criterios:
    - N_notas > 1000 (sin recortar ni interpolar).
    - Cd: diferencias MIDI CON SIGNO, rangos con empates, K=2 fijo.
    - Ajuste: log(C_d) = log(A) - lambda_obs*d para d=1,...,8.
    - Nulo: shuffle DE LOS INTERVALOS, NO de las alturas MIDI;
      lambda_null = promedio de los lambda ajustados a cada shuffle.
    - kappa = lambda_obs/lambda_null.
    - alpha: seleccionable entre DFA y magnitud-DFA (mDFA) en MIDI.
      mDFA: |diff(MIDI)| -> centrar -> integrar -> DFA -> F(s)/s.
      Para mDFA, las ventanas se toman desde el inicio (revSeg=False),
      como en indice_no_linealidad_mdfa(2).py.
    - beta: PSD Welch sobre las alturas MIDI, ajustando S(f)~f^(-beta).

Salidas: CSV por melodía, resumen de correlaciones globales/por método,
resumen por compositor y gráficas. Reanudación automática cuando la
configuración es idéntica (se guarda después de cada melodía).

Dependencias: numpy, pandas, scipy, matplotlib, mi_libreria.
La función utilizada en mi_libreria es:
    correlation_integrals_rank_ties(x, max_d=8, K=2, usar_rangos=True)

Nota: 'alfa' y 'beta' son pendientes globales en rangos específicos de
escalas. Comparar también con R² y sensibilidad al límite N/8.

Se conserva la columna alpha para cualquiera de los dos métodos; la columna
metodo_DFA identifica la elección. Por compatibilidad, las columnas R2_DFA,
RMSE_DFA, etc., representan la curva elegida (DFA o mDFA).

No se calcula xi ni se generan surrogados IAAFT: este programa estudia el
exponente alpha y su relación con lambda_obs, kappa y beta. En mDFA, alpha
es un exponente de la magnitud de los incrementos; NO debe usarse la relación
beta = 2*alpha-1 con la PSD de las alturas MIDI.
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
matplotlib.use('Agg')  # guardar figuras sin requerir ventana de GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr, rankdata, t as t_dist

import mi_libreria as ml


# ================================================================
# CONFIGURACIÓN
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

# Elegir "dfa" para el análisis original o "mdfa" para magnitud-DFA.
# Los dos trabajan sobre la serie de alturas MIDI (mDFA usa |diff(MIDI)|).
DFA_METHOD = 'mdfa'  # opciones: 'dfa', 'mdfa'

# En mDFA, por defecto se conserva la malla de tamaños s de tu código
# indice_no_linealidad_mdfa(2).py: s=20,21,...,min(500,(N-1)//4).
# False permite usar las 32 escalas geométricas del script de corpus.
MDFA_REFERENCE_GRID = True
MDFA_S_MAX_CAP = 500

# Directorio nuevo por método: nunca se mezclan resultados DFA/mDFA
# ni se modifica la salida de versiones anteriores del programa.
OUTPUT_DIR = BASE_DIR / f'resultados_Cd_{DFA_METHOD.upper()}_PSD_corpus_largo_v2'
METHODS = ('hybrid', 'complexity')
MIN_NOTES_EXCLUSIVE = 1000  # N > 1000, no N >= 1000

# Parámetros Cd (igual enfoque que en el estudio de cuartetos K=2)
K_FIXED = 2
MAX_D = 8
FIT_D_MIN, FIT_D_MAX = 1, 8
MIN_C_FIT_POINTS = 4
N_SHUFFLES = 100       # prueba rápida: 10; cálculo definitivo: 100 o 200
MIN_NULL_VALID_FRACTION = 0.8

# DFA / mDFA sobre alturas MIDI
DFA_ORDER = 2
DFA_S_MIN = 20
DFA_N_SCALES = 32
DFA_MAX_DIVISOR = 4   # s_max = N//4
DFA_SENSITIVITY_DIVISOR = 8

# PSD Welch sobre alturas MIDI
PSD_MAX_NPERSEG = 4096
PSD_MIN_NPERSEG = 512
PSD_F_MAX = 1.0 / DFA_S_MIN  # ciclos por nota; concordante con s_min DFA
PSD_MIN_FIT_POINTS = 10

# Procesamiento y diagnóstico
SEED = 20261103
N_WORKERS = 1            # Windows: empezar en 1; cambiar a 4 para paralelizar
MAX_SERIES_PER_METHOD = None   # opcional: 10 para ensayo, luego None
MIN_GROUP_SIZE = 5
BOOTSTRAP_BY_COMPOSER = 500  # 0 desactiva intervalos por remuestreo de compositor
SAVE_PLOTS = True
DPI = 170

CONFIG_KEYS = {
    'DFA_METHOD': DFA_METHOD,
    'MDFA_REFERENCE_GRID': MDFA_REFERENCE_GRID,
    'MDFA_S_MAX_CAP': MDFA_S_MAX_CAP,
    'METHODS': list(METHODS),
    'MIN_NOTES_EXCLUSIVE': MIN_NOTES_EXCLUSIVE,
    'K_FIXED': K_FIXED,
    'MAX_D': MAX_D,
    'FIT_D_MIN': FIT_D_MIN,
    'FIT_D_MAX': FIT_D_MAX,
    'MIN_C_FIT_POINTS': MIN_C_FIT_POINTS,
    'N_SHUFFLES': N_SHUFFLES,
    'MIN_NULL_VALID_FRACTION': MIN_NULL_VALID_FRACTION,
    'DFA_ORDER': DFA_ORDER,
    'DFA_S_MIN': DFA_S_MIN,
    'DFA_N_SCALES': DFA_N_SCALES,
    'DFA_MAX_DIVISOR': DFA_MAX_DIVISOR,
    'DFA_SENSITIVITY_DIVISOR': DFA_SENSITIVITY_DIVISOR,
    'PSD_MAX_NPERSEG': PSD_MAX_NPERSEG,
    'PSD_MIN_NPERSEG': PSD_MIN_NPERSEG,
    'PSD_F_MAX': PSD_F_MAX,
    'PSD_MIN_FIT_POINTS': PSD_MIN_FIT_POINTS,
    'SEED': SEED,
}


# ================================================================
# EXTRACCIÓN Y REPRODUCIBILIDAD
# ================================================================
def stable_seed(method: str, composer: str, name: str) -> int:
    msg = json.dumps([SEED, method, composer, name], ensure_ascii=False).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), 'little')


def valid_midi(raw):
    x = np.asarray(raw, dtype=float)
    if x.ndim != 1:
        raise ValueError(f'La serie no es un array 1D (shape={x.shape}).')
    if not np.all(np.isfinite(x)):
        raise ValueError('La serie incluye NaN/inf; no se eliminan notas para no crear intervalos falsos.')
    return x


def load_tasks():
    tasks = []
    overview = []
    for method in METHODS:
        path = DATA_DIR / f'composers_{method}.pkl'
        meta_path = DATA_DIR / f'datos_composers_{method}.pkl'
        for needed in (path, meta_path):
            if not needed.exists():
                raise FileNotFoundError(
                    f'No se encontró {needed}. Coloca el script junto a la carpeta '
                    f'data usada por dists(1).py.'
                )
        with path.open('rb') as f:
            composers = pickle.load(f)
        with meta_path.open('rb') as f:
            metadata = pickle.load(f)

        n_total, n_long, n_broken = 0, 0, 0
        for composer in sorted(composers, key=str):
            series_dict = composers[composer]
            info = metadata.get(composer, {}) if isinstance(metadata, dict) else {}
            if not isinstance(info, dict):
                info = {}
            for name in sorted(series_dict, key=str):
                n_total += 1
                try:
                    midi = valid_midi(series_dict[name])
                except Exception as exc:
                    n_broken += 1
                    overview.append({
                        'method': method, 'composer': str(composer), 'serie': str(name),
                        'estado': 'invalida', 'motivo': str(exc), 'n_notes': np.nan
                    })
                    continue
                n = midi.size
                if n <= MIN_NOTES_EXCLUSIVE:
                    continue
                n_long += 1
                tasks.append({
                    'method': str(method),
                    'composer': str(composer),
                    'serie': str(name),
                    'birth_year': info.get('Birth_year', np.nan),
                    'composer_index': info.get('Indice', np.nan),
                    'midi': midi,
                })
                if MAX_SERIES_PER_METHOD is not None and n_long >= MAX_SERIES_PER_METHOD:
                    break
            if MAX_SERIES_PER_METHOD is not None and n_long >= MAX_SERIES_PER_METHOD:
                break
        print(f'[{method}] series: {n_total}; N>{MIN_NOTES_EXCLUSIVE}: {n_long}; invalidas: {n_broken}', flush=True)
    return tasks, overview


# ================================================================
# AJUSTES LOG-LOG / SEMILOG
# ================================================================
def linear_fit(x, y, min_points=4):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    out = dict(slope=np.nan, intercept=np.nan, r2=np.nan,
               rmse=np.nan, n=int(x.size))
    if x.size < min_points or np.unique(x).size < 2:
        return out
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope*x)
    sst = np.sum((y-y.mean())**2)
    out.update(slope=float(slope), intercept=float(intercept),
               r2=float(1 - np.sum(residuals**2)/sst) if sst>0 else np.nan,
               rmse=float(np.sqrt(np.mean(residuals**2))))
    return out


def fit_cd(C):
    C = np.asarray(C, dtype=float).ravel()
    d = np.arange(C.size)
    mask = (d >= FIT_D_MIN) & (d <= FIT_D_MAX) & np.isfinite(C) & (C>0)
    fit = linear_fit(d[mask], np.log(C[mask]), MIN_C_FIT_POINTS)
    return dict(lam=-fit['slope'] if np.isfinite(fit['slope']) else np.nan,
                r2=fit['r2'], rmse_log=fit['rmse'], n_fit=fit['n'],
                fit_full=bool(mask[FIT_D_MIN:FIT_D_MAX+1].all()),
                A=float(np.exp(fit['intercept'])) if np.isfinite(fit['intercept']) else np.nan)


# ================================================================
# Cd SOBRE INTERVALOS CON SIGNO
# ================================================================
def compute_cd(x_intervals):
    C = ml.correlation_integrals_rank_ties(
        np.asarray(x_intervals, dtype=np.float64),
        max_d=MAX_D, K=float(K_FIXED), usar_rangos=True,
    )
    C = np.asarray(C, dtype=float).ravel()
    if len(C) != MAX_D + 1:
        raise RuntimeError(f'mi_libreria devolvio {len(C)} valores Cd y esperaba {MAX_D+1}.')
    return C


def theoretical_c1_null(x):
    """Probabilidad de cercanía entre rangos de 2 muestras sin reemplazo.

    Calculada usando el vocabulario y frecuencias de TODOS los intervalos.
    En la implementación de Cd el número efectivo de inicios es N-max_d+1;
    por ello el C1 medio de shuffles puede diferir ligeramente.
    """
    _, counts = np.unique(x, return_counts=True)
    counts = counts.astype(np.int64)
    N = int(counts.sum())
    K = min(K_FIXED, len(counts)-1)
    pairs = int(np.sum(counts*(counts-1)//2))
    for h in range(1, K+1):
        pairs += int(np.sum(counts[:-h]*counts[h:]))
    return float(pairs/(N*(N-1)/2)) if N>1 else np.nan


def compute_cd_stats(intervals, rng):
    C = compute_cd(intervals)
    observed = fit_cd(C)
    lam_null = []
    C1_null = []
    n_valid = 0
    for _ in range(N_SHUFFLES):
        perm = rng.permutation(intervals)  # SE PERMUTAN LOS INTERVALOS CON SIGNO
        C_s = compute_cd(perm)
        fitted = fit_cd(C_s)
        C1_null.append(C_s[1])
        if np.isfinite(fitted['lam']) and fitted['lam'] > 0:
            lam_null.append(fitted['lam'])
            n_valid += 1
    minimum_valid = math.ceil(N_SHUFFLES*MIN_NULL_VALID_FRACTION)
    null_ok = n_valid >= minimum_valid
    null_mean = float(np.mean(lam_null)) if null_ok else np.nan
    null_std = float(np.std(lam_null, ddof=1)) if null_ok and len(lam_null)>1 else np.nan
    kappa = observed['lam']/null_mean if (
        np.isfinite(observed['lam']) and observed['lam'] > 0 and
        np.isfinite(null_mean) and null_mean > 0
    ) else np.nan
    result = {
        'lambda_obs': observed['lam'],
        'lambda_null_mean': null_mean,
        'lambda_null_std': null_std,
        'kappa': kappa,
        'R2_Cd': observed['r2'],
        'RMSE_log_Cd': observed['rmse_log'],
        'n_fit_Cd': observed['n_fit'],
        'fit_Cd_full': observed['fit_full'],
        'C1_obs': float(C[1]),
        'C1_null_exact': theoretical_c1_null(intervals),
        'C1_null_shuffle_mean': float(np.mean(C1_null)),
        'n_null_valid': n_valid,
        'n_shuffles': N_SHUFFLES,
    }
    for d, value in enumerate(C):
        result[f'C{d}_obs'] = float(value)
    return result


# ================================================================
# DFA o mDFA SOBRE MIDI
# ================================================================
def dfa_fluctuations(x, sizes, order=2, reverse_segments=True):
    """DFA: perfil integrado y detrending polinomial por ventana.

    reverse_segments=True conserva exactamente la implementación anterior
    (ventanas desde ambos extremos).

    reverse_segments=False reproduce el uso de fathon con revSeg=False
    del programa indice_no_linealidad_mdfa(2).py para mDFA.
    """
    x = np.asarray(x, dtype=float)
    profile = np.cumsum(x - x.mean())
    N = len(profile)
    F = np.full(len(sizes), np.nan, dtype=float)
    for i, s in enumerate(sizes):
        s = int(s)
        n_segments = N // s
        if n_segments < 2 or s <= order + 2:
            continue
        seg_forward = profile[:n_segments*s].reshape(n_segments, s)
        if reverse_segments:
            seg_backward = profile[N-n_segments*s:].reshape(n_segments, s)
            segments = np.concatenate((seg_forward, seg_backward), axis=0)
        else:
            segments = seg_forward
        grid = np.linspace(-1.0, 1.0, s)
        design = np.polynomial.legendre.legvander(grid, order)
        Q, _ = np.linalg.qr(design, mode='reduced')
        residual = segments - (segments @ Q) @ Q.T
        F[i] = np.sqrt(np.mean(residual**2))
    return F


def mdfa_fluctuations(x, sizes, order=2):
    """Magnitud-DFA de tu código: curva corregida F(s)/s.

    1. Delta MIDI, 2. |Delta MIDI|, 3. restar la media,
    4. integrar magnitudes, 5. aplicar DFA estándar (segunda integración),
    6. dividir la fluctuación por s.
    """
    x = np.asarray(x, dtype=float)
    magnitude = np.abs(np.diff(x))
    magnitude = magnitude - np.mean(magnitude)
    integrated_magnitude = np.cumsum(magnitude)
    F = dfa_fluctuations(
        integrated_magnitude, sizes, order=order, reverse_segments=False,
    )
    return F / np.asarray(sizes, dtype=float)


def dfa_sizes(n_effective):
    """Elección de ventanas: DFA anterior o mDFA del código de referencia."""
    s_max = n_effective // DFA_MAX_DIVISOR
    if DFA_METHOD == 'mdfa':
        s_max = min(s_max, MDFA_S_MAX_CAP)
    if s_max <= DFA_S_MIN:
        raise ValueError(f'Rango de escalas inválido: {DFA_S_MIN=}, {s_max=}.')
    if DFA_METHOD == 'mdfa' and MDFA_REFERENCE_GRID:
        # Malla lineal de paso 1 (equivalente a linRangeByStep(...)).
        return np.arange(DFA_S_MIN, s_max + 1, dtype=int)
    return np.unique(np.geomspace(DFA_S_MIN, s_max, DFA_N_SCALES).astype(int))


def compute_dfa(x):
    """Calcula alpha por DFA clásico o por magnitud-DFA sobre MIDI."""
    x = np.asarray(x, dtype=float)
    n_effective = len(x) if DFA_METHOD == 'dfa' else len(x) - 1
    sizes = dfa_sizes(n_effective)
    if DFA_METHOD == 'dfa':
        F = dfa_fluctuations(x, sizes, order=DFA_ORDER)
    elif DFA_METHOD == 'mdfa':
        F = mdfa_fluctuations(x, sizes, order=DFA_ORDER)
    else:
        raise ValueError('DFA_METHOD debe ser "dfa" o "mdfa".')

    m = np.isfinite(F) & (F > 0)
    fit = linear_fit(np.log10(sizes[m]), np.log10(F[m]), min_points=8)
    sensitivity = m & (sizes <= n_effective // DFA_SENSITIVITY_DIVISOR)
    fit_short = linear_fit(np.log10(sizes[sensitivity]),
                           np.log10(F[sensitivity]), min_points=8)
    return dict(
        metodo_DFA=DFA_METHOD,
        alpha=fit['slope'],
        R2_DFA=fit['r2'], RMSE_DFA=fit['rmse'], n_fit_DFA=fit['n'],
        alpha_N8=fit_short['slope'],
        delta_alpha_N8=fit['slope']-fit_short['slope'],
        s_min_DFA=int(sizes[m].min()) if np.any(m) else np.nan,
        s_max_DFA=int(sizes[m].max()) if np.any(m) else np.nan,
    )


# ================================================================
# PSD WELCH SOBRE MIDI
# ================================================================
def compute_psd(x):
    N = len(x)
    nperseg = min(N, PSD_MAX_NPERSEG, max(PSD_MIN_NPERSEG, N//2))
    freqs, power = welch(x, fs=1.0, window='hann',
                         nperseg=nperseg, noverlap=nperseg//2,
                         detrend='linear', scaling='density')
    fmin = 1.0 / (N//DFA_MAX_DIVISOR)
    fmin_short = 1.0 / (N//DFA_SENSITIVITY_DIVISOR)
    positive = (freqs>0) & np.isfinite(power) & (power>0) & (freqs<=PSD_F_MAX)
    mask = positive & (freqs>=fmin)
    mask_short = positive & (freqs>=fmin_short)
    fit = linear_fit(np.log10(freqs[mask]), np.log10(power[mask]),
                     min_points=PSD_MIN_FIT_POINTS)
    fit_short = linear_fit(np.log10(freqs[mask_short]), np.log10(power[mask_short]),
                           min_points=PSD_MIN_FIT_POINTS)
    beta = -fit['slope'] if np.isfinite(fit['slope']) else np.nan
    beta_short = -fit_short['slope'] if np.isfinite(fit_short['slope']) else np.nan
    return dict(beta=beta, R2_PSD=fit['r2'], RMSE_PSD=fit['rmse'],
                n_fit_PSD=fit['n'], beta_N8=beta_short,
                delta_beta_N8=beta-beta_short,
                f_min_PSD=float(freqs[mask].min()) if np.any(mask) else np.nan,
                f_max_PSD=float(freqs[mask].max()) if np.any(mask) else np.nan,
                nperseg_PSD=int(nperseg))


# ================================================================
# TRABAJO POR SERIE
# ================================================================
def process_task(task):
    method, composer, name = task['method'], task['composer'], task['serie']
    seed = stable_seed(method, composer, name)
    row = {k: v for k, v in task.items() if k != 'midi'}
    row['K'] = K_FIXED
    row['rng_seed'] = seed
    try:
        midi = valid_midi(task['midi'])
        intervals = np.diff(midi)  # ¡NO tomar valor absoluto!
        row.update({
            'N_notas': int(len(midi)), 'N_intervalos': int(len(intervals)),
            'n_unique_MIDI': int(np.unique(midi).size),
            'n_unique_intervalos': int(np.unique(intervals).size),
            'std_MIDI': float(np.std(midi)),
            'std_intervalos': float(np.std(intervals)),
        })
        if row['std_MIDI'] <= 1e-12:
            raise ValueError('Serie MIDI constante; exponentes sin definir.')
        if row['n_unique_intervalos'] <= K_FIXED+1:
            raise ValueError('Muy pocos intervalos distintos para K fijo: C1=1.')
        rng = np.random.default_rng(seed)
        row.update(compute_cd_stats(intervals, rng))
        row.update(compute_dfa(midi))
        row.update(compute_psd(midi))
        # beta≈2*alpha-1 solo es pertinente para DFA de la serie MIDI
        # en condiciones apropiadas. No se aplica al mDFA de |Delta MIDI|.
        row['delta_beta_alpha'] = (
            row['beta'] - (2*row['alpha'] - 1)
            if DFA_METHOD == 'dfa' else np.nan
        )
        row['estado'] = 'ok'
        row['error'] = ''
    except Exception as exc:
        row['estado'] = 'error'
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


# ================================================================
# CORRELACIONES Y CONTROLES
# ================================================================
PAIRS = [('lambda_obs', 'alpha'), ('lambda_obs', 'beta'),
         ('kappa', 'alpha'), ('kappa', 'beta')]


def spearman_safe(x, y):
    if len(x) < MIN_GROUP_SIZE or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def bootstrap_composers(df, xcol, ycol, seed):
    if BOOTSTRAP_BY_COMPOSER <= 0 or df['composer'].nunique() < 3:
        return np.nan, np.nan
    indices = [part.index.to_numpy() for _, part in df.groupby('composer', sort=True)]
    rng = np.random.default_rng(seed)
    coeffs = []
    for _ in range(BOOTSTRAP_BY_COMPOSER):
        drawn = rng.integers(0, len(indices), size=len(indices))
        selected = np.concatenate([indices[i] for i in drawn])
        sub = df.loc[selected]
        coef, _ = spearman_safe(sub[xcol].to_numpy(), sub[ycol].to_numpy())
        if np.isfinite(coef):
            coeffs.append(coef)
    if len(coeffs) < 20:
        return np.nan, np.nan
    return tuple(map(float, np.percentile(coeffs, [2.5, 97.5])))


def partial_spearman(df, xcol, ycol, control_method):
    """Spearman parcial residualizando rangos de N y vocabulario.

    Control por método solo en análisis conjunto. P-valores parciales
    son orientativos; la dependencia por compositor requiere cautela.
    """
    n = len(df)
    if n < MIN_GROUP_SIZE+3:
        return np.nan, np.nan
    sx = rankdata(df[xcol].to_numpy(dtype=float))
    sy = rankdata(df[ycol].to_numpy(dtype=float))
    col_list = [rankdata(np.log(df['N_notas'].to_numpy(dtype=float))),
                rankdata(df['n_unique_intervalos'].to_numpy(dtype=float))]
    if control_method and df['method'].nunique() > 1:
        dummies = pd.get_dummies(df['method'], drop_first=True).to_numpy(dtype=float)
        col_list.extend([dummies[:, j] for j in range(dummies.shape[1])])
    Z = np.column_stack([np.ones(n)] + col_list)
    rx = sx - Z @ np.linalg.lstsq(Z, sx, rcond=None)[0]
    ry = sy - Z @ np.linalg.lstsq(Z, sy, rcond=None)[0]
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    n_covariates = np.linalg.matrix_rank(Z)-1
    dof = n-n_covariates-2
    if dof <= 0:
        return r, np.nan
    p = float(2*t_dist.sf(abs(r)*np.sqrt(dof/max(1-r*r, 1e-14)), dof))
    return r, p


def correlations(df):
    all_rows = []
    groups = [('todos', df)] + [(m, df[df.method == m]) for m in METHODS]
    for group_name, group in groups:
        for xcol, ycol in PAIRS:
            data = group[['method','composer',xcol,ycol,'N_notas',
                          'n_unique_intervalos']].replace([np.inf,-np.inf],np.nan).dropna().copy()
            rho, p = spearman_safe(data[xcol].to_numpy(), data[ycol].to_numpy())
            rho_control, p_control = partial_spearman(
                data, xcol, ycol, control_method=(group_name=='todos'))
            ci1, ci2 = bootstrap_composers(data, xcol, ycol,
                                         seed=stable_seed(group_name,xcol,ycol))
            all_rows.append({
                'grupo': group_name, 'variable_Cd': xcol, 'exponente': ycol,
                'n': len(data), 'n_compositores': data.composer.nunique(),
                'rho_spearman': rho, 'p_spearman': p,
                'rho_parcial_N_vocab': rho_control,
                'p_parcial_N_vocab': p_control,
                'CI95_compositor_inf': ci1, 'CI95_compositor_sup': ci2,
            })
    return pd.DataFrame(all_rows)


# ================================================================
# GRÁFICAS
# ================================================================
def draw_correlations(df, name):
    fig, axs = plt.subplots(2,2,figsize=(11.5,8.2), constrained_layout=True)
    for ax,(xcol,ycol) in zip(axs.ravel(), PAIRS):
        part = df[[xcol,ycol,'N_notas']].replace([np.inf,-np.inf],np.nan).dropna()
        if not part.empty:
            points = ax.scatter(part[xcol], part[ycol], c=np.log10(part.N_notas),
                                s=19, alpha=0.7, cmap='viridis', edgecolor='none')
            r,p = spearman_safe(part[xcol].to_numpy(),part[ycol].to_numpy())
            ax.text(0.025,0.975,f'Spearman = {r:.3f}\np = {p:.2g}\nn = {len(part)}',
                    transform=ax.transAxes,ha='left',va='top',fontsize=9,
                    bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none'})
        ax.set_xlabel(xcol.replace('_',' '))
        ax.set_ylabel(f'alpha ({DFA_METHOD.upper()})' if ycol=='alpha' else ycol)
        ax.grid(alpha=0.2)
    fig.suptitle(f'{name}: Cd(intervalos con signo) vs {DFA_METHOD.upper()}/PSD(MIDI), K={K_FIXED}')
    fig.savefig(OUTPUT_DIR/f'correlaciones_{name}.png',dpi=DPI)
    plt.close(fig)


def draw_confounders(df, name):
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for row_idx,metric in enumerate(('lambda_obs','kappa')):
        for col_idx,(col,label) in enumerate((('N_notas','N notas'),
                                               ('n_unique_intervalos','Vocabulario intervalos'))):
            ax=axs[row_idx,col_idx]
            part=df[[col,metric]].replace([np.inf,-np.inf],np.nan).dropna()
            ax.scatter(part[col],part[metric],s=16,alpha=0.65)
            r,p=spearman_safe(part[col].to_numpy(),part[metric].to_numpy())
            ax.set(xlabel=label,ylabel=metric,title=f'Spearman={r:.3f}, p={p:.2g}')
            if col=='N_notas': ax.set_xscale('log')
            ax.grid(alpha=0.2)
    fig.suptitle(f'Controles potenciales: {name}, K={K_FIXED}')
    fig.savefig(OUTPUT_DIR/f'controles_{name}.png',dpi=DPI)
    plt.close(fig)


def draw_length_distribution(df):
    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    values=df['N_notas'].dropna().to_numpy(dtype=float)
    if values.size:
        bins=np.geomspace(max(MIN_NOTES_EXCLUSIVE,1),max(values.max()+1,MIN_NOTES_EXCLUSIVE+10),24)
        for method in METHODS:
            part=df.loc[df.method==method,'N_notas'].dropna()
            if len(part):
                ax.hist(part,bins=bins,histtype='step',lw=2,label=f'{method} (n={len(part)})')
        ax.set_xscale('log')
        ax.legend()
    ax.set(xlabel='Número de notas MIDI',ylabel='Número de melodías',
           title=f'Distribución de longitudes: N>{MIN_NOTES_EXCLUSIVE}')
    ax.grid(alpha=.2)
    fig.savefig(OUTPUT_DIR/'histograma_longitudes.png',dpi=DPI)
    plt.close(fig)


def draw_alpha_beta_diagnostics(df):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in METHODS:
        part = df[df.method == method]
        axs[0].scatter(part['alpha'], part['beta'], s=16, alpha=.6, label=method)
        if DFA_METHOD == 'dfa':
            axs[1].scatter(part['alpha'], part['delta_beta_alpha'],
                           s=16, alpha=.6, label=method)
        else:
            axs[1].scatter(part['alpha'], part['R2_DFA'],
                           s=16, alpha=.6, label=method)

    if DFA_METHOD == 'dfa':
        finite = df['alpha'][np.isfinite(df['alpha'])]
        if len(finite):
            a = np.linspace(finite.min(), finite.max(), 100)
            axs[0].plot(a, 2*a-1, 'k--', lw=1,
                        label=r'$\beta=2\alpha-1$ (referencia)')
        axs[1].axhline(0, c='k', ls='--', lw=1)
        axs[1].set(xlabel=r'$\alpha$ DFA', ylabel=r'$\beta-(2\alpha-1)$')
    else:
        # mDFA opera sobre la magnitud de los incrementos; beta sigue
        # siendo la pendiente espectral de las alturas MIDI.
        axs[1].set(xlabel=r'$\alpha$ mDFA',
                   ylabel=r'$R^2$ del ajuste mDFA')
    axs[0].set(xlabel=rf'$\alpha$ {DFA_METHOD.upper()}', ylabel=r'$\beta$ PSD(MIDI)')
    for ax in axs:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT_DIR/'diagnostico_alpha_beta.png', dpi=DPI)
    plt.close(fig)


# ================================================================
# GUARDADO PROGRESIVO Y EJECUCIÓN
# ================================================================
def append_row(path, row):
    file_exists=path.exists() and path.stat().st_size>0
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=OUTPUT_COLUMNS,extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow({k:row.get(k,'') for k in OUTPUT_COLUMNS})

OUTPUT_COLUMNS = (
    ['method','composer','serie','birth_year','composer_index','N_notas','N_intervalos',
     'n_unique_MIDI','n_unique_intervalos','std_MIDI','std_intervalos','K','rng_seed',
     'lambda_obs','lambda_null_mean','lambda_null_std','kappa',
     'R2_Cd','RMSE_log_Cd','n_fit_Cd','fit_Cd_full',
     'C1_obs','C1_null_exact','C1_null_shuffle_mean','n_null_valid','n_shuffles']
    + [f'C{d}_obs' for d in range(MAX_D+1)]
    + ['metodo_DFA','alpha','R2_DFA','RMSE_DFA','n_fit_DFA','alpha_N8','delta_alpha_N8',
       's_min_DFA','s_max_DFA','beta','R2_PSD','RMSE_PSD','n_fit_PSD',
       'beta_N8','delta_beta_N8','f_min_PSD','f_max_PSD',
       'nperseg_PSD','delta_beta_alpha','estado','error']
)


def ensure_config():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    config_path=OUTPUT_DIR/'configuracion.json'
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding='utf-8'))
        if old != CONFIG_KEYS:
            raise RuntimeError(
                f'La configuración cambió desde la ejecución anterior.\n'
                f'Para evitar mezclar resultados, mueve/renombra {OUTPUT_DIR} '
                f'o recupera la configuración original.\n'
                f'Anterior: {old}\nNueva: {CONFIG_KEYS}'
            )
    else:
        config_path.write_text(json.dumps(CONFIG_KEYS,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    if DFA_METHOD not in ('dfa', 'mdfa'):
        raise ValueError('DFA_METHOD debe ser "dfa" o "mdfa".')
    if not isinstance(MDFA_S_MAX_CAP, int) or MDFA_S_MAX_CAP <= DFA_S_MIN:
        raise ValueError('MDFA_S_MAX_CAP debe ser un entero mayor que DFA_S_MIN.')
    if not isinstance(N_SHUFFLES,int) or N_SHUFFLES<1:
        raise ValueError('N_SHUFFLES debe ser entero positivo')
    if FIT_D_MAX>MAX_D or FIT_D_MIN<1:
        raise ValueError('Revisar FIT_D_MIN/FIT_D_MAX/MAX_D')
    ensure_config()
    tasks, invalid = load_tasks()
    data_path=OUTPUT_DIR/'resultados_por_melodia.csv'
    if data_path.exists():
        previous=pd.read_csv(data_path)
        if not {'method','composer','serie'}.issubset(previous.columns):
            raise RuntimeError('El CSV previo no tiene las claves de identificación esperadas.')
        completed=set(zip(previous.method.astype(str),previous.composer.astype(str),
                          previous.serie.astype(str)))
    else:
        completed=set()
    pending=[t for t in tasks if (t['method'],t['composer'],t['serie']) not in completed]
    print(f'Total elegibles={len(tasks)}, ya calculadas={len(tasks)-len(pending)}, pendientes={len(pending)}',flush=True)
    for item in pending[:3]:
        print(f"  ejemplo: {item['method']}/{item['composer']}/{item['serie']} N={len(item['midi'])}")
    if N_WORKERS == 1:
        iterator = map(process_task,pending)
        for i,row in enumerate(iterator,1):
            append_row(data_path,row)
            if row['estado']!='ok':
                print(f"  ERROR {row['method']}/{row['composer']}/{row['serie']}: {row['error']}",flush=True)
            if i==1 or i%10==0:
                print(f'Procesadas {i}/{len(pending)} nuevas melodías',flush=True)
    else:
        # Windows: bajo __main__ se evita el error de spawn recursivo.
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for i,row in enumerate(pool.map(process_task,pending,chunksize=1),1):
                append_row(data_path,row)
                if row['estado']!='ok':
                    print(f"  ERROR {row['method']}/{row['composer']}/{row['serie']}: {row['error']}",flush=True)
                if i==1 or i%10==0:
                    print(f'Procesadas {i}/{len(pending)} nuevas melodías',flush=True)

    if not data_path.exists():
        print('No hay series elegibles (N>1000).')
        return
    all_df=pd.read_csv(data_path)
    all_df=all_df.drop_duplicates(subset=['method','composer','serie'],keep='last')
    all_df.to_csv(data_path,index=False,encoding='utf-8')
    errs=all_df[all_df.estado!='ok'].copy()
    errs.to_csv(OUTPUT_DIR/'errores.csv',index=False)
    pd.DataFrame(invalid).to_csv(OUTPUT_DIR/'series_invalidas_en_entrada.csv',index=False)
    df=all_df[all_df.estado=='ok'].copy()
    if df.empty:
        print('Ninguna melodía pudo calcularse. Consultar errores.csv.')
        return
    for col in ('N_notas','n_unique_intervalos','lambda_obs','kappa','alpha','beta'):
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df.groupby(['method','composer'],as_index=False).agg(
        n_melodias=('serie','size'),mediana_N=('N_notas','median'),
        mediana_lambda=('lambda_obs','median'),mediana_kappa=('kappa','median'),
        mediana_alpha=('alpha','median'),mediana_beta=('beta','median')
    ).to_csv(OUTPUT_DIR/'resumen_por_compositor.csv',index=False)
    corr=correlations(df)
    corr.to_csv(OUTPUT_DIR/'correlaciones.csv',index=False)
    print('\nCORRELACIONES DE INTERÉS:')
    print(corr.to_string(index=False,float_format=lambda x:f'{x:.5g}'))
    print('\nN de melodías por método (OK):')
    print(df.groupby('method').size().to_string())
    if SAVE_PLOTS:
        for method in METHODS:
            part=df[df.method==method]
            if len(part):
                draw_correlations(part,method)
                draw_confounders(part,method)
        draw_length_distribution(df)
        draw_alpha_beta_diagnostics(df)
    print(f'\nArchivos guardados en: {OUTPUT_DIR.resolve()}')
    print('Archivo principal: resultados_por_melodia.csv')
    print('Tabla estadística: correlaciones.csv')
    print(f'Método usado para alpha: {DFA_METHOD.upper()}')


if __name__=='__main__':
    main()
