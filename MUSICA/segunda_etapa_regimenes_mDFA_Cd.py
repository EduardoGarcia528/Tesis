# -*- coding: utf-8 -*-
"""
SEGUNDA ETAPA: ¿LAS MELODIAS CON mDFA MONOESCALA CORRELACIONAN MEJOR CON Cd?

Se ejecuta DESPUES de comparar_Cd_DFA_mDFA_PSD_corpus_largo(1).py.
No recalcula Cd, los shuffles ni la PSD: reutiliza lambda_obs y kappa
almacenados en resultados_por_melodia.csv y reconstruye F(s)/s desde los
arrays MIDI originales de data/composers_{hybrid,complexity}.pkl.

Reproduce el mDFA original (revSeg=False, orden 2, magnitudes de incrementos
centradas e integradas y correccion F(s)/s). Ajusta 1, 2, 3 regímenes
CONTINUOS en escala log-log, con 0, 1, 2 puntos de quiebre, respectivamente.
Usa BIC como diagnostico exploratorio sobre los puntos LOG-AGRUPADOS, no
como prueba estadística formal: F(s) de escalas vecinas esta correlacionada.

No hay selección por lambda_obs ni por kappa; el filtro depende solo de F(s).
Incluye correlaciones globales, por corpus, bootstrap pareado por compositor,
controles por N y vocabulario, auditoría contra alpha original, gráficos F(s),
atlas, y controles de sesgo de selección.

Instalacion: pip install numpy pandas scipy matplotlib
Organizacion: coloque este archivo junto al script original y la carpeta data/.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

# ====================== CONFIGURACION ===============================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
ORIGINAL_DIR = ROOT / 'resultados_Cd_MDFA_PSD_corpus_largo_v2'
CSV_ORIGINAL = ORIGINAL_DIR / 'resultados_por_melodia.csv'
DEST = ORIGINAL_DIR / 'segunda_etapa_regimenes_mDFA'

# Igual que el experimento original. Si modificaste el original,
# ajusta estos parámetros y comprueba la auditoría alpha_refit_vs_original.
DFA_ORDER = 2
S_MIN = 20
S_MAX_CAP = 500
S_MAX_DIVISOR = 4
MALLA_REFERENCIA = True
N_SCALES_GEOMETRIC = 32

# Las escalas son muy densas en el extremo grande (s=20,21,...,500).
# Se ajusta sobre medianas por bin en log(s) para no sobrerrepresentarlas.
BINS_PRINCIPALES = 24
BINS_SENSIBILIDAD = (18, 24, 30)
MIN_BINS = 14
MIN_SPAN_DECADAS = 0.95
MIN_PUNTOS_SEGMENTO = 5
MIN_ANCHO_SEGMENTO_DECADAS = 0.24

# Criterios PREDEFINIDOS, independientes de lambda/kappa.
R2_MIN = 0.970                  # ajuste único en los puntos log-agrupados
MAX_DELTA_ALFA_MEDIAS = 0.20  # pendientes por mitades de log(s)
GANANCIA_BIC_MULTIESCALA = 6  # BIC_1 - min(BIC_2,BIC_3)
MIN_CAMBIO_PENDIENTE = 0.15    # magnitud de cambio de pendiente entre regímenes
CONSENSO_MINIMO = 2            # entre tres granularidades de agrupación
RECHAZAR_SI_CUALQUIER_MUL = True

# Diagnósticos, gráficas y bootstrap.
BOOTSTRAP_REPS = 1000
PERMUTACIONES_IGUAL_N = 1500   # controles al azar, por método y compositor
RANDOM_SEED = 20261116
N_MIN_CORRELACION = 5
GENERAR_ATLAS = True             # todas las curvas, paginadas en PNG
CURVAS_POR_PAGINA = 12
GENERAR_PNG_INDIVIDUALES_ACEPTADAS = True
GENERAR_PNG_RECHAZADAS_EJEMPLO = 36
DPI = 145

# Para probar con pocas melodías: entero, luego volver a None.
MAX_MELODIAS = None

ANALYSIS_CONFIG = {
    'DFA_ORDER': DFA_ORDER, 'S_MIN': S_MIN, 'S_MAX_CAP': S_MAX_CAP,
    'S_MAX_DIVISOR': S_MAX_DIVISOR, 'MALLA_REFERENCIA': MALLA_REFERENCIA,
    'N_SCALES_GEOMETRIC': N_SCALES_GEOMETRIC,
    'BINS_PRINCIPALES': BINS_PRINCIPALES,
    'BINS_SENSIBILIDAD': list(BINS_SENSIBILIDAD),
    'MIN_BINS': MIN_BINS, 'MIN_SPAN_DECADAS': MIN_SPAN_DECADAS,
    'MIN_PUNTOS_SEGMENTO': MIN_PUNTOS_SEGMENTO,
    'MIN_ANCHO_SEGMENTO_DECADAS': MIN_ANCHO_SEGMENTO_DECADAS,
    'R2_MIN': R2_MIN,
    'MAX_DELTA_ALFA_MEDIAS': MAX_DELTA_ALFA_MEDIAS,
    'GANANCIA_BIC_MULTIESCALA': GANANCIA_BIC_MULTIESCALA,
    'MIN_CAMBIO_PENDIENTE': MIN_CAMBIO_PENDIENTE,
    'CONSENSO_MINIMO': CONSENSO_MINIMO,
    'RECHAZAR_SI_CUALQUIER_MUL': RECHAZAR_SI_CUALQUIER_MUL,
}


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def is_finite(v):
    return bool(np.isfinite(safe_float(v)))


def key_hash(method, composer, serie):
    value = json.dumps([str(method), str(composer), str(serie)], ensure_ascii=False)
    return hashlib.blake2b(value.encode('utf-8'), digest_size=10).hexdigest()


def file_label(method, composer, serie):
    def slug(a):
        a = re.sub(r'[^\w-]+', '_', str(a), flags=re.UNICODE)
        return a.strip('_')[:32] or 'serie'
    return f'{slug(method)}_{slug(composer)}_{slug(serie)}_{key_hash(method, composer, serie)}'


def validate_paths_and_config():
    if not CSV_ORIGINAL.exists():
        raise FileNotFoundError(
            f'No existe {CSV_ORIGINAL}. Ejecuta primero el script original '
            f'o cambia ORIGINAL_DIR para apuntar a sus resultados.')
    parent_config = ORIGINAL_DIR / 'configuracion.json'
    if parent_config.exists():
        conf = json.loads(parent_config.read_text(encoding='utf-8'))
        expected = {'DFA_METHOD': 'mdfa', 'MDFA_REFERENCE_GRID': MALLA_REFERENCIA,
                    'MDFA_S_MAX_CAP': S_MAX_CAP, 'DFA_ORDER': DFA_ORDER,
                    'DFA_S_MIN': S_MIN, 'DFA_MAX_DIVISOR': S_MAX_DIVISOR}
        for k, value in expected.items():
            if k in conf and conf[k] != value:
                raise ValueError(f'La configuración original {k}={conf[k]!r} '
                                 f'no coincide con esta segunda etapa ({value!r}).')
    DEST.mkdir(parents=True, exist_ok=True)
    conf_path = DEST / 'configuracion_segunda_etapa.json'
    if conf_path.exists():
        old = json.loads(conf_path.read_text(encoding='utf-8'))
        if old != ANALYSIS_CONFIG:
            raise RuntimeError(
                'Cambiaste parámetros respecto a la ejecución anterior. '
                f'Renombra/elimina el directorio {DEST} para no mezclar resultados.')
    else:
        conf_path.write_text(json.dumps(ANALYSIS_CONFIG, ensure_ascii=False, indent=2),
                             encoding='utf-8')


# ==================== REPRODUCCION EXACTA mDFA =======================

def get_sizes(n_midi):
    n = n_midi - 1
    s_max = min(S_MAX_CAP, n // S_MAX_DIVISOR)
    if s_max <= S_MIN:
        raise ValueError(f'Longitud insuficiente: N={n_midi}')
    if MALLA_REFERENCIA:
        return np.arange(S_MIN, s_max + 1, dtype=int)
    return np.unique(np.geomspace(S_MIN, s_max, N_SCALES_GEOMETRIC).astype(int))


def dfa_fluctuations(x, sizes, order=2):
    """Misma implementación del original, exclusivamente revSeg=False."""
    x = np.asarray(x, dtype=float)
    profile = np.cumsum(x - x.mean())
    n = len(profile)
    fluct = np.full(len(sizes), np.nan, dtype=float)
    for i, s in enumerate(sizes):
        s = int(s)
        n_segments = n // s
        if n_segments < 2 or s <= order + 2:
            continue
        segments = profile[:n_segments*s].reshape(n_segments, s)
        grid = np.linspace(-1.0, 1.0, s)
        design = np.polynomial.legendre.legvander(grid, order)
        q, _ = np.linalg.qr(design, mode='reduced')
        residual = segments - (segments @ q) @ q.T
        fluct[i] = np.sqrt(np.mean(residual**2))
    return fluct


def mdfa_curve(midi):
    """Reproduce mdfa_fluctuations del script del usuario, incluido F(s)/s."""
    midi = np.asarray(midi, dtype=float)
    if midi.ndim != 1 or not np.isfinite(midi).all():
        raise ValueError('MIDI debe ser 1D y finito; no se eliminan notas.')
    sizes = get_sizes(len(midi))
    magnitude = np.abs(np.diff(midi))
    magnitude = magnitude - magnitude.mean()
    integrated_magnitude = np.cumsum(magnitude)
    fluct = dfa_fluctuations(integrated_magnitude, sizes, order=DFA_ORDER)
    fluct = fluct / sizes
    ok = (fluct > 0) & np.isfinite(fluct)
    if ok.sum() < 15:
        raise ValueError('Insuficientes fluctuaciones F(s)>0.')
    return sizes[ok], fluct[ok]


def line_fit(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) <= 0:
        return dict(alpha=np.nan, intercept=np.nan, r2=np.nan, rmse=np.nan)
    a, b = np.polyfit(x, y, 1)
    pred = a*x + b
    sse = float(np.sum((y-pred)**2))
    sst = float(np.sum((y-y.mean())**2))
    return dict(alpha=float(a), intercept=float(b),
                r2=float(1-sse/sst) if sst > 0 else np.nan,
                rmse=float(np.sqrt(sse/len(x))))


def bin_log_curves(sizes, fluct, n_bins):
    """Medianas en celdas uniformemente espaciadas en log10(s)."""
    lx = np.log10(np.asarray(sizes, dtype=float))
    ly = np.log10(np.asarray(fluct, dtype=float))
    edges = np.linspace(lx.min(), lx.max(), n_bins+1)
    membership = np.clip(np.searchsorted(edges, lx, side='right')-1,
                         0, n_bins-1)
    bx, by, counts = [], [], []
    for i in range(n_bins):
        mask = membership == i
        if mask.any():
            bx.append(np.median(lx[mask]))
            by.append(np.median(ly[mask]))
            counts.append(int(mask.sum()))
    return np.asarray(bx), np.asarray(by), np.asarray(counts)


# ============== REGRESIONES SEGMENTADAS CONTINUAS ====================

def fit_hinges(x, y, breaks):
    """y = a + b*x + sum_j delta_b_j max(0,x-k_j)."""
    n = len(x)
    knots = tuple(x[i] for i in breaks)
    X = np.column_stack([np.ones(n), x] + [np.maximum(0, x-t) for t in knots])
    coefs = np.linalg.lstsq(X, y, rcond=None)[0]
    prediction = X @ coefs
    resid = y - prediction
    sse = float(resid @ resid)
    # Penalizamos también las posiciones estimadas de los cortes:
    # 1 segmento k=2; 2 segmentos k=4; 3 segmentos k=6.
    k = 2 + 2*len(breaks)
    # Evitar evidencia espuria por errores de redondeo cuando la curva es
    # (casi) una recta perfecta: RSS inferior al ruido numérico no distingue.
    numeric_floor = (max(np.ptp(y), 1.0)*1e-12)**2
    bic = n*np.log(max(sse/n, numeric_floor)) + k*np.log(n)
    slopes = np.cumsum(coefs[1:2+len(breaks)]).tolist()
    return dict(n_reg=len(breaks)+1, breaks=tuple(breaks), knots=knots,
                coeffs=coefs, pred=prediction, sse=sse, bic=float(bic),
                slopes=[float(t) for t in slopes])


def valid_breaks(x, breaks):
    # Cortes definen indices iniciales de los siguientes segmentos.
    limits = (0,) + tuple(breaks) + (len(x),)
    for start, end in zip(limits[:-1], limits[1:]):
        if end-start < MIN_PUNTOS_SEGMENTO:
            return False
        if (x[end-1] - x[start]) < MIN_ANCHO_SEGMENTO_DECADAS:
            return False
    return True


def best_piecewise(x, y):
    """Minimiza BIC para 1, 2 y 3 regímenes; cortes por búsqueda exhaustiva."""
    n = len(x)
    fits = {1: fit_hinges(x, y, ())}
    for i in range(MIN_PUNTOS_SEGMENTO, n-MIN_PUNTOS_SEGMENTO+1):
        br = (i,)
        if valid_breaks(x, br):
            new = fit_hinges(x, y, br)
            if 2 not in fits or new['bic'] < fits[2]['bic']:
                fits[2] = new
    for i in range(MIN_PUNTOS_SEGMENTO, n-2*MIN_PUNTOS_SEGMENTO+1):
        for j in range(i+MIN_PUNTOS_SEGMENTO, n-MIN_PUNTOS_SEGMENTO+1):
            br = (i, j)
            if valid_breaks(x, br):
                new = fit_hinges(x, y, br)
                if 3 not in fits or new['bic'] < fits[3]['bic']:
                    fits[3] = new
    return fits


def assess_binning(sizes, fluct, n_bins):
    x, y, counts = bin_log_curves(sizes, fluct, n_bins)
    row = dict(bins_requested=n_bins, n_bins=len(x),
               span_dec=float(x[-1]-x[0]) if len(x) else np.nan,
               r2_1=np.nan, alpha_1=np.nan,
               alpha_izquierda=np.nan, alpha_derecha=np.nan,
               delta_alpha_mitades=np.nan, bic_1=np.nan, bic_2=np.nan,
               bic_3=np.nan, ganancia_bic=np.nan, model_best=0,
               cambio_pendiente=np.nan, evidencia='indeterminado',
               calidad=False, unique_compatible=False)
    if len(x) < MIN_BINS or row['span_dec'] < MIN_SPAN_DECADAS:
        return row, (x, y, {}), counts
    single = line_fit(x, y)
    mid = 0.5 * (x[0] + x[-1])
    mask = x <= mid
    left = line_fit(x[mask], y[mask])
    right = line_fit(x[~mask], y[~mask])
    delta = abs(left['alpha'] - right['alpha'])
    fits = best_piecewise(x, y)
    best = min(fits.values(), key=lambda d: d['bic'])
    other_bic = min((fit['bic'] for k, fit in fits.items() if k > 1), default=np.inf)
    gain = float(fits[1]['bic'] - other_bic)
    # Si el modelo con BIC minimo es segmentado, medir cambio de pendientes.
    jumps = np.diff(best['slopes'])
    change = float(np.max(abs(jumps))) if len(jumps) else 0.0
    evidence = ('multiescala' if gain >= GANANCIA_BIC_MULTIESCALA
                and best['n_reg'] > 1 and change >= MIN_CAMBIO_PENDIENTE
                else ('ambiguo' if gain >= GANANCIA_BIC_MULTIESCALA
                      else 'sin_evidencia_fuerte_multiescala'))
    quality = bool(is_finite(single['r2']) and single['r2'] >= R2_MIN)
    compatible = bool(quality and evidence == 'sin_evidencia_fuerte_multiescala'
                      and is_finite(delta) and delta <= MAX_DELTA_ALFA_MEDIAS)
    row.update(r2_1=single['r2'], alpha_1=single['alpha'],
               alpha_izquierda=left['alpha'], alpha_derecha=right['alpha'],
               delta_alpha_mitades=delta,
               bic_1=fits[1]['bic'], bic_2=fits[2]['bic'] if 2 in fits else np.nan,
               bic_3=fits[3]['bic'] if 3 in fits else np.nan,
               ganancia_bic=gain if np.isfinite(gain) else np.nan,
               model_best=best['n_reg'], cambio_pendiente=change,
               evidencia=evidence, calidad=quality,
               unique_compatible=compatible)
    return row, (x, y, fits), counts


def diagnose_curve(sizes, fluct):
    raw_fit = line_fit(np.log10(sizes), np.log10(fluct))
    by_bins = {}
    for n in sorted(set(BINS_SENSIBILIDAD + (BINS_PRINCIPALES,))):
        stats, details, counts = assess_binning(sizes, fluct, n)
        by_bins[n] = (stats, details, counts)
    main = by_bins[BINS_PRINCIPALES][0]
    n_unique = sum(by_bins[n][0]['unique_compatible'] for n in BINS_SENSIBILIDAD)
    n_multi = sum(by_bins[n][0]['evidencia'] == 'multiescala'
                  for n in BINS_SENSIBILIDAD)
    n_unknown = sum(by_bins[n][0]['evidencia'] == 'indeterminado'
                    for n in BINS_SENSIBILIDAD)
    reject_multi = RECHAZAR_SI_CUALQUIER_MUL and n_multi > 0
    selected = bool(main['unique_compatible'] and n_unique >= CONSENSO_MINIMO
                    and not reject_multi)
    if selected:
        category = 'monoescala_compatible'
    elif main['evidencia'] == 'multiescala' or n_multi >= 2:
        category = 'multiescala'
    else:
        category = 'indeterminado'
    result = dict(alpha_refit_full=raw_fit['alpha'],
                  R2_refit_full=raw_fit['r2'],
                  alpha_logbins=main['alpha_1'],
                  R2_logbins=main['r2_1'],
                  alpha_izquierda=main['alpha_izquierda'],
                  alpha_derecha=main['alpha_derecha'],
                  delta_alpha_mitades=main['delta_alpha_mitades'],
                  BIC_1=main['bic_1'], BIC_2=main['bic_2'], BIC_3=main['bic_3'],
                  ganancia_BIC_multiescala=main['ganancia_bic'],
                  modelo_BIC_regimenes=main['model_best'],
                  cambio_pendiente_max=main['cambio_pendiente'],
                  n_bins_log=main['n_bins'],
                  span_decadas=main['span_dec'],
                  evidencia_principal=main['evidencia'],
                  n_consenso_monoescala=n_unique,
                  n_consenso_multiescala=n_multi,
                  n_indeterminados_sensibilidad=n_unknown,
                  calidad_R2=main['calidad'],
                  monoescala_seleccionada=selected,
                  categoria=category)
    for n in BINS_SENSIBILIDAD:
        result[f'evidencia_bins_{n}'] = by_bins[n][0]['evidencia']
        result[f'alpha_bins_{n}'] = by_bins[n][0]['alpha_1']
        result[f'R2_bins_{n}'] = by_bins[n][0]['r2_1']
    return result, by_bins


# =================== CARGA, CACHE Y DIAGNOSTICOS ====================

def load_csv():
    frame = pd.read_csv(CSV_ORIGINAL, dtype={'method': str,
                                              'composer': str, 'serie': str})
    columns = {'method', 'composer', 'serie', 'estado', 'lambda_obs',
               'kappa', 'alpha', 'N_notas', 'n_unique_intervalos'}
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError('Faltan columnas en el CSV base: ' + ', '.join(sorted(missing)))
    frame = frame.loc[frame['estado'].astype(str).str.lower() == 'ok'].copy()
    if 'metodo_DFA' in frame.columns:
        frame = frame[frame['metodo_DFA'].astype(str).str.lower() == 'mdfa'].copy()
    if 'K' in frame.columns and not (pd.to_numeric(frame['K'], errors='coerce') == 2).all():
        raise ValueError('Este archivo espera el experimento con K=2, sin mezclar K.')
    for col in ('lambda_obs','kappa','alpha','N_notas','n_unique_intervalos'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.drop_duplicates(['method','composer','serie'], keep='last')
    frame = frame.reset_index(drop=True)
    if MAX_MELODIAS is not None:
        frame = frame.head(MAX_MELODIAS).copy()
    if frame.empty:
        raise ValueError('No se encontraron filas mDFA de estado=ok.')
    return frame


def load_midi_corresponding(frame):
    """Carga sólo los métodos presentes; metadata no es necesaria."""
    result = {}
    for method in sorted(frame['method'].dropna().unique()):
        path = DATA_DIR / f'composers_{method}.pkl'
        if not path.exists():
            raise FileNotFoundError(f'Falta {path}. Se necesita MIDI para reconstruir F(s).')
        with path.open('rb') as h:
            result[method] = pickle.load(h)
    return result


def run_all(frame, composers):
    cache_dir = DEST / 'cache_curvas'
    cache_dir.mkdir(exist_ok=True)
    metrics_rows, errors = [], []
    for position, record in enumerate(frame.itertuples(index=False), 1):
        method, composer, serie = record.method, record.composer, record.serie
        digest = key_hash(method, composer, serie)
        path = cache_dir / (digest + '.npz')
        try:
            midi = np.asarray(composers[method][composer][serie], dtype=float)
            if midi.ndim != 1 or len(midi) != record.N_notas:
                raise ValueError('La longitud MIDI no coincide con el CSV original.')
            if path.exists():
                with np.load(path) as obj:
                    sizes, fluct = obj['sizes'], obj['fluct']
            else:
                sizes, fluct = mdfa_curve(midi)
                np.savez_compressed(path, sizes=sizes, fluct=fluct)
            diagnose, _ = diagnose_curve(sizes, fluct)
            diagnose.update(method=method, composer=composer, serie=serie,
                            id_curva=digest, estado_revision='ok')
            diagnose['delta_alpha_vs_original'] = (diagnose['alpha_refit_full']
                                                    - safe_float(record.alpha))
            metrics_rows.append(diagnose)
        except Exception as exc:
            errors.append(dict(method=method, composer=composer, serie=serie,
                               error=f'{type(exc).__name__}: {exc}'))
        if position == 1 or position % 25 == 0 or position == len(frame):
            print(f'Analizadas {position}/{len(frame)} | errores={len(errors)}', flush=True)
        # Guardado progresivo por si se interrumpe el cálculo.
        if position % 25 == 0 or position == len(frame):
            pd.DataFrame(metrics_rows).to_csv(DEST/'diagnosticos_parciales.csv', index=False)
            pd.DataFrame(errors, columns=['method','composer','serie','error']).to_csv(
                DEST/'errores_revision.csv', index=False)
    if not metrics_rows:
        raise RuntimeError('No fue posible reconstruir ninguna curva; revisar errores_revision.csv.')
    diag = pd.DataFrame(metrics_rows)
    merged = frame.merge(diag, on=['method','composer','serie'], how='left', validate='one_to_one')
    merged['monoescala_seleccionada'] = merged['monoescala_seleccionada'].fillna(False).astype(bool)
    merged.to_csv(DEST/'resultados_clasificados.csv', index=False, encoding='utf-8-sig')
    diag.to_csv(DEST/'diagnosticos_mDFA.csv', index=False, encoding='utf-8-sig')
    return merged


# ===================== CORRELACIONES Y BOOTSTRAP =====================

def correlation(x, y, which='spearman'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < N_MIN_CORRELACION or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return np.nan, np.nan, len(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = spearmanr(x, y) if which == 'spearman' else pearsonr(x, y)
    return float(result.statistic), float(result.pvalue), len(x)


def partial_spearman(frame, xname, yname, include_method):
    needed = [xname, yname, 'N_notas', 'n_unique_intervalos', 'method']
    sub = frame[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < N_MIN_CORRELACION + 3:
        return np.nan
    n = len(sub)
    rx = rankdata(sub[xname].to_numpy(float))
    ry = rankdata(sub[yname].to_numpy(float))
    controls = [rankdata(np.log(sub['N_notas'].to_numpy(float))),
                rankdata(sub['n_unique_intervalos'].to_numpy(float))]
    if include_method and sub.method.nunique() > 1:
        dummies = pd.get_dummies(sub['method'], drop_first=True).to_numpy(float)
        controls.extend([dummies[:, j] for j in range(dummies.shape[1])])
    X = np.column_stack([np.ones(n)] + controls)
    dx = rx-X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    dy = ry-X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    if np.std(dx) < 1e-12 or np.std(dy) < 1e-12:
        return np.nan
    return float(np.corrcoef(dx, dy)[0, 1])


def rho_mask(frame, xname, yname, mask):
    sub = frame.loc[mask, [xname, yname]].replace([np.inf, -np.inf], np.nan).dropna()
    return correlation(sub[xname], sub[yname])[0]


def bootstrap_pareado(frame, xname, yname, selected_name, rng):
    """Misma muestra por compositor para estimar rho_all, rho_sel y diferencia."""
    groups = [part.index.to_numpy(dtype=int) for _, part in frame.groupby('composer', sort=True)]
    if len(groups) < 3 or BOOTSTRAP_REPS <= 0:
        return [np.nan]*8
    arr = []
    for _ in range(BOOTSTRAP_REPS):
        drawn = rng.integers(len(groups), size=len(groups))
        ids = np.concatenate([groups[j] for j in drawn])
        sample = frame.loc[ids]
        a = rho_mask(sample, xname, yname, np.ones(len(sample), dtype=bool))
        b = rho_mask(sample, xname, yname, sample[selected_name].to_numpy(bool))
        if np.isfinite(a) and np.isfinite(b):
            arr.append((a, b, abs(b)-abs(a), b-a))
    if len(arr) < max(30, 0.1*BOOTSTRAP_REPS):
        return [np.nan]*8
    vals = np.asarray(arr)
    intervals = np.quantile(vals, [.025, .975], axis=0).T
    return intervals.flatten().tolist()  # all(2), selected(2), delta_abs(2), delta(2)


def random_subsets_same_composers(frame, xname, yname, rng):
    """¿Es |rho_sel| excepcional entre subconjuntos del MISMO tamaño?

    Mantiene exactamente el número de seleccionadas en cada par
    (method, composer); sólo aleatoriza cuáles melodías entran. Así
    protege frente a una mejora artificial por reducir N o cambiar la
    proporción de compositores. Es una prueba exploratoria condicionada
    al patrón de selección, no una demostración de causalidad.
    """
    chosen = frame['monoescala_seleccionada'].to_numpy(bool)
    observed = rho_mask(frame, xname, yname, chosen)
    if not np.isfinite(observed):
        return dict(n_permutaciones_validas=0, p_vs_azar=np.nan,
                    mediana_abs_rho_azar=np.nan, IC95_abs_rho_azar_inf=np.nan,
                    IC95_abs_rho_azar_sup=np.nan,
                    ventaja_vs_mediana_azar=np.nan)
    clusters = []
    for _, group in frame.groupby(['method', 'composer'], sort=True):
        ids = group.index.to_numpy(dtype=int)
        count = int(frame.loc[ids,'monoescala_seleccionada'].sum())
        clusters.append((ids, count))
    values = []
    for _ in range(PERMUTACIONES_IGUAL_N):
        indices = np.concatenate([rng.choice(ids, size=count, replace=False)
                                  for ids, count in clusters if count > 0])
        r = rho_mask(frame, xname, yname, frame.index.isin(indices))
        if np.isfinite(r):
            values.append(abs(r))
    if len(values) < 30:
        return dict(n_permutaciones_validas=len(values), p_vs_azar=np.nan,
                    mediana_abs_rho_azar=np.nan, IC95_abs_rho_azar_inf=np.nan,
                    IC95_abs_rho_azar_sup=np.nan,
                    ventaja_vs_mediana_azar=np.nan)
    values = np.asarray(values)
    return dict(n_permutaciones_validas=len(values),
                p_vs_azar=float((1+np.count_nonzero(values >= abs(observed))) /
                                (1+len(values))),
                mediana_abs_rho_azar=float(np.median(values)),
                IC95_abs_rho_azar_inf=float(np.quantile(values,.025)),
                IC95_abs_rho_azar_sup=float(np.quantile(values,.975)),
                ventaja_vs_mediana_azar=float(abs(observed)-np.median(values)))


def correlation_summaries(frame):
    rows, comparisons, random_tests = [], [], []
    groups = [('todos', frame)]
    groups.extend([(str(m), part) for m, part in frame.groupby('method')])
    for group_name, sub0 in groups:
        sub = sub0.reset_index(drop=True).copy()
        partitions = {
            'todos': np.ones(len(sub), dtype=bool),
            'solo_R2_alto': sub['calidad_R2'].fillna(False).to_numpy(bool),
            'monoescala': sub['monoescala_seleccionada'].to_numpy(bool),
            'multiescala': sub['categoria'].eq('multiescala').to_numpy(bool),
            'indeterminado': sub['categoria'].eq('indeterminado').to_numpy(bool),
        }
        for xname in ('lambda_obs', 'kappa'):
            for yname in ('alpha_logbins', 'alpha'):
                for subset_name, mask in partitions.items():
                    chosen = sub.loc[mask].copy()
                    rho, p, n = correlation(chosen[xname], chosen[yname])
                    pear, ppear, _ = correlation(chosen[xname], chosen[yname], 'pearson')
                    par = partial_spearman(chosen, xname, yname,
                                           include_method=(group_name == 'todos'))
                    rows.append(dict(grupo=group_name, subset=subset_name,
                                     variable_Cd=xname, variable_alpha=yname,
                                     n=n, n_compositores=chosen.composer.nunique(),
                                     rho_spearman=rho, p_spearman=p,
                                     r_pearson=pear, p_pearson=ppear,
                                     rho_parcial_N_vocab=par))
                # Bootstrap únicamente entre all y selected, con los mismos com-positores.
                rng = np.random.default_rng(RANDOM_SEED + (
                    int.from_bytes(hashlib.blake2b((group_name+xname+yname).encode(),
                                                    digest_size=4).digest(), 'little')))
                ci = bootstrap_pareado(sub, xname, yname,
                                      'monoescala_seleccionada', rng)
                all_rho = rho_mask(sub, xname, yname, partitions['todos'])
                selected_rho = rho_mask(sub, xname, yname, partitions['monoescala'])
                perm_rng = np.random.default_rng(RANDOM_SEED + (
                    int.from_bytes(hashlib.blake2b(('permutation'+group_name+xname+yname)
                                                    .encode(), digest_size=4).digest(), 'little')))
                test = random_subsets_same_composers(sub, xname, yname, perm_rng)
                random_tests.append(dict(grupo=group_name, variable_Cd=xname,
                                         variable_alpha=yname,
                                         n_total=len(sub),
                                         n_seleccionadas=int(partitions['monoescala'].sum()),
                                         **test))
                comparisons.append(dict(
                    grupo=group_name, variable_Cd=xname, variable_alpha=yname,
                    n_total=int(partitions['todos'].sum()),
                    n_seleccionadas=int(partitions['monoescala'].sum()),
                    rho_todas=all_rho, rho_seleccionadas=selected_rho,
                    mejora_abs_rho=(abs(selected_rho)-abs(all_rho))
                    if np.isfinite(all_rho) and np.isfinite(selected_rho) else np.nan,
                    delta_rho=selected_rho-all_rho,
                    IC95_rho_todas_inf=ci[0], IC95_rho_todas_sup=ci[1],
                    IC95_rho_sel_inf=ci[2], IC95_rho_sel_sup=ci[3],
                    IC95_mejora_abs_inf=ci[4], IC95_mejora_abs_sup=ci[5],
                    IC95_delta_inf=ci[6], IC95_delta_sup=ci[7]))
    pd.DataFrame(rows).to_csv(DEST/'correlaciones_por_filtro.csv', index=False,
                              encoding='utf-8-sig')
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(DEST/'comparacion_hipotesis.csv', index=False,
                       encoding='utf-8-sig')
    pd.DataFrame(random_tests).to_csv(DEST/'prueba_submuestras_aleatorias.csv',
                                      index=False, encoding='utf-8-sig')
    return comparisons


# ============================= FIGURAS ==============================

def render_fluctuation(ax, row, sizes, fluct):
    stats, by_bins = diagnose_curve(sizes, fluct)
    lx, ly = np.log10(sizes), np.log10(fluct)
    bx, by, _ = by_bins[BINS_PRINCIPALES][1][0], by_bins[BINS_PRINCIPALES][1][1], by_bins[BINS_PRINCIPALES][2]
    fits = by_bins[BINS_PRINCIPALES][1][2]
    ax.plot(sizes, fluct, '.', color='0.73', markersize=2, label='F(s) / s: escalas originales')
    ax.plot(10**bx, 10**by, 'o', c='#20578a', markersize=3.5, label='Medianas log-bin')
    if 1 in fits:
        model = fits[1]
        xx = np.linspace(bx.min(), bx.max(), 160)
        yy = model['coeffs'][0] + model['coeffs'][1]*xx
        ax.plot(10**xx, 10**yy, '-', color='#198754', lw=1.7,
                label=f'Una pendiente α={stats["alpha_logbins"]:.3f}')
        best = min(fits.values(), key=lambda d: d['bic'])
        if best['n_reg'] > 1:
            yb = best['coeffs'][0] + best['coeffs'][1]*xx
            for c, knot in zip(best['coeffs'][2:], best['knots']):
                yb += c*np.maximum(0, xx-knot)
                ax.axvline(10**knot, color='#b24a48', ls=':', lw=1)
            ax.plot(10**xx, 10**yb, '--', color='#b24a48', lw=1.5,
                    label=f'{best["n_reg"]} regímenes; α=' +
                    ','.join(f'{a:.2f}' for a in best['slopes']))
    ax.set(xscale='log', yscale='log', xlabel='Escala s (notas)',
           ylabel='mDFA: F(s)/s')
    label = str(row['serie'])
    title = f'{row["method"]} / {row["composer"]} / {label}'
    if len(title) > 90:
        title = title[:87] + '...'
    ax.set_title(title, fontsize=9)
    color = {'monoescala_compatible':'#198754', 'multiescala':'#b24a48',
             'indeterminado':'#a97327'}.get(stats['categoria'], '0.4')
    info = (f'{stats["categoria"]} | R²={stats["R2_logbins"]:.3f} | '
            f'ΔBIC={stats["ganancia_BIC_multiescala"]:.1f}\n'
            f'Δα mitades={stats["delta_alpha_mitades"]:.3f} | '
            f'consenso={stats["n_consenso_monoescala"]}/{len(BINS_SENSIBILIDAD)}')
    ax.text(.02, .02, info, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=7, color=color,
            bbox={'facecolor':'white', 'alpha':.86, 'edgecolor':'0.8'})
    ax.grid(alpha=.16, which='both')
    return stats


def generate_fluctuation_plots(frame):
    ordered = frame[frame['estado_revision'] == 'ok'].sort_values(
        ['categoria','method','composer','serie'], kind='stable').reset_index(drop=True)
    if not len(ordered):
        return
    cache = DEST / 'cache_curvas'
    if GENERAR_PNG_INDIVIDUALES_ACEPTADAS:
        path = DEST / 'F_s_monoescala_individual'
        path.mkdir(exist_ok=True)
        selected = ordered[ordered['monoescala_seleccionada']]
        for _, row in selected.iterrows():
            with np.load(cache/(row['id_curva']+'.npz')) as p:
                fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
                render_fluctuation(ax, row, p['sizes'], p['fluct'])
                ax.legend(loc='upper right', fontsize=7)
                fig.savefig(path/(file_label(row['method'],row['composer'],row['serie'])+'.png'),
                            dpi=DPI)
                plt.close(fig)
    if GENERAR_PNG_RECHAZADAS_EJEMPLO:
        path = DEST / 'F_s_rechazadas_ejemplos'
        path.mkdir(exist_ok=True)
        rejected = ordered[~ordered['monoescala_seleccionada']].copy()
        rejected = rejected.sort_values('ganancia_BIC_multiescala', ascending=False)
        for _, row in rejected.head(GENERAR_PNG_RECHAZADAS_EJEMPLO).iterrows():
            with np.load(cache/(row['id_curva']+'.npz')) as p:
                fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
                render_fluctuation(ax, row, p['sizes'], p['fluct'])
                ax.legend(loc='upper right', fontsize=7)
                fig.savefig(path/(file_label(row['method'],row['composer'],row['serie'])+'.png'),
                            dpi=DPI)
                plt.close(fig)
    if GENERAR_ATLAS:
        folder = DEST/'atlas_F_s_todas'
        folder.mkdir(exist_ok=True)
        pages = math.ceil(len(ordered)/CURVAS_POR_PAGINA)
        for page in range(pages):
            subset = ordered.iloc[page*CURVAS_POR_PAGINA:(page+1)*CURVAS_POR_PAGINA]
            ncols = 3
            nrows = math.ceil(len(subset)/ncols)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(15, 3.8*nrows), squeeze=False,
                                     constrained_layout=True)
            for ax, (_, row) in zip(axes.ravel(), subset.iterrows()):
                with np.load(cache/(row['id_curva']+'.npz')) as p:
                    render_fluctuation(ax, row, p['sizes'], p['fluct'])
            for ax in axes.ravel()[len(subset):]:
                ax.axis('off')
            fig.savefig(folder/f'atlas_{page+1:04d}_de_{pages:04d}.png', dpi=DPI)
            plt.close(fig)
            if (page+1) % 10 == 0 or page+1 == pages:
                print(f'Atlas: {page+1}/{pages} páginas', flush=True)


def plot_corr(frame, group_name):
    sub = frame[frame['estado_revision'] == 'ok'].copy()
    if group_name != 'todos':
        sub = sub[sub['method'] == group_name]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, xcol in zip(axes, ('lambda_obs', 'kappa')):
        mask = sub['monoescala_seleccionada'].astype(bool)
        base = sub[~mask]
        sel = sub[mask]
        ax.scatter(base[xcol], base['alpha_logbins'], s=17, alpha=.35,
                   c='0.45', label='No seleccionadas')
        ax.scatter(sel[xcol], sel['alpha_logbins'], s=22, alpha=.8,
                   c='#198754', label='Monoescala compatible')
        rall, _, n = correlation(sub[xcol], sub['alpha_logbins'])
        rsel, _, ns = correlation(sel[xcol], sel['alpha_logbins'])
        ax.text(.04, .97,
                f'Todas: ρ={rall:.3f} (n={n})\nMonoescala: ρ={rsel:.3f} (n={ns})',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=.9, edgecolor='0.8'))
        ax.set(xlabel=xcol, ylabel=r'$\alpha_{mDFA}$ (log-bins)')
        ax.grid(alpha=.15)
    axes[1].legend(loc='best', fontsize=8)
    fig.suptitle(f'{group_name}: relación Cd–mDFA, con/sin filtro de escala')
    fig.savefig(DEST/f'correlacion_monoescala_{group_name}.png', dpi=DPI)
    plt.close(fig)


def plot_selection_controls(frame):
    sub = frame[frame['estado_revision'] == 'ok']
    mask = sub['monoescala_seleccionada'].astype(bool)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4), constrained_layout=True)
    for ax, col, xlabel in zip(axes[:2], ['N_notas','n_unique_intervalos'],
                               ['Longitud MIDI','Vocabulario intervalos']):
        a = sub.loc[mask,col].dropna()
        b = sub.loc[~mask,col].dropna()
        if len(a):
            ax.hist(a, bins=20, density=True, alpha=.5, label='Seleccionadas')
        if len(b):
            ax.hist(b, bins=20, density=True, alpha=.5, label='No seleccionadas')
        if col == 'N_notas':
            ax.set_xscale('log')
        ax.set(xlabel=xlabel, ylabel='Densidad')
        ax.grid(alpha=.15)
    if mask.any():
        axes[2].scatter(sub.loc[mask,'N_notas'],
                        sub.loc[mask,'ganancia_BIC_multiescala'], s=15, alpha=.6)
    if (~mask).any():
        axes[2].scatter(sub.loc[~mask,'N_notas'],
                        sub.loc[~mask,'ganancia_BIC_multiescala'],
                        s=13, c='0.6', alpha=.3)
    axes[2].axhline(GANANCIA_BIC_MULTIESCALA, ls='--', color='0.4')
    axes[2].set(xscale='log', xlabel='Longitud MIDI',
                ylabel='BIC(1 régimen) - mejor BIC multirégimen')
    axes[0].legend(fontsize=8)
    fig.savefig(DEST/'diagnostico_sesgo_seleccion.png', dpi=DPI)
    plt.close(fig)


def main():
    validate_paths_and_config()
    original = load_csv()
    print(f'Archivo original: {CSV_ORIGINAL}\nMelodías en CSV: {len(original)}')
    composers = load_midi_corresponding(original)
    merged = run_all(original, composers)
    comparisons = correlation_summaries(merged)
    counts = (merged.groupby(['method','categoria'], dropna=False)
              .agg(n=('serie','size'), n_compositores=('composer','nunique'))
              .reset_index())
    counts.to_csv(DEST/'resumen_categorias.csv', index=False, encoding='utf-8-sig')
    qa = merged.loc[merged['estado_revision']=='ok', 'delta_alpha_vs_original'].abs()
    print('\nAUDITORÍA: |alpha_refit_full - alpha_original|')
    print(qa.describe().to_string() if len(qa) else 'No hay valores')
    print('\nCLASIFICACION:')
    print(counts.to_string(index=False))
    print('\nCOMPARACIÓN DE HIPÓTESIS (alpha_logbins):')
    display = comparisons[comparisons.variable_alpha == 'alpha_logbins']
    print(display[['grupo','variable_Cd','n_total','n_seleccionadas',
                   'rho_todas','rho_seleccionadas','mejora_abs_rho',
                   'IC95_mejora_abs_inf','IC95_mejora_abs_sup']].to_string(index=False))
    for group_name in ['todos'] + sorted(merged.method.unique().tolist()):
        plot_corr(merged, group_name)
    plot_selection_controls(merged)
    generate_fluctuation_plots(merged)
    print(f'\nResultados en: {DEST.resolve()}')
    print('Consultar prueba_submuestras_aleatorias.csv: p_vs_azar compara')
    print('selección monoescala con otras selecciones del mismo tamaño por compositor.')
    print('IMPORTANTE: categoria monoescala_compatible no demuestra una ley de potencia.')
    print('Revisar las figuras, span_decadas, sensibilidad a log-bins y sesgo de selección.')


if __name__ == '__main__':
    main()
