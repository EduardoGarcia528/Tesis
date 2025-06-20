# =============================================================================
# Script: procesa_RR_SBP.py (versión 3)
# -----------------------------------------------------------------------------
#   • Procesa archivos .txt con 4 columnas (t_RR, RR, t_SBP, SBP) para pacientes
#     0001–1121.
#   • Calcula **solo** índice J univariante (RR y SBP) y **gamma** para cada
#     serie.
#   • NO guarda archivos por paciente.  ❗️
#   • Salida única por índice / conjunto de parámetros → matrices 1121 × 3:
#       resultados/J_method-<METHOD>_size-<SIZE>.npy
#       resultados/gamma_gamma-<INDICE_GAMMA>_MS-<MS>.npy
#     Columnas: [ID, valor_RR, valor_SBP] (np.nan cuando falta).
# =============================================================================

from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count, freeze_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# === FUNCIONES DE CÁLCULO =====================================================
# -----------------------------------------------------------------------------

def J_univariante(X, tau=1, corte=False):
    def distancia(p1, p2):
        return np.linalg.norm(np.array(p2) - np.array(p1))
    X = np.array(X)
    x1 = X[tau:]
    y1 = X[:-tau]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    vectores = []
    for i in range(len(ff1) - 1):
        p1 = [ff1[i], ff2[i]]
        p2 = [ff1[i + 1], ff2[i + 1]]
        cuadrante = [
            [p2[0] - p1[0], p2[1] - p1[1]],
            [p2[0] - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
        ]
        distancias = np.array([distancia(p1, c) for c in cuadrante])
        p2 = cuadrante[np.argmin(distancias)]
        vectores.append([p2[0] - p1[0], p2[1] - p1[1]])

    vectores = np.array(vectores)
    norms = np.linalg.norm(vectores, axis=1, keepdims=True)
    v_norm = np.where(norms == 0, vectores, vectores / norms)
    
    angulos = np.arccos(np.clip(np.einsum('ij,ij->i', v_norm[:-1], v_norm[1:]), -1.0, 1.0))
    cruces = np.cross(v_norm[:-1], v_norm[1:])
    angulos = np.where(cruces > 0, np.pi - angulos, angulos)
    angulos = np.where((cruces == 0) & (angulos < 0), np.pi, angulos)
    angulos = np.where(cruces < 0, angulos + np.pi, angulos)

    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    
    return J


def calcular_gamma_opt(data, gamma_index, MS):
    N = len(data)
    sd = np.std(data, ddof=1)
    eps = sd / MS
    maxdat = np.max(data)
    data = np.concatenate(([0.0], data, [maxdat + 100 * eps]))  # data[0], data[N+1]

    Ci = [0] * (gamma_index + 2)  # Necesitamos Ci[i-1], Ci[i], Ci[i+1]

    for j in range(1, N + 1):
        for i in range(1, j):
            k = 0
            while k <= gamma_index + 1 and abs(data[i + k] - data[j + k]) <= eps:
                if k in (gamma_index - 1, gamma_index, gamma_index + 1):
                    Ci[k] += 1
                k += 1

    norm = 2.0 / (N * (N - 1))
    C = [1.0] + [0.0] * (gamma_index + 1)
    for k in (gamma_index - 1, gamma_index, gamma_index + 1):
        if k >= 1:
            C[k] = Ci[k] * norm

    denominator = C[gamma_index - 1] * C[gamma_index + 1]
    gamma = 0.0
    if denominator != 0:
        gamma = 1.0 - (C[gamma_index] ** 2) / denominator

    return gamma


def interpolar(serie: np.ndarray, method: str, size: int) -> np.ndarray:
    if size<=0:
        return serie
    x = np.arange(len(serie))
    x_new = np.linspace(0, len(serie)-1, size*(len(serie)-1)+len(serie))
    if method == 'lineal':
        return np.interp(x_new, x, serie)
    if method == 'herm':
        return PchipInterpolator(x, serie)(x_new)
    raise ValueError('Método de interpolación desconocido')

# -----------------------------------------------------------------------------
# === CONFIGURACIÓN ============================================================
# -----------------------------------------------------------------------------

DATA_DIR          = 'DatosRRnSBP700'
EXCEL_EXCLUSIONES = 'CRPagingBetas.xlsx'
EXCEL_SHEET       = 'ClavesBetasAging'
PACIENTES         = [f'{i:04d}' for i in range(1, 1122)]  # 0001–1121

# Parámetros
METHOD_INTERP = 'lineal'   # 'lineal' | 'herm'
SIZE_INTERP   = 0          # 0 ⇒ sin interpolación
MS_PARAM      = 3.0
INDICE_GAMMA  = 1
OUTPUT_DIR    = 'resultados'
GUARDAR_PNG   = False      # figuras opcionales
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# === EXCLUSIONES =============================================================
# -----------------------------------------------------------------------------

def cargar_exclusiones():
    df = pd.read_excel(EXCEL_EXCLUSIONES, sheet_name=EXCEL_SHEET, dtype={'ID':str})
    df['ID'] = df['ID'].str.zfill(4)
    df['ExcludeRR']  = df['ExcludeRR'].astype(int).astype(bool)
    df['ExcludeSBP'] = df['ExcludeSBP'].astype(int).astype(bool)
    return {r.ID: {'RR': r.ExcludeRR, 'SBP': r.ExcludeSBP} for r in df.itertuples()}

EXCLUSIONES = cargar_exclusiones()

# -----------------------------------------------------------------------------
# === PROCESO POR PACIENTE =====================================================
# -----------------------------------------------------------------------------

def procesar_paciente(pid: str):
    excl = EXCLUSIONES.get(pid, {'RR':False, 'SBP':False})
    txt = os.path.join(DATA_DIR, f'{pid}RRnSBP700.txt')
    if not os.path.exists(txt):
        print('[!] Falta', txt)
        return pid, np.nan, np.nan, np.nan, np.nan  # id, J_rr, J_sbp, G_rr, G_sbp
    dat = np.loadtxt(txt)
    if dat.shape[1] < 4:
        print('[!] Formato raro en', txt)
        return pid, np.nan, np.nan, np.nan, np.nan

    rr, sbp = dat[:,1], dat[:,3]
    J_rr = J_sbp = G_rr = G_sbp = np.nan
    if not excl['RR']:
        rr_i = interpolar(rr, METHOD_INTERP, SIZE_INTERP)
        J_rr = J_univariante(rr_i)
        G_rr = calcular_gamma_opt(rr_i, INDICE_GAMMA, MS_PARAM)
    if not excl['SBP']:
        sbp_i = interpolar(sbp, METHOD_INTERP, SIZE_INTERP)
        J_sbp = J_univariante(sbp_i)
        G_sbp = calcular_gamma_opt(sbp_i, INDICE_GAMMA, MS_PARAM)

    return pid, J_rr, J_sbp, G_rr, G_sbp

# -----------------------------------------------------------------------------
# === MAIN ====================================================================
# -----------------------------------------------------------------------------

def main():
    freeze_support()
    with Pool(min(cpu_count(), len(PACIENTES))) as pool:
        res = pool.map(procesar_paciente, PACIENTES)

    # Matrices globales
    J_mat     = np.full((len(PACIENTES), 3), np.nan)
    gamma_mat = np.full_like(J_mat, np.nan)

    for idx, (pid, J_rr, J_sbp, G_rr, G_sbp) in enumerate(res):
        J_mat[idx, 0] = gamma_mat[idx, 0] = int(pid)
        J_mat[idx, 1:], gamma_mat[idx, 1:] = [J_rr, J_sbp], [G_rr, G_sbp]

    np.save(os.path.join(OUTPUT_DIR, f'J_method-{METHOD_INTERP}_size-{SIZE_INTERP}.npy'),       J_mat)
    np.save(os.path.join(OUTPUT_DIR, f'gamma_gamma-{INDICE_GAMMA}_MS-{MS_PARAM}.npy'), gamma_mat)

    # --- Correlación opcional -------------------------------------------------
    if GUARDAR_PNG:
        corrs = {}
        for col, serie in enumerate(['RR','SBP'], start=1):
            mask = ~np.isnan(J_mat[:,col]) & ~np.isnan(gamma_mat[:,col])
            if np.sum(mask) >= 2:
                corrs[serie] = pearsonr(J_mat[mask,col], gamma_mat[mask,col])[0]
        if corrs:
            plt.figure(figsize=(4,4))
            plt.bar(corrs.keys(), corrs.values(), color='steelblue')
            plt.ylabel('r')
            plt.title('Correlación J vs γ')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR,
                f'corr_J-gamma_method-{METHOD_INTERP}_size-{SIZE_INTERP}_gamma-{INDICE_GAMMA}_MS-{MS_PARAM}.png'))
            plt.close()

    print('[✓] Matrices globales guardadas en', OUTPUT_DIR)

if __name__ == '__main__':
    main()
