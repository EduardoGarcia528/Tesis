import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict, Any
from collections import Counter

def interpolador(subject, method, size):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    data = subject
    x = np.arange(len(data))
    
    # Crear 'size' puntos equidistantes
    x_new = np.linspace(0, len(data) - 1, size*(len(data)-1) + len(data))
    
    if method == 'lineal':
        data_interp = np.interp(x_new, x, data)
    elif method == 'herm':
        interpolator = PchipInterpolator(x, data)
        data_interp = interpolator(x_new)
    
    return data_interp

def interpolador_constante(subject):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    x = np.arange(2)
    data_interp = np.array([])
    diferencias =np.abs(np.diff(subject))
    if (np.max(diferencias) - np.min(diferencias)) <= 1e-5:
        return subject
    diferencias_sin_0 = diferencias[diferencias != 0]
    diferencias = np.round(diferencias/np.min(diferencias_sin_0))
    for i in range(len(subject)-1):
        data = subject[i:i+2]
        if diferencias[i] == 0:
            data_interp = np.concatenate((data_interp,data[:-1]))
            continue
        size = int(diferencias[i])- 1
        # Crear 'size' puntos equidistantes
        x_new = np.linspace(0, 1, size + 2)
        interp_step = np.interp(x_new, x, data)
        data_interp = np.concatenate((data_interp,interp_step[:-1]))
    data_interp = np.concatenate((data_interp,np.array([data[-1]])))

    return data_interp

def generar_uniforme_centrada(n, varianza):
    # Calcular el límite superior e inferior de la distribución uniforme
    limite = np.sqrt(varianza) # varianza*3
    # Generar n números aleatorios con distribución uniforme entre -limite y limite
    return np.random.uniform(-limite, limite, n)

def brownian_bridge(t0, tT, x0, xT, n_steps):
    t = np.linspace(t0, tT, n_steps)
    #Generar movimiento browniano con media cero
    W = np.random.normal(0, np.sqrt(t[1] - t[0]), size=n_steps-1)
    W = np.insert(np.cumsum(W), 0, 0)  # Inserta W(0) = 0 y suma acumulativa
    # Interpolación del Brownian Bridge
    X = x0 + (t - t0) / (tT - t0) * (xT - x0) + W - (t - t0) / (tT - t0) * W[-1]
    return t, X

def interpolador_estocastico(s_0_discreto, n_steps):
    t_list, X_list = [], []
    for i in range(len(s_0_discreto)- 1):
        t0 = i
        tT = i+1
        x0 = s_0_discreto[i]
        xT = s_0_discreto[i+1]
        t, X = brownian_bridge(t0, tT, x0, xT, n_steps)
        t_list = np.concatenate((t_list, t))
        X_list = np.concatenate((X_list, X))
    return X_list

def remove_consecutive_duplicates(data, tolerance=1e-1):
    """
    Remueve duplicados consecutivos en los datos con un margen de tolerancia.

    Args:
        data (list or array): Array de datos a procesar.
        tolerance (float): Margen de tolerancia para considerar valores similares.

    Returns:
        list: Array con duplicados consecutivos removidos.
    """

    result = [data[0]]  # Comenzamos con el primer elemento
    for i in range(1, len(data)):
        if abs(data[i] - data[i - 1]) > tolerance:
            result.append(data[i])

    return np.array(result)

def derivada_index(array):
    array = np.asarray(array)
    derivada = np.abs(np.diff(array))  # Calcula la diferencia entre puntos subsecuentes
    return derivada

def remover_duplicados(array):
    resultado = []
    vistos = set()
    for x in array:
        if x not in vistos:
            resultado.append(x)
            vistos.add(x)
    return resultado


@njit
def distancia(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

@njit
def mejor_vector(p1, p2):
    # Precomputar diferencias en los 9 cuadrantes
    diffs = [
        [p2[0], p2[1]],
        [p2[0], p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1]],
        [p2[0] + 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0], p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1]],
        [p2[0] - 2 * np.pi, p2[1] + 2 * np.pi],
    ]
    # Encontrar el índice con menor distancia
    d_og = distancia(p1,p2)
    min_idx = 0
    for i in range(9):
        d = distancia(p1, diffs[i])
        if d < d_og:
            min_idx = i
            d_og = d
    p2 = diffs[min_idx]
    return [p2[0] - 2*p1[0], p2[1] - 2*p1[1]]

@njit
def calcular_angulos(vectores):
    n = len(vectores) - 1
    angulos = np.empty(n)
    for i in range(n):
        v1 = vectores[i]
        v2 = vectores[i + 1]
        norm_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if norm_v1 == 0 or norm_v2 == 0:
            angulo = 0.0
        else:
            v1n0 = v1[0] / norm_v1
            v1n1 = v1[1] / norm_v1
            v2n0 = v2[0] / norm_v2
            v2n1 = v2[1] / norm_v2
            dot = v1n0 * v2n0 + v1n1 * v2n1
            if dot > 1.0: dot = 1.0
            if dot < -1.0: dot = -1.0
            angulo = np.arccos(dot)
            cruz = v1[0] * v2[1] - v1[1] * v2[0]
            if cruz > 0:
                angulo = np.pi - angulo
            elif cruz == 0 and angulo < 0:
                angulo = np.pi
            elif cruz < 0:
                angulo += np.pi
        angulos[i] = angulo
    return angulos

def caminata_univariante(X, tau):
    x1 = X[tau:]
    y1 = X[:-tau]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    n = len(ff1) - 1
    vectores = np.empty((n, 2))
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def caminata_bivariante(X, Y):
    x1 = X
    y1 = Y
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    n = len(ff1) - 1
    vectores = np.empty((n, 2))
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def indice_J(angulos):
    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    return J

def entropia_shannon(x, bins=100):
    x = np.array(x, dtype=float)   # fuerza a convertir a float
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    p = hist / hist.sum()
    H = -np.sum(p * np.log2(p))
    return H / np.log2(bins)

def entropia_permutacion(x, m=4, tau=1):
    n = len(x)
    if n < (m - 1) * tau + 1:
        return np.nan

    patrones = []
    for i in range(n - (m - 1) * tau):
        ventana = x[i:i + tau * m:tau]
        orden = tuple(np.argsort(ventana))
        patrones.append(orden)

    cuenta = Counter(patrones)
    total = sum(cuenta.values())
    p = np.array(list(cuenta.values())) / total
    H = -np.sum(p * np.log2(p))
    H_norm = H / np.log2(math.factorial(m))  # normalización
    return H_norm

def diff_S(d, angulos):
    if d == 0:
        return entropia_permutacion(angulos)
    for _ in range(d):
        angulos = np.diff(angulos)

    return entropia_permutacion(angulos)

def main(serie,d,bivariante):
    if bivariante == False:
        vectores = caminata_univariante(serie,tau = 1)
    else:
        print("Bivariante")
        vectores = caminata_bivariante(serie,bivariante)
    angulos = calcular_angulos(vectores)
    # plt.plot(angulos[:20_000], '.')
    # plt.title('Angulos alpha')
    # plt.xlabel('n')
    # plt.ylabel('ángulo')
    # plt.show()
    entropia = diff_S(d,angulos=angulos)
    print(d)
    J = indice_J(angulos)
    if not np.isfinite(entropia):
        entropia = 0
    return J, entropia

def juntar_y_ordenar(arr1, arr2):
    """
    Recibe dos arrays de dimensión Nx5, los junta y los ordena por:
    1) Columna 5 (índice 4)
    2) Columna 0 (índice 0)
    3) Columna 3 (índice 3)
    """
    # Verificar que ambos tengan 5 columnas
    if arr1.shape[1] != 5 or arr2.shape[1] != 5:
        raise ValueError("Ambos arrays deben tener exactamente 5 columnas")

    # Concatenar verticalmente
    arr = np.vstack((arr1, arr2))

    # Ordenar por múltiples columnas (última clave es la de mayor prioridad)
    # Prioridad: col5 > col0 > col3
    idx_orden = np.lexsort((arr[:, 3], arr[:, 0], arr[:, 4]))

    return arr[idx_orden]

def extract_melody_grow(
    arr: np.ndarray,
    silence_value: float = -1.0,
    seed_threshold: float = 56.0,     # semillas: pitch > 58
    neighbor_semitones: float = 6.0   # propagación: |Δpitch| < 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Entrada:
      arr: Nx5 -> [onset, offset, dur, midi, compas]
    Reglas:
      A) Por (compás, onset): conservar solo la nota más aguda (max MIDI); demás filas del onset -> silencio.
      B) Semillas iniciales de melodía:
         - onsets con pitch > seed_threshold
         - primer y último onset de TODA la partitura (indiscriminadamente)
         - primer y último onset de CADA compás si su duración es mayor que el vecino en ese compás
           (si solo hay un onset en el compás, se incluye).
      C) Propagación iterativa global: un onset incluido “arrastra” a sus vecinos (anterior/siguiente en la línea
         temporal global) si |Δpitch| < neighbor_semitones (y ambos tienen nota).
    Devuelve:
      - melody_matrix: Nx5 con melodía activa y el resto como silencios (midi = silence_value)
      - melody_series: 1D con pitches de la melodía en orden temporal (sin silencios)
    """
    assert arr.shape[1] == 5, "La matriz debe ser Nx5"
    out = arr.copy()

    # ---------- Paso A: conservar SOLO la nota más aguda por (compás, onset) ----------
    unique_compases = np.unique(out[:, 4])
    groups_info: Dict[Tuple[float, float], Dict[str, Any]] = {}
    timeline_keys: List[Tuple[float, float]] = []

    for comp in np.sort(unique_compases):
        mask_c = (out[:, 4] == comp)
        rows_c = np.where(mask_c)[0]
        if rows_c.size == 0:
            continue
        onsets_c = np.unique(out[rows_c, 0])
        for onset_val in np.sort(onsets_c):
            key = (float(comp), float(onset_val))
            idxs = rows_c[out[rows_c, 0] == onset_val]
            if idxs.size == 0:
                continue
            pitches = out[idxs, 3]
            durs    = out[idxs, 2]

            # conservar indiscriminadamente la nota más aguda (si empata: mayor duración; luego primera)
            rel = np.argmax(pitches)
            same_max = np.where(pitches == pitches[rel])[0]
            if same_max.size > 1:
                rel = same_max[np.argmax(durs[same_max])]
            keep_abs   = int(idxs[rel])
            keep_pitch = float(out[keep_abs, 3])
            keep_dur   = float(out[keep_abs, 2])

            # silenciar TODAS las demás filas del onset
            for r in idxs:
                if r != keep_abs:
                    out[r, 3] = silence_value

            groups_info[key] = {
                "selected_row": keep_abs,
                "selected_pitch": keep_pitch,
                "selected_dur": keep_dur,
            }
            timeline_keys.append(key)

    # Línea temporal global: (compás asc, onset asc)
    timeline_keys.sort(key=lambda k: (k[0], k[1]))
    selected_pitches = np.array([groups_info[k]["selected_pitch"] for k in timeline_keys], dtype=float)
    selected_durs    = np.array([groups_info[k]["selected_dur"]   for k in timeline_keys], dtype=float)

    # ---------- Paso B: semillas ----------
    included = selected_pitches > seed_threshold  # semillas > 58

    # (B1) primer y último onset de TODA la partitura → siempre melodía
    if len(timeline_keys) > 0:
        included[0]  = True
        included[-1] = True

    # (B2) primer y último onset de CADA compás si su duración > vecino interno
    # Construimos índice de onsets por compás dentro de timeline
    comp_to_indices: Dict[float, List[int]] = {}
    for i, (comp, onset) in enumerate(timeline_keys):
        comp_to_indices.setdefault(comp, []).append(i)

    for comp, idxs in comp_to_indices.items():
        if len(idxs) == 1:
            # solo un onset en el compás → incluirlo
            included[idxs[0]] = True
        else:
            first_i, last_i = idxs[0], idxs[-1]
            # comparar duraciones dentro del compás
            # primer onset > su siguiente?
            if selected_durs[first_i] > selected_durs[idxs[1]]:
                included[first_i] = True
            # último onset > su anterior?
            if selected_durs[last_i] > selected_durs[idxs[-2]]:
                included[last_i] = True

    # ---------- Paso C: propagación iterativa (entre compases también) ----------
    def valid_pitch(p: float) -> bool:
        return p > silence_value  # asume que silence_value < cualquier MIDI válido

    changed = True
    while changed:
        changed = False
        for i in range(len(timeline_keys)):
            if not included[i]:
                continue
            p_i = selected_pitches[i]
            if not valid_pitch(p_i):
                continue
            # vecino global anterior
            if i - 1 >= 0 and not included[i - 1]:
                p_prev = selected_pitches[i - 1]
                if valid_pitch(p_prev) and abs(p_prev - p_i) < neighbor_semitones:
                    included[i - 1] = True
                    changed = True
            # vecino global siguiente
            if i + 1 < len(timeline_keys) and not included[i + 1]:
                p_next = selected_pitches[i + 1]
                if valid_pitch(p_next) and abs(p_next - p_i) < neighbor_semitones:
                    included[i + 1] = True
                    changed = True

    # Aplicar 'included' a la matriz: onsets no incluidos → silenciar su fila seleccionada
    for i, key in enumerate(timeline_keys):
        if not included[i]:
            sel_row = groups_info[key]["selected_row"]
            out[sel_row, 3] = silence_value

    # Serie 1D (pitches incluidos en orden temporal)
    melody_series = selected_pitches[included]
    melody_series = melody_series[melody_series > silence_value]

    return out, melody_series
