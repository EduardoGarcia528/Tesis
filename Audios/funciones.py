import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict, Any
from collections import Counter

import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit


def remover_duplicados(array):
    resultado = []
    vistos = set()
    for x in array:
        if x not in vistos:
            resultado.append(x)
            vistos.add(x)
    return resultado


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


import numpy as np
from typing import Tuple

def extract_melody_simple(arr: np.ndarray,
                          silence_value: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae la melodía principal seleccionando la nota más aguda en cada onset.
    Todas las demás notas del mismo onset se convierten en silencio.
    
    Input:
        arr: matriz Nx5 con columnas [onset, offset, dur, midi, compas]
    
    Output:
        melody_matrix: misma matriz pero con solo la melodía (resto = silence_value)
        melody_series: array 1D con la secuencia MIDI de la melodía sin silencios
    """

    out = arr.copy()

    # Obtener lista de onsets únicos (globales)
    unique_onsets = np.unique(out[:, 0])

    for onset in unique_onsets:
        # Filas con ese onset
        idxs = np.where(out[:, 0] == onset)[0]
        pitches = out[idxs, 3]

        # Selección indiscriminada: nota más aguda
        # (si hubiera silencios ya codificados, seguirán siendo menor)
        rel_max = np.argmax(pitches)
        keep_abs = idxs[rel_max]

        # Silenciar las demás filas
        for r in idxs:
            if r != keep_abs:
                out[r, 3] = silence_value

    # Extraer serie 1D de la melodía
    melody_series = out[out[:, 3] != silence_value, 3]

    return out, melody_series
