import numpy as np
from typing import List, Tuple

def extract_melody_multi_voices(voices: List[np.ndarray], del_repeats = True):
    """
    Extrae la melodía de varias voces.

    voices: lista de arrays (uno por voz), cada uno de forma Nx6:
        [onset, offset, dur, midi, compas, tipo de compas]

    Regla:
      - Por compás, se elige una voz que actúa como melodía:
          * Primero se intenta con main_voice.
          * Si esa voz no tiene ninguna nota en el compás (midi > silence_value),
            se prueba con la siguiente voz, y así sucesivamente.
      - En el compás y voz elegidos:
          * Para cada onset, si hay múltiples notas, se conserva la nota
            con valor MIDI más alto y las demás se convierten en silencio.
      - En las demás voces en ese compás:
          * Todas las notas se convierten en silencio.
    
    Devuelve:
      - melody_matrix: matriz con todas las filas de todas las voces (apiladas),
        pero sólo la melodía activa; el resto con midi = silence_value.
      - melody_series: array 1D con la secuencia MIDI de la melodía sin silencios,
        ordenada por compás y onset.
    """
    def quitar_filas_duplicadas(A):
        """
        Devuelve A sin filas duplicadas, conservando la primera vez que aparece cada fila.
        Funciona aunque A tenga dtype=object.
        """
        vistos = set()
        indices_a_conservar = []

        for i, fila in enumerate(A):
            # Convertimos la fila a tupla para que sea hashable
            key = tuple(fila.tolist() if hasattr(fila, "tolist") else fila)
            if key not in vistos:
                vistos.add(key)
                indices_a_conservar.append(i)

        indices_a_conservar = np.array(indices_a_conservar, dtype=int)
        return A[indices_a_conservar]
    
    # Copias para no modificar los originales
    voices_out = [v.copy() for v in voices]

    # Obtener todos los compases presentes en cualquier voz
    all_compases = np.unique(
        np.concatenate([v[:, 4] for v in voices_out if v.size > 0])
    )
    all_compases = np.sort(all_compases)

    voz = np.empty((0, 6))
    for k in range(len(all_compases)):
        for i,A in enumerate(voices_out):
            # compas k
            mask = (A[:, 4] == k) 
            A = A[mask]
            # Eliminar iguales en onset, offset, dur
            mask = (A[:, 0] != A[:, 1]) & \
                (A[:, 2] != 0.0) 
            A = A[mask]
            # A las notas que se repitan en todo los onsets, convertirlas a silencio
            if del_repeats:
                for target in np.unique(A[:,3]):
                    missing = []
                    for g in np.unique(A[:,0]):
                        mask_g = (A[:, 0] == g)
                        if not np.any(A[:,3][mask_g] == target):
                            missing.append(k)
                    all_ok = (len(missing) == 0)
                    if all_ok:
                        mask_cambiar = (A[:, 3] == target)
                        A[mask_cambiar, 3] = -1.0
            
            # Si solo hay una nota sola en el primer onset, cambiarla por un silencio
            # pos_idx = np.nonzero(A[:,3] > 0.0)[0]
            # if pos_idx.size == 1:
            #     i = pos_idx[0]
            #     if A[:,0][i] == np.min(A[:,0]) and A[:,2][i] == np.max(A[:,1]):
            #         A[i,3] = -1.0

            # Cambiar midi por -1 si por lo menos una nota
            for g in np.unique(A[:,0]):
                mask_g = (A[:, 0] == g)
                col3_g = A[mask_g, 3]
                hay_positivos = np.any(col3_g > 0)
                if hay_positivos:
                    max_val = col3_g.max()
                    # dentro del grupo, todo lo que sea menor al máximo se pone a -1
                    mask_cambiar = mask_g & (A[:, 3] < max_val)
                    A[mask_cambiar, 3] = -1.0
                
            hay_positivos_grupo = np.any(A[:,3] > 0)

            #totalmente iguales
            A = quitar_filas_duplicadas(A)
            # Añadir bloque a voz
            if hay_positivos_grupo:
                voz = np.vstack([voz, A]) 
                break
            elif not hay_positivos_grupo and i == len(voices)-1:
                voz = np.vstack([voz, A]) 

    melody_series = voz[:,3][voz[:,3] != -1]

    return voz, melody_series
