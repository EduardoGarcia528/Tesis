import numpy as np
from music21 import stream, note, chord
from typing import List, Tuple

def process_notes_by_voice(m21_data, start_measure=1, verbose=True):
    """
    Procesa una partitura de music21 y devuelve una lista de matrices,
    una por cada voz (part) del .mxl.

    Incluye notas, acordes y silencios (rests).

    Columnas de cada matriz (por voz), compatibles con generador_partitura:
        0: onset_local   (origen 0.0 en cada compás)
        1: offset_local  (onset_local + duration)
        2: duration      (en negras, igual a quarterLength original)
        3: midi value    (nota: pitch.ps, silencio: -1.0)
        4: #measure      (numeración ajustada con start_measure)
        5: timeSignature (string, por ejemplo '4/4', '3/8', etc.)
    """

    # ------------------------------------------------------------
    # 1. Info de compases (time signatures) usando la primera voz
    # ------------------------------------------------------------
    time_sigs: List[Tuple[int, str]] = []

    if len(m21_data.parts) > 0:
        for meas in m21_data.parts[0].getElementsByClass(stream.Measure):
            if meas.timeSignature is not None:
                m_num = (meas.number or 0) + (start_measure - 1)
                time_sigs.append((m_num, meas.timeSignature.ratioString))

    # Ordenar y comprimir cambios de compás (solo para imprimir)
    time_sigs.sort(key=lambda x: x[0])
    if verbose:
        if not time_sigs:
            print("No se encontraron indicaciones de compás.")
        else:
            compressed_ts = [time_sigs[0]]
            for m_num, ts in time_sigs[1:]:
                if ts != compressed_ts[-1][1]:
                    compressed_ts.append((m_num, ts))

            print(f"Compás inicial: {compressed_ts[0][1]}")
            if len(compressed_ts) > 1:
                for m_num, ts in compressed_ts[1:]:
                    print(f"Cambio de compás en el compás {m_num}: {ts}")
            else:
                print("No hay cambios de tipo de compás en la pieza.")

    # Función auxiliar: dado un número de compás, devolver el time sig (persistente)
    def get_timesig_for_measure(m: float) -> str:
        if not time_sigs:
            return "NA"
        current_ts = time_sigs[0][1]
        for m_num, ts in time_sigs:
            if m >= m_num:
                current_ts = ts
            else:
                break
        return current_ts

    # ------------------------------------------------------------
    # 2. Procesar cada voz por separado
    # ------------------------------------------------------------
    all_parts_matrices = []

    for part_idx, part in enumerate(m21_data.parts):
        notes_rows = []

        # Offsets globales de inicio de cada compás en esta voz
        measure_start_offsets = {}
        for meas in part.getElementsByClass(stream.Measure):
            meas_offset_global = float(meas.offset)
            meas_num_shifted = (meas.number or 0) + (start_measure - 1)
            measure_start_offsets[meas_num_shifted] = meas_offset_global

        # Recorremos notas, acordes y silencios
        for n in part.flatten().notesAndRests:
            global_start = float(n.offset)
            duration = float(n.quarterLength)  # usamos quarterLength tal como viene

            try:
                measure_number = (n.measureNumber or 0) + (start_measure - 1)
            except Exception:
                measure_number = 0

            measure_start_global = measure_start_offsets.get(measure_number, 0.0)

            # Onset / offset locales dentro del compás (también tal cual)
            local_start = global_start - measure_start_global
            local_offset = local_start + duration

            # Silencio
            if getattr(n, "isRest", False) or isinstance(n, note.Rest):
                midi_val = -1.0
                notes_rows.append([
                    measure_number,   # 0 measure
                    local_start,      # 1 onset_local
                    local_offset,     # 2 offset_local
                    duration,         # 3 duration
                    midi_val          # 4 midi
                ])

            # Acorde
            elif getattr(n, "isChord", False) or isinstance(n, chord.Chord):
                for p in n.pitches:
                    midi_val = float(p.ps)
                    notes_rows.append([
                        measure_number,
                        local_start,
                        local_offset,
                        duration,
                        midi_val
                    ])

            # Nota simple
            else:
                midi_val = float(n.pitch.ps)
                notes_rows.append([
                    measure_number,
                    local_start,
                    local_offset,
                    duration,
                    midi_val
                ])

        # Si la voz no tiene eventos, matriz vacía
        if not notes_rows:
            all_parts_matrices.append(np.zeros((0, 6), dtype=object))
            continue

        notes_arr = np.array(notes_rows, dtype=float)
        # columnas: 0=measure, 1=onset_local, 2=offset_local,
        #           3=duration, 4=midi

        # Ordenar por compás y onset_local
        order = np.lexsort((notes_arr[:, 1], notes_arr[:, 0]))
        notes_arr = notes_arr[order]

        measure_col      = notes_arr[:, 0]
        onset_local_col  = notes_arr[:, 1]
        offset_local_col = notes_arr[:, 2]
        duration_col     = notes_arr[:, 3]
        midi_col         = notes_arr[:, 4]

        # Columna de time signature por fila (persistente entre cambios)
        ts_col = np.array(
            [get_timesig_for_measure(m) for m in measure_col],
            dtype=object
        )

        # Matriz final por voz (dtype=object porque mezclamos floats y strings)
        numeric = np.column_stack(
            (onset_local_col, offset_local_col,
             duration_col, midi_col, measure_col)
        ).astype(object)

        out_matrix = np.column_stack((numeric, ts_col))

        all_parts_matrices.append(out_matrix)

    return all_parts_matrices
