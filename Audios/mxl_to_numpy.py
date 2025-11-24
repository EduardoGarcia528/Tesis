import numpy as np
from music21 import stream, note, chord, converter

def unique_rows_ordered(a):
    # 1. Eliminar filas donde col0 == col1
    mask = a[:, 0] != a[:, 1]
    a = a[mask]

    b = np.ascontiguousarray(a)
    dtype = np.dtype((np.void, b.dtype.itemsize * b.shape[1]))
    _, idx = np.unique(b.view(dtype), return_index=True)
    return a[np.sort(idx)]

def process_notes_by_voice(mxl_file, start_measure=1, verbose=True):
    """
    Procesa una partitura de music21 y devuelve una lista de matrices,
    una por cada voz (part) del .mxl.

    Incluye notas, acordes y silencios (rests).

    Columnas de cada matriz (por voz):
        0: onset_local   (origen 0.0 en cada compás)
        1: offset_local  (onset_local + duration)
        2: interonset_interval_global (diferencia de onsets globales consecutivos)
        3: duration      (en negras)
        4: midi value    (nota: pitch.ps, silencio: -1.0)
        5: #measure      (numeración ajustada con start_measure)
    """
    m21_data = converter.parse(mxl_file)
    # ------------------------------------------------------------
    # 1. Info de compases (time signatures) usando la primera voz
    # ------------------------------------------------------------
    time_sigs = []

    if len(m21_data.parts) > 0:
        for meas in m21_data.parts[0].getElementsByClass(stream.Measure):
            if meas.timeSignature is not None:
                m_num = (meas.number or 0) + (start_measure - 1)
                time_sigs.append((m_num, meas.timeSignature.ratioString))

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
            duration = float(n.quarterLength)

            try:
                measure_number = (n.measureNumber or 0) + (start_measure - 1)
            except Exception:
                measure_number = 0

            measure_start_global = measure_start_offsets.get(measure_number, 0.0)

            # Onset / offset locales dentro del compás
            local_start = global_start - measure_start_global
            local_offset = local_start + duration

            # Silencio
            if getattr(n, "isRest", False) or isinstance(n, note.Rest):
                midi_val = -1.0
                notes_rows.append([
                    measure_number,   # 0
                    global_start,     # 1 (onset global)
                    local_start,      # 2 (onset local)
                    local_offset,     # 3 (offset local)
                    duration,         # 4
                    midi_val          # 5
                ])

            # Acorde
            elif getattr(n, "isChord", False) or isinstance(n, chord.Chord):
                for p in n.pitches:
                    midi_val = float(p.ps)
                    notes_rows.append([
                        measure_number,
                        global_start,
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
                    global_start,
                    local_start,
                    local_offset,
                    duration,
                    midi_val
                ])

        # Si la voz no tiene eventos, devolvemos matriz vacía
        if not notes_rows:
            all_parts_matrices.append(np.zeros((0, 6), dtype=float))
            continue

        notes_arr = np.array(notes_rows, dtype=float)
        # columnas: 0=measure, 1=onset_global, 2=onset_local, 3=offset_local,
        #           4=duration, 5=midi

        # Ordenar por compás y onset_global
        order = np.lexsort((notes_arr[:, 1], notes_arr[:, 0]))
        notes_arr = notes_arr[order]

        # --------------------------------------------------------
        # 3. Interonset interval (IOI) global
        # --------------------------------------------------------
        global_onsets = notes_arr[:, 1]
        ioi_global = np.zeros(len(global_onsets))
        if len(global_onsets) > 1:
            ioi_global[1:] = np.diff(global_onsets)

        # --------------------------------------------------------
        # 4. Matriz final: onset_local, offset_local, IOI_global,
        #                  duration, midi, measure
        # --------------------------------------------------------
        onset_local_col  = notes_arr[:, 2]
        offset_local_col = notes_arr[:, 3]
        duration_col     = notes_arr[:, 4]
        midi_col         = notes_arr[:, 5]
        measure_col      = notes_arr[:, 0]

        out_matrix = np.column_stack(
            (onset_local_col, offset_local_col,
             duration_col, midi_col, measure_col)
        )


        all_parts_matrices.append(unique_rows_ordered(out_matrix))

    return all_parts_matrices

# array_complete = process_notes_by_voice(r"data/chopin-ballade-no-1-in-g-minor-op-23.mxl", start_measure=0)
# np.set_printoptions(suppress=True)
# # print(array_complete[0][2047:2050,:])  
# print(array_complete[0][np.where(array_complete[0][:,-1] == 8.),:])  
