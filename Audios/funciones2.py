import numpy as np
from fractions import Fraction
from typing import Optional

def _fractionize(x: float, max_den: int = 960) -> Fraction:
    return Fraction(x).limit_denominator(max_den)

def _grid_unit_from_bar(durs: np.ndarray,
                        strategy: str = "min",
                        max_den: int = 960,
                        fallback: float = 1/16) -> float:
    if len(durs) == 0:
        return fallback
    if strategy == "gcd":
        fracs = [_fractionize(float(d), max_den) for d in durs]
        den_lcm = 1
        for f in fracs:
            den_lcm = np.lcm(den_lcm, f.denominator)
        ints = [int(f.numerator * (den_lcm // f.denominator)) for f in fracs]
        gcd_int = 0
        for v in ints:
            gcd_int = int(np.gcd(gcd_int, v))
        unit = Fraction(gcd_int, den_lcm)
        val = float(unit) if unit > 0 else float(np.min(durs))
    else:
        val = float(np.min(durs))
    return val if val > 0 else fallback

def _merge_intervals(intervals, tol=1e-12):
    """Une intervalos [a,b) solapados o contiguos (con tolerancia)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + tol:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def _complement_intervals(covered, a0, a1, tol=1e-12):
    """Regresa huecos dentro de [a0,a1) dados intervalos cubiertos (unidos)."""
    if a1 <= a0:
        return []
    gaps = []
    cur = a0
    for a, b in covered:
        if a > cur + tol:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < a1 - tol:
        gaps.append((cur, a1))
    # Limpieza numérica
    gaps2 = []
    for a, b in gaps:
        if b - a > tol:
            gaps2.append((a, b))
    return gaps2

def randomize_rhythm_per_bar_with_rests(
    arr: np.ndarray,
    measure_length: float = 4.0,     # p. ej. 4/4 => 4 negras
    unit_strategy: str = "min",      # "min" (nota más corta) o "gcd"
    max_den: int = 960,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Aleatoriza onsets por compás en una rejilla, permite solapamientos,
    y rellena los espacios vacíos con silencios (MIDI=-1.0) para que
    la cobertura del compás sea exactamente 'measure_length'.

    Entradas:
        arr: Nx5 -> [onset, offset, duration, MIDI, compas] (tiempos en negras)
    Salida:
        Mx5 con notas reubicadas + silencios añadidos (M>=N).
    """
    rng = np.random.default_rng(seed)
    compases = np.unique(arr[:, 4])
    out_rows = []

    for c in compases:
        mask = (arr[:, 4] == c)
        idxs = np.flatnonzero(mask)
        # Orden estable: por onset original y, en empate, por índice
        idxs = idxs[np.lexsort((idxs, arr[idxs, 0]))]

        durs = arr[idxs, 2]
        grid = _grid_unit_from_bar(durs, strategy=unit_strategy, max_den=max_den)
        grid_frac = _fractionize(grid, max_den)
        grid = float(grid_frac)
        if grid <= 0:
            grid = 1/16

        total_slots = int(round(measure_length / grid))
        measure_end = measure_length
        placed_intervals = []  # para calcular huecos (unión)

        # Restricción de orden: la nota i+1 no inicia antes que la i
        prev_start_slot_min = 0

        # 1) Colocar notas (permitiendo solapamientos)
        for k in idxs:
            dur = float(arr[k, 2])
            # número de slots máximo que deja espacio para el offset
            dur_slots = max(1, int(round(_fractionize(dur, max_den) / grid_frac)))
            # Pero la consistencia será exacta: offset = onset + dur (no “redondeamos” duración).
            last_start_slot = total_slots - dur_slots
            if last_start_slot < 0:
                # Nota más larga que el compás: la truncamos al borde para colocar onset=0
                # y recortamos duración efectiva de colocación, pero mantenemos duration original
                # asegurando offset<=final.
                last_start_slot = 0

            # Respetar el orden relativo
            start_min = max(0, prev_start_slot_min)
            start_max = max(0, last_start_slot)
            if start_min > start_max:
                # Si por orden ya no queda ventana, “pegamos” al último inicio posible.
                start_slot = start_max
            else:
                start_slot = int(rng.integers(start_min, start_max + 1))

            onset = start_slot * grid
            offset = onset + dur
            if offset > measure_end:
                # Si por flotantes se pasa, recórtalo al borde exacto
                offset = measure_end
                # (duración almacenada en col 2 sigue siendo la original)
            placed_intervals.append((onset, offset))

            out_rows.append([onset, offset, arr[k, 2], arr[k, 3], c])
            # actualizar restricción de orden (permite empates -> acordes)
            prev_start_slot_min = start_slot

        # 2) Añadir silencios que cubran el complemento de la unión
        merged = _merge_intervals(placed_intervals, tol=1e-12)
        gaps = _complement_intervals(merged, 0.0, measure_end, tol=1e-12)

        for a, b in gaps:
            rest_dur = b - a
            out_rows.append([a, b, rest_dur, -1.0, c])

        # 3) Verificación: la suma de longitudes de (unión de notas + silencios) = measure_length
        # (No ajustamos nada aquí; los gaps ya garantizan la cobertura exacta.)

    out = np.array(out_rows, dtype=float)
    # Orden final: por compás, onset y (para acordes) por MIDI ascendente (silencios quedan donde toca)
    order = np.lexsort((out[:, 3], out[:, 0], out[:, 4]))
    return out[order]


# ---------------- Ejemplo de uso ----------------
if __name__ == "__main__":
    arr = np.array([
        [0.0, 1.0, 1.0, 72.0, 1.0],
        [1.0, 2.0, 1.0, 60.0, 1.0],
        [2.0, 3.0, 1.0, 48.0, 1.0],
        [3.0, 4.0, 1.0, 84.0, 1.0],
        [0.0, 1.0, 1.0, 60.0, 2.0],
        [1.0, 2.0, 1.0, 60.0, 2.0],
        [2.0, 3.0, 1.0, 48.0, 2.0],
        [3.0, 4.0, 1.0, 60.0, 2.0],
        [0.0, 1.0, 1.0, 72.0, 3.0],
        [1.0, 2.0, 1.0, 72.0, 3.0],
        [2.0, 3.0, 1.0, 60.0, 3.0],
        [3.0, 4.0, 1.0, 48.0, 3.0],
        [0.0, 1.0, 1.0, 84.0, 4.0],
        [1.0, 2.0, 1.0, 72.0, 4.0],
        [2.0, 3.0, 1.0, 84.0, 4.0],
        [3.0, 4.0, 1.0, 60.0, 4.0],
    ], dtype=float)

    out = randomize_rhythm_per_bar_with_rests(
        arr, measure_length=4.0, unit_strategy="min", seed=7
    )
    # 'out' contiene notas reubicadas + silencios (MIDI = -1.0)
    print(out)
