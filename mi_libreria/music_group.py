from dataclasses import dataclass
from typing import Sequence
import numpy as np
from collections import Counter



@dataclass(frozen=True)
class Transformation:
    """
    Transformación aplicada a una ventana de clases de altura.

    inversion:
        False -> identidad
        True  -> I(p) = -p mod 12

    retrograde:
        False -> orden original
        True  -> orden invertido

    cyclic_shift:
        Desplazamiento hacia la izquierda.
        shift=1 transforma [a, b, c] en [b, c, a].

    transposition:
        T_t(p) = p + t mod 12.
    """
    inversion: bool
    retrograde: bool
    cyclic_shift: int
    transposition: int


def _to_pitch_classes(
    window: Sequence[int]) -> np.ndarray:
    """
    Valida una ventana de alturas y la reduce módulo 12.

    Parameters
    ----------
    window : Sequence[int]
        Ventana de alturas MIDI sin silencios.

    Returns
    -------
    np.ndarray
        Ventana de clases de altura en Z_12.
    """
    x = np.asarray(window)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("La ventana debe ser un arreglo unidimensional no vacío.")

    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("La ventana debe contener valores numéricos.")

    if not np.all(np.isfinite(x)):
        raise ValueError("La ventana no puede contener NaN o infinitos.")

    if not np.all(x == np.floor(x)):
        raise ValueError("Las alturas MIDI deben ser números enteros.")

    x = x.astype(np.int64)

    if np.any((x < 0) | (x > 127)):
        raise ValueError("Las alturas MIDI deben estar en el rango [0, 127].")

    return x % 12


def apply_transformation(
    window: Sequence[int],
    transformation: Transformation) -> tuple[int, ...]:
    """
    Aplica una transformación completa a una ventana melódica.
    """
    pcs = _to_pitch_classes(window)

    if transformation.inversion:
        pcs = (-pcs) % 12

    if transformation.retrograde:
        pcs = pcs[::-1]

    pcs = np.roll(pcs, -transformation.cyclic_shift)
    pcs = (pcs + transformation.transposition) % 12

    return tuple(int(v) for v in pcs)


def orbit_window(
    window: Sequence[int]) -> set[tuple[int, ...]]:
    """
    Construye la órbita completa de una ventana bajo:
    transposición, inversión, retrogradación y desplazamiento cíclico.

    Returns
    -------
    set[tuple[int, ...]]
        Conjunto de representantes distintos de la órbita.
    """
    pcs = _to_pitch_classes(window)
    m = len(pcs)

    orbit = set()
    stab = set()

    for inversion in (False, True):
        for retrograde in (False, True):
            for shift in range(m):
                for transposition in range(12):
                    transformation = Transformation(
                        inversion=inversion,
                        retrograde=retrograde,
                        cyclic_shift=shift,
                        transposition=transposition,
                    )

                    candidate = apply_transformation(
                        pcs,
                        transformation)
                    orbit.add(candidate)

    return orbit


def canonical_window(
    window: Sequence[int],
    *,
    return_transformation: bool = False
):
    """
    Obtiene la representación canónica de una ventana melódica.

    La representación canónica es el mínimo lexicográfico de la órbita
    generada por transposición, inversión, retrogradación y
    desplazamiento cíclico.

    Esta función evita recorrer explícitamente las 12 transposiciones:
    para cada forma temporal/invertida, la transposición lexicográficamente
    mínima es aquella que hace que el primer elemento sea 0.

    Parameters
    ----------
    window : Sequence[int]
        Ventana de alturas MIDI.
    return_transformation : bool
        Si True, también regresa una transformación que produce
        el representante canónico.

    Returns
    -------
    tuple[int, ...]
        Representación canónica.

    o bien

    tuple[tuple[int, ...], Transformation]
        Representación canónica y transformación asociada.
    """
    pcs = _to_pitch_classes(window)
    m = len(pcs)

    best_candidate = None
    best_transformation = None

    for inversion in (False, True):

        if inversion:
            pitch_form = (-pcs) % 12
        else:
            pitch_form = pcs.copy()

        for retrograde in (False, True):

            if retrograde:
                ordered_form = pitch_form[::-1]
            else:
                ordered_form = pitch_form

            for shift in range(m):
                shifted_form = np.roll(ordered_form, -shift)

                # Debido a la equivalencia por transposición,
                # el representante mínimo debe comenzar en 0.
                transposition = int((-shifted_form[0]) % 12)

                candidate_array = (shifted_form + transposition) % 12
                candidate = tuple(int(v) for v in candidate_array)

                transformation = Transformation(
                    inversion=inversion,
                    retrograde=retrograde,
                    cyclic_shift=shift,
                    transposition=transposition,
                )

                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
                    best_transformation = transformation

    if return_transformation:
        return best_candidate, best_transformation

    return best_candidate

def canonical_orbit_sequence(
    melody: Sequence[int],
    m: int,
    *,
    strict_midi: bool = True
) -> list[tuple[int, ...]]:
    """
    Convierte una melodía en una secuencia de representantes canónicos
    obtenidos con ventanas deslizantes de tamaño m.
    """
    pcs = _to_pitch_classes(melody, strict_midi=strict_midi)

    if not isinstance(m, int) or m < 1:
        raise ValueError("m debe ser un entero positivo.")

    if m > len(pcs):
        raise ValueError("m no puede ser mayor que la longitud de la melodía.")

    return [
        canonical_window(pcs[i:i + m], strict_midi=False)
        for i in range(len(pcs) - m + 1)
    ]



# Número total de órbitas posibles para cada tamaño de ventana
K_ORBIT = {
    3: 19,
    4: 163,
    5: 1110,
    6: 10_962,
    7: 107_509,
    8: 1_126_566,
}


@dataclass(frozen=True)
class OrbitEntropyResult:
    """
    Resultados del cálculo de H_orbit.
    """
    m: int
    N: int
    K_m: int
    observed_orbits: int
    H_orbit: float
    H_orbit_normalized: float
    normalization_factor: float
    probabilities: dict[tuple[int, ...], float]
    counts: dict[tuple[int, ...], int]


def orbit_entropy(
    orbit_sequence: Sequence[tuple[int, ...]],
    *,
    return_details: bool = False
):
    """
    Calcula la entropía H_orbit de una secuencia de órbitas canónicas.

    Parameters
    ----------
    orbit_sequence : Sequence[tuple[int, ...]]
        Secuencia de representantes canónicos previamente calculados.
        Cada elemento debe ser una tupla de longitud m.

    return_details : bool, default=False
        Si False, regresa únicamente H_orbit normalizada.
        Si True, regresa un objeto OrbitEntropyResult con información
        adicional.

    Returns
    -------
    float
        Entropía normalizada H_orbit / log(min(N, K(m))),
        si return_details=False.

    OrbitEntropyResult
        Resultados completos, si return_details=True.

    Notes
    -----
    Se utiliza logaritmo natural.

    Si N = 1, la entropía observada es cero y el factor de
    normalización log(min(N, K(m))) también es cero. En ese caso
    se define H_orbit_normalized = 0.0.
    """
    if len(orbit_sequence) == 0:
        raise ValueError("La secuencia de órbitas no puede estar vacía.")

    # Convertir a tuplas para garantizar que sean hashables
    sequence = [tuple(orbit) for orbit in orbit_sequence]

    # Inferir m a partir del primer representante canónico
    m = len(sequence[0])

    if m not in K_ORBIT:
        raise ValueError(
            f"No se dispone de K(m) para m={m}. "
            f"Los valores permitidos son {sorted(K_ORBIT.keys())}."
        )

    if any(len(orbit) != m for orbit in sequence):
        raise ValueError(
            "Todos los representantes canónicos deben tener la misma longitud."
        )

    N = len(sequence)
    K_m = K_ORBIT[m]

    # Conteo empírico de órbitas
    counts = Counter(sequence)

    # Probabilidades empíricas
    probabilities = {
        orbit: count / N
        for orbit, count in counts.items()
    }

    # Entropía de Shannon de la distribución de órbitas
    H_orbit = -sum(
        p * np.log(p)
        for p in probabilities.values()
        if p > 0
    )

    # Factor de normalización corregido por tamaño de muestra
    max_observable_states = min(N, K_m)
    normalization_factor = np.log(max_observable_states)

    if normalization_factor == 0:
        H_orbit_normalized = 0.0
    else:
        H_orbit_normalized = H_orbit / normalization_factor

    if return_details:
        return OrbitEntropyResult(
            m=m,
            N=N,
            K_m=K_m,
            observed_orbits=len(counts),
            H_orbit=float(H_orbit),
            H_orbit_normalized=float(H_orbit_normalized),
            normalization_factor=float(normalization_factor),
            probabilities=dict(probabilities),
            counts=dict(counts),
        )

    return float(H_orbit_normalized)


def H_orbit(melody: Sequence[int], m: int, return_details: bool = False) -> float | OrbitEntropyResult:
    """
    Calcula H_orbit para una melodía dada y tamaño de ventana m.

    Parameters
    ----------
    melody : Sequence[int]
        Melodía representada como una secuencia de alturas MIDI.
    m : int
        Tamaño de la ventana deslizante.

    return_details : bool, default=False
        Si False, regresa únicamente H_orbit normalizada.
        Si True, regresa un objeto OrbitEntropyResult con información
        adicional.

    Returns
    -------
    float
        Entropía H_orbit normalizada.
    OrbitEntropyResult
        Resultados completos, si return_details=True.
    """
    orbit_seq = canonical_orbit_sequence(melody, m)
    return orbit_entropy(orbit_seq, return_details=return_details)