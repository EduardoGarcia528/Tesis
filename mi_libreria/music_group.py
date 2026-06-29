from dataclasses import dataclass
from collections import Counter
from numba import njit
import numpy as np
from typing import Sequence


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
        print("Las alturas MIDI deben estar en el rango [0, 127].")
        return np.nan 

    return x % 12


@njit(cache=True)
def _canonical_orbit_sequence_kernel(pcs: np.ndarray, m: int) -> np.ndarray:
    """
    Núcleo compilado con Numba.

    Parameters
    ----------
    pcs : np.ndarray
        Melodía ya reducida módulo 12.
    m : int
        Tamaño de ventana.

    Returns
    -------
    np.ndarray
        Arreglo de forma (N - m + 1, m), donde cada fila es el
        representante canónico de una ventana.
    """
    n_windows = pcs.size - m + 1

    # Las clases de altura están en {0, ..., 11}; int8 es suficiente.
    result = np.empty((n_windows, m), dtype=np.int8)

    candidate = np.empty(m, dtype=np.int8)
    best = np.empty(m, dtype=np.int8)

    for i in range(n_windows):

        first_candidate = True

        # Cuatro formas: identidad, inversión, retrogradación
        # e inversión + retrogradación.
        for inversion in range(2):
            for retrograde in range(2):

                # Todos los desplazamientos cíclicos.
                for shift in range(m):

                    # Índice del primer elemento después del desplazamiento.
                    k0 = shift

                    if retrograde == 1:
                        k0 = m - 1 - k0

                    first_value = pcs[i + k0]

                    if inversion == 1:
                        first_value = (-first_value) % 12

                    # La transposición mínima hace que el primer elemento sea 0.
                    transposition = (-first_value) % 12

                    # Construir candidato.
                    for q in range(m):

                        k = (q + shift) % m

                        if retrograde == 1:
                            k = m - 1 - k

                        value = pcs[i + k]

                        if inversion == 1:
                            value = (-value) % 12

                        candidate[q] = (value + transposition) % 12

                    # Comparación lexicográfica manual.
                    if first_candidate:
                        for q in range(m):
                            best[q] = candidate[q]

                        first_candidate = False

                    else:
                        replace = False

                        for q in range(m):
                            if candidate[q] < best[q]:
                                replace = True
                                break

                            if candidate[q] > best[q]:
                                break

                        if replace:
                            for q in range(m):
                                best[q] = candidate[q]

        for q in range(m):
            result[i, q] = best[q]

    return result


def canonical_orbit_sequence(
    melody: Sequence[int],
    m: int
) -> np.ndarray:
    """
    Convierte una melodía en una secuencia de representantes canónicos
    usando ventanas deslizantes de tamaño m.

    La validación se realiza en Python y el cálculo intensivo se ejecuta
    mediante Numba.

    Returns
    -------
    np.ndarray
        Arreglo de forma (len(melody) - m + 1, m).
        Cada fila es una órbita canónica.
    """
    pcs = _to_pitch_classes(melody)
    if pcs is np.nan:
        return np.nan

    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m debe ser un entero positivo.")

    if m > len(pcs):
        raise ValueError("m no puede ser mayor que la longitud de la melodía.")

    return _canonical_orbit_sequence_kernel(pcs, int(m))



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
    norm: bool = False,
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
    if norm:
        return float(H_orbit_normalized)
    return float(H_orbit)


def H_orbit(melody: Sequence[int], m: int, norm: bool = False, return_details: bool = False) -> float | OrbitEntropyResult:
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
    if orbit_seq is np.nan:
        return np.nan
    return orbit_entropy(orbit_seq,norm = norm ,return_details=return_details)