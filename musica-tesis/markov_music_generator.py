import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
def marginal_probability(melody):
    """
    melody: array-like de valores discretos (ej. notas MIDI)
    """
    melody = np.asarray(melody)
    counts = Counter(melody)
    total = len(melody)

    P = {k: v / total for k, v in counts.items()}
    return P

def pair_probability(melody):
    """
    Calcula P(x_{n+1}, x_n)
    """
    melody = np.asarray(melody)
    pairs = list(zip(melody[:-1], melody[1:]))
    counts = Counter(pairs)
    total = len(pairs)

    P2 = {k: v / total for k, v in counts.items()}
    return P2


def transition_matrix(melody):
    """
    Devuelve:
    - T: matriz de transición
    - states: array con los valores discretos (notas)
    - state_to_idx: diccionario estado -> índice
    """
    melody = np.asarray(melody)
    states = np.unique(melody)
    n = len(states)

    state_to_idx = {s: i for i, s in enumerate(states)}
    T = np.zeros((n, n))

    for a, b in zip(melody[:-1], melody[1:]):
        i = state_to_idx[a]
        j = state_to_idx[b]
        T[i, j] += 1

    # normalización fila por fila
    row_sums = T.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 0:
            T[i] /= row_sums[i]

    return T, states, state_to_idx

def generate_markov_melody(
    melody,
    length,
    start_note=None,
    end_note=None,
    seed=None
):
    """
    length: longitud total de la melodía
    start_note, end_note: valores discretos (ej. MIDI)
    """
    P = marginal_probability(melody)
    P2 = pair_probability(melody)
    T, states, state_to_idx = transition_matrix(melody)

    rng = np.random.default_rng(seed)

    if start_note is None:
        current = rng.choice(states)
    else:
        current = start_note

    melody = [current]

    for _ in range(length - 2):
        i = state_to_idx[current]
        probs = T[i]

        if probs.sum() == 0:
            # estado sin salidas → reinicio aleatorio
            current = rng.choice(states)
        else:
            current = rng.choice(states, p=probs)

        melody.append(current)

    if end_note is not None:
        melody.append(end_note)
    else:
        melody.append(current)

    return np.array(melody)


melody = np.load(r"new_data/1.npy", allow_pickle=True)
melody = np.asarray(melody, dtype=np.float64)
melody = melody[~np.isnan(melody)]

new_melody = generate_markov_melody(
    melody,
    length=len(melody),
    start_note=melody[0],
    end_note=melody[-1],
    seed=42
)

def transition_model_k(melody, k):
    """
    Construye un modelo de Markov de orden k
    Devuelve:
    - model: dict {estado_k: dict {siguiente_valor: prob}}
    - states_k: lista de estados k observados
    """
    melody = np.asarray(melody)

    transitions = defaultdict(Counter)

    for i in range(len(melody) - k):
        state = tuple(melody[i:i+k])
        next_note = melody[i+k]
        transitions[state][next_note] += 1

    # normalización
    model = {}
    for state, counter in transitions.items():
        total = sum(counter.values())
        model[state] = {note: count / total
                        for note, count in counter.items()}

    states_k = list(model.keys())
    return model, states_k

def block_probability(melody, k):
    blocks = [tuple(melody[i:i+k]) for i in range(len(melody) - k + 1)]
    counts = Counter(blocks)
    total = len(blocks)
    return {b: c / total for b, c in counts.items()}

def generate_markov_k(
    melody,
    k,
    length,
    start_state=None, # tupla k
    end_note=None,
    seed=None
):
    """
    model: salida de transition_model_k
    start_state: tupla de longitud k
    """
    model, states_k = transition_model_k(melody, k)
    rng = np.random.default_rng(seed)
    states = list(model.keys())
    k = len(states[0])

    if start_state is None:
        current = states[rng.integers(len(states))]
    else:
        current = tuple(start_state)

    melody = list(current)

    for _ in range(length - k - 1):
        if current not in model:
            # estado no observado → reinicio
            current = states[rng.integers(len(states))]
            melody.extend(list(current))
            continue

        next_notes = list(model[current].keys())
        probs = list(model[current].values())
        next_note = rng.choice(next_notes, p=probs)

        melody.append(next_note)
        current = tuple(melody[-k:])

    if end_note is not None:
        melody.append(end_note)

    return np.array(melody[:length])
