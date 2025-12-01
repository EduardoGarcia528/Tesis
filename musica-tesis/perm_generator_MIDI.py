import numpy as np
from itertools import permutations

# ============================================
# 1. Patrones ordinales (convención RANGOS)
# ============================================

def ordinal_pattern_ranks(window):
    """
    Devuelve el patrón ordinal como VECTOR DE RANGOS:
    pattern[i] = rango (0..m-1) del elemento en la posición i.
    Desempate: si hay empates en el valor, gana el índice más viejo.
    """
    window = np.asarray(window)
    m = len(window)
    idx = np.arange(m)
    # Ordenar por valor (primario) y por índice (secundario)
    order = np.lexsort((idx, window))  # p.ej. (2,0,1)

    ranks = np.empty(m, dtype=int)
    for r, i in enumerate(order):
        ranks[i] = r  # el elemento en la posición i tiene rango r

    return tuple(ranks)


def inverse_perm(p):
    """
    p: tupla con una permutación de 0..m-1 interpretada como
       ORDEN DE ÍNDICES (índices en orden creciente de valor).
    Devuelve q (perm inversa) tal que:
       q[i] = rango del índice i  (convención DE RANGOS).
    """
    m = len(p)
    q = [None] * m
    for rank, idx in enumerate(p):
        q[idx] = rank
    return tuple(q)

# ============================================
# 2. Bordes y grafo en espacio de patrones
#    (construidos en convención ÍNDICES y
#     convertidos luego a RANGOS)
# ============================================

def relabel(order):
    """
    Convierte una lista de etiquetas arbitrarias a 0..k-1 preservando orden.
    Ej: order = [2,4,3] -> [0,2,1]
    """
    inv = {v: i for i, v in enumerate(sorted(order))}
    return tuple(inv[v] for v in order)


def border_first_index(pattern):
    """
    'Borde izquierdo' para un patrón dado como ORDEN DE ÍNDICES:
    - pattern es una permutación de 0..m-1; su longitud es m.
    - quitamos el índice m-1 (correspondiente al punto nuevo de la ventana anterior)
      y relabelamos a 0..m-2.
    """
    m = len(pattern)
    keep = [x for x in pattern if x != (m - 1)]
    return relabel(keep)


def border_last_index(pattern):
    """
    'Borde derecho' para un patrón dado como ORDEN DE ÍNDICES:
    - los índices 1..m-1 se corresponden con las posiciones 0..m-2 de la ventana siguiente;
      quitamos el 0 y hacemos x -> x-1, luego relabelamos.
    """
    shifted = []
    for x in pattern:
        if x == 0:
            continue
        shifted.append(x - 1)
    return relabel(shifted)


def build_transition_graph_rank(m):
    """
    Construye el grafo de compatibilidad entre patrones ordinales de longitud m.

    Internamente:
      - Usa la convención 'orden de índices' para construir el grafo.
    Hacia fuera:
      - Devuelve los patrones en la convención DE RANGOS:
            pattern[i] = rango del elemento en la posición i.

    Devuelve:
      pats_rank : lista de patrones (tuplas) en convención rangos
      idx_rank  : diccionario {pattern_rangos -> índice}
      adj       : lista de listas; adj[s] = lista de índices de estados compatibles
    """
    # 1) Patrones en convención "orden de índices"
    pats_index = [tuple(p) for p in permutations(range(m))]
    idx_index = {p: i for i, p in enumerate(pats_index)}
    adj_index = [[] for _ in pats_index]

    # 2) Construir adyacencias usando bordes en convención índice
    for i, p in enumerate(pats_index):
        bp = border_last_index(p)
        for j, q in enumerate(pats_index):
            if border_first_index(q) == bp:
                adj_index[i].append(j)

    # 3) Convertir etiquetas a convención de rangos
    pats_rank = [inverse_perm(p) for p in pats_index]
    idx_rank = {pats_rank[i]: i for i in range(len(pats_rank))}

    # La numeración de estados es la misma; adj_index sirve también para rangos
    return pats_rank, idx_rank, adj_index

# ============================================
# 3. Ventana inicial discreta (notas MIDI)
# ============================================

def init_window_for_pattern_discrete(pattern, vocab, rng=None, allow_repeats=False):
    """
    Crea una ventana inicial x[0:m] con valores en 'vocab' que cumplen
    el patrón de RANGOS dado:
        pattern[i] = rango del elemento en posición i.

    - vocab: iterable de notas (enteros MIDI, por ejemplo).
    - allow_repeats: si False, intenta usar notas distintas si hay suficientes.
    """
    pattern = tuple(pattern)
    m = len(pattern)
    vocab = np.array(vocab)

    if rng is None:
        rng = np.random.default_rng()

    if (not allow_repeats) and len(vocab) >= m:
        notes = rng.choice(vocab, size=m, replace=False)
    else:
        notes = rng.choice(vocab, size=m, replace=True)

    notes = np.sort(notes)  # notas en orden creciente

    x = np.empty(m, dtype=int)
    # notes[0] corresponde al rango 0, notes[1] al rango 1, etc.
    for i in range(m):
        r = pattern[i]      # rango deseado en la posición i
        x[i] = notes[r]

    # Chequeo de seguridad
    if ordinal_pattern_ranks(x) != pattern:
        raise RuntimeError(
            f"init_window_for_pattern_discrete generó ventana incompatible: "
            f"x={x}, patrón esperado={pattern}, patrón obtenido={ordinal_pattern_ranks(x)}"
        )

    return x

# ============================================
# 4. Emisión discreta (empates como ÚLTIMO recurso)
# ============================================

def append_discrete_value_for_next_pattern(prev_window, next_pattern, vocab, rng=None):
    """
    Elige x_new de 'vocab' tal que la ventana nueva
        [prev_window[1:], x_new]
    tenga patrón de RANGOS igual a 'next_pattern'.

    Estrategia:
      1) Intentar primero valores que NO repitan ninguna nota de prev_window[1:].
      2) Si no se puede, permitir empates (repetir notas).
      3) Si aun así no hay ningún valor que cumpla el patrón, lanza ValueError.

    'next_pattern' está en convención DE RANGOS.
    """
    prev_window = np.asarray(prev_window)
    m = len(next_pattern)
    prev_vals = prev_window[1:]  # últimos m-1 valores
    assert len(prev_vals) == m - 1

    if rng is None:
        rng = np.random.default_rng()

    candidates = np.array(vocab, copy=True)
    rng.shuffle(candidates)

    # 1) Sin empates: candidatos que no estén en prev_vals
    no_repeat = [v for v in candidates if v not in prev_vals]
    for v in no_repeat:
        win = np.empty(m, dtype=int)
        win[:-1] = prev_vals
        win[-1] = v
        if ordinal_pattern_ranks(win) == next_pattern:
            # print(next_pattern)
            # print(v)
            return v

    # 2) Con empates permitidos
    for v in candidates:
        win = np.empty(m, dtype=int)
        win[:-1] = prev_vals
        win[-1] = v
        if ordinal_pattern_ranks(win) == next_pattern:
            # print(next_pattern)
            # print(v)
            return v

    # 3) No existe solución (incompatibilidad geométrica)
    raise ValueError(
        f"No existe valor en 'vocab' que realice el patrón {next_pattern} "
        f"para prev_window={prev_window} (prev_vals={prev_vals})."
    )

# ============================================
# 5. Simulación Semi-Markov discreta (notas MIDI)
# ============================================

def simulate_op_smm_discrete(
    N=2000,
    m=4,
    vocab=None,              # iterable de valores discretos (ej. notas MIDI)
    allowed_states='all',    # 'all' o lista de patrones (en convención rangos)
    P=None,                  # matriz de transición entre estados permitidos
    seed=None,
):
    """
    Simula un modelo semi-Markov sobre patrones ordinales de tamaño m
    (convención DE RANGOS), generando una serie de longitud N
    con valores discretos (p. ej. notas MIDI).

    NUEVO: si en algún paso no existe valor en 'vocab' que realice el patrón
    next_pattern dado el prev_window, se hace un RESET a la ventana inicial
    (los primeros m valores de la serie) y al estado inicial.
    """
    if vocab is None:
        raise ValueError("Debes proporcionar un vocabulario de notas discretas 'vocab'.")

    rng = np.random.default_rng(seed)
    # Grafo de compatibilidad en espacio de patrones (convención rangos)
    pats_all, idx_all, adj_all = build_transition_graph_rank(m)

    # Subconjunto de estados a usar
    if allowed_states == 'all':
        chosen = list(range(len(pats_all)))
    else:
        # allowed_states es una lista de patrones (en convención rangos)
        chosen = [idx_all[p] for p in allowed_states]
    pats_sub = [pats_all[i] for i in chosen]

    # Adyacencias restringidas al subgrafo elegido
    pos_in_sub = {i: k for k, i in enumerate(chosen)}
    adj_sub = [[] for _ in chosen]
    for k, i in enumerate(chosen):
        for j in adj_all[i]:
            if j in pos_in_sub:
                adj_sub[k].append(pos_in_sub[j])

    S = len(chosen)
    if S == 0:
        raise ValueError("No hay estados (patrones) permitidos.")

    # Matriz de transición P por defecto: uniforme sobre vecinos compatibles
    if P is None:
        P = np.zeros((S, S), dtype=float)
        for s in range(S):
            outs = adj_sub[s]
            if len(outs) == 0:
                P[s, s] = 1.0
            else:
                P[s, outs] = 1.0 / len(outs)

    # Muestreo de duraciones

    # Estado inicial y duración
    s0 = rng.integers(0, S)
    s = s0

    # Serie y book-keeping
    x = np.empty(N, dtype=int)
    x[:m] = init_window_for_pattern_discrete(pats_sub[s], vocab=vocab, rng=rng)

    patterns_seq = [s]  # estado de la ventana que empieza en t=0

    t = m

    while t < N:
        # Proponemos siguiente estado (como antes)
        probs = P[s]
        next_s = rng.choice(S, p=probs)

        prev_window = x[t-m:t].copy()

        try:
            # Intento normal: usar prev_window y next_s
            x_new = append_discrete_value_for_next_pattern(
                prev_window, pats_sub[next_s], vocab, rng=rng
            )
            x[t] = x_new
            patterns_seq.append(next_s)
            s = next_s
            t += 1

        except ValueError:
            # ========= RESET =========
            # No hay ningún valor en vocab que realice el patrón next_s
            # dadas las condiciones actuales. Reseteamos a la ventana y
            # estado iniciales.
            # Sobrescribimos la ventana actual con la ventana inicial:
            while True:
                s0 = next_s
                initial_window = init_window_for_pattern_discrete(pats_sub[s0], vocab=vocab, rng=rng)
                initial_state = s0
                x[t-m:t] = initial_window
                # Recalculamos prev_window (ahora es la ventana inicial)
                probs = P[initial_state]
                next_s = rng.choice(S, p=probs)
                prev_window = initial_window.copy()



                # Intentamos emitir acorde al estado inicial.
                # Si incluso esto falla, dejamos que explote (eso ya es un problema serio).
                try:
                    x_new = append_discrete_value_for_next_pattern(
                        prev_window, pats_sub[next_s], vocab, rng=rng
                    )
                    x[t] = x_new
                    patterns_seq.append(s)
                    s = next_s
                    t += 1
                    break
                except:
                    continue


    # patterns_seq tiene longitud N-m+1, alineado con ventanas x[t:t+m]
    return x, patterns_seq, pats_sub


# ============================================
# 6. Ejemplo de uso con notas MIDI
# ============================================
from funciones import permutation_entropy
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # Escala de Do mayor (C mayor) en una octava
    # f = np.array([57,58, 59, 60,61, 62,63, 64, 65,66, 67]) # De La a sol, do mayor
    f = np.array([57, 59, 60, 62, 64, 65, 67]) # De La a sol, do mayor
    vocab_do_mayor = np.concatenate((f-12, f, f +12, f+12*2, f+12*3))

    x_melodia, seq, pats = simulate_op_smm_discrete(
        N=10_000,
        m=4,
        vocab=vocab_do_mayor,
        allowed_states='all',
        seed=None,
    )
    plt.plot(x_melodia, marker='.')
    plt.show()

    PEs = []
    for m in range(3,7):
        PE = permutation_entropy(x_melodia, m=m, tau=1)
        PEs.append(PE)
    plt.plot(range(3,7),PEs)
    plt.xlabel('m')
    plt.ylim(0,1)
    plt.ylabel('PE')
    plt.show()
