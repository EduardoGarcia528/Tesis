import numpy as np
from itertools import permutations

# ==============================
# Utilidades de patrones ordinales
# ==============================
def all_patterns(m):
    # Cada patrón es una tupla de índices (0..m-1) en orden ascendente de valores
    return [tuple(p) for p in permutations(range(m))]

def relabel(order):
    # Convierte una lista de índices arbitrarios a clases 0..k-1 preservando el orden relativo
    # Ej: order = [2,4,3] -> ranks de 3 elementos -> [0,2,1] después de ordenar y relabel
    inv = {v:i for i, v in enumerate(sorted(order))}
    return tuple(inv[v] for v in order)

def border_first(pattern):
    # Orden relativo de los primeros m-1 elementos (índices 0..m-2)
    keep = [x for x in pattern if x != pattern[-1]]  # quitar el último índice (m-1) NO es correcto aún
    # OJO: pattern son índices 0..m-1; para el "borde" queremos el orden relativo de {0,1,...,m-2}
    # Basta eliminar el elemento 'm-1' si está, y luego relabel a 0..m-2
    keep = [x for x in pattern if x != (len(pattern)-1)]
    return relabel(keep)

def border_last(pattern):
    # Orden relativo de los últimos m-1 elementos (índices 1..m-1) mapeados a 0..m-2
    # Tomamos el patrón y reemplazamos i->(i-1) para i>=1, y eliminamos el 0.
    m = len(pattern)
    shifted = []
    for x in pattern:
        if x == 0:   # el índice 0 (más viejo) no aparece en el siguiente borde
            continue
        shifted.append(x-1)
    return relabel(shifted)

def build_transition_graph(m):
    """
    Un patrón p puede transicionar a q si el orden relativo de los m-1 elementos
    compartidos por ventanas consecutivas coincide:
    border_last(p) == border_first(q)
    """
    pats = all_patterns(m)
    idx = {p:i for i,p in enumerate(pats)}
    adj = [[] for _ in pats]
    for i, p in enumerate(pats):
        bp = border_last(p)
        for j, q in enumerate(pats):
            if border_first(q) == bp:
                adj[i].append(j)
    return pats, idx, adj

# ==============================
# Emisión numérica consistente
# ==============================
def init_window_for_pattern(pattern, margin=1.0):
    """
    Crea una ventana inicial x[0:m] cuyos valores cumplen exactamente el patrón dado.
    Asignamos valores crecientes separados por 'margin' y luego los permutamos
    para que argsort(x) == pattern.
    """
    m = len(pattern)
    base_sorted = np.arange(m, dtype=float) * margin
    x = np.empty(m, dtype=float)
    # pattern = tupla de indices en orden ascendente de valores
    # Si argsort(x) = pattern, entonces x[pattern[k]] = base_sorted[k]
    for rank, idx in enumerate(pattern):
        x[idx] = base_sorted[rank]
    return x

def append_value_for_next_pattern(prev_window, next_pattern, noise=0.0, margin=1e-2):
    """
    Dado prev_window (m-1 + valor nuevo se convertirá en la ventana siguiente),
    elegimos el nuevo valor x_new para que la ventana cumpla 'next_pattern'.
    Requiere que la transición sea compatible (asegurado por el grafo de transiciones).
    """
    m = len(next_pattern)
    # Re-etiquetar los m-1 previos en el marco de la ventana nueva: posiciones 0..m-2
    prev_vals = prev_window[1:]  # los últimos m-1 valores
    # Ordenar esos previos:
    order = np.argsort(prev_vals)  # del menor al mayor
    sorted_prev = prev_vals[order]

    # En next_pattern, el índice m-1 corresponde al NUEVO punto.
    # ¿En qué 'rank' debe caer ese nuevo punto respecto a los previos?
    # Construimos el orden relativo de la ventana new_win = [prev_vals(0..m-2), x_new]
    # next_pattern es una permutación de [0,1,...,m-1]. La posición (rank) donde está (m-1)
    # dice cuántos previos debe dejar por debajo.
    rank_new = next_pattern.index(m-1)  # entre 0 y m-1

    if rank_new == 0:
        # Debe ser menor que todos los previos
        x_new = sorted_prev[0] - margin
    elif rank_new == m-1:
        # Mayor que todos los previos
        x_new = sorted_prev[-1] + margin
    else:
        # Entre los ordenes rank_new-1 y rank_new
        low = sorted_prev[rank_new-1]
        high = sorted_prev[rank_new]
        # Sitúalo en el medio con una pequeña separación
        x_new = (low + high) / 2.0
    if noise > 0:
        x_new += np.random.normal(0.0, noise)
    return x_new

# ==============================
# Simulación Semi-Markov
# ==============================
def simulate_op_smm(
    N=2000, m=4, 
    allowed_states='all',      # 'all' o lista de patrones (tuplas)
    mean_durations=None,       # dict {pattern_index: media geométrica} o escalar o None (fijo=1)
    P=None,                    # matriz de transición entre estados permitidos (después de agotar duración)
    determinist_duration=False,# si True, las duraciones son fijas (round(mean))
    noise=0.0,                 # jitter en la emisión
    seed=None
):
    """
    Devuelve:
      x : serie numérica de longitud N cuyo sliding window (m, tau=1) sigue la secuencia de patrones
      patterns_seq : índices de patrón para cada ventana que inicia en t (longitud N-m+1)
      pats_sub : lista de patrones (tuplas) efectivamente usados como estados
    """
    rng = np.random.default_rng(seed)

    # Construir grafo de compatibilidad
    pats_all, idx_all, adj_all = build_transition_graph(m)

    # Subconjunto de estados a usar
    if allowed_states == 'all':
        chosen = list(range(len(pats_all)))
    else:
        # allowed_states es una lista de tuplas (patrones)
        chosen = [idx_all[p] for p in allowed_states]
    pats_sub = [pats_all[i] for i in chosen]

    # Adyacencias restringidas al subgrafo elegido
    pos_in_sub = {i:k for k,i in enumerate(chosen)}
    adj_sub = [[] for _ in chosen]
    for k,i in enumerate(chosen):
        for j in adj_all[i]:
            if j in pos_in_sub:
                adj_sub[k].append(pos_in_sub[j])

    S = len(chosen)
    if S == 0:
        raise ValueError("No hay estados (patrones) permitidos.")

    # Matriz de transición por defecto: uniforme sobre compatibles (excluido self si la duración terminó)
    if P is None:
        P = np.zeros((S,S), dtype=float)
        for s in range(S):
            outs = adj_sub[s]
            if len(outs) == 0:
                P[s,s] = 1.0
            else:
                # uniforme
                P[s, outs] = 1.0 / len(outs)

    # Duraciones
    def sample_duration(s):
        if mean_durations is None:
            return 1
        if isinstance(mean_durations, (int, float)):
            mu = float(mean_durations)
        else:
            # dict por estado (índice s)
            mu = float(mean_durations.get(s, 1.0))
        if determinist_duration:
            return max(1, int(round(mu)))
        # geométrica con media mu => p = 1/mu
        p = 1.0 / max(mu, 1.0)
        # duración >=1
        return 1 + rng.geometric(p) - 1

    # Elegir estado inicial y duración
    s = 0
    # opcional: empezar en cualquiera
    s = rng.integers(0, S)
    dur = sample_duration(s)

    # Serie y book-keeping
    x = np.empty(N, dtype=float)
    patterns_seq = []

    # Semilla: primera ventana
    x[:m] = init_window_for_pattern(pats_sub[s], margin=1.0)

    t = m
    # Consumimos dur-1 ventanas más (ya usamos una al inicializar)
    remaining = dur - 1

    while t < N:
        # Si aún quedan repeticiones del mismo patrón (self-loop implícito)
        next_s = s
        if remaining <= 0:
            # Elegimos siguiente estado según P
            probs = P[s]
            next_s = rng.choice(S, p=probs)
            dur = sample_duration(next_s)
            remaining = dur

        # Asegurar compatibilidad (por construcción, P y adj_sub ya la garantizan)
        # Emisión: calcular x[t] para cumplir el patrón next_s en la nueva ventana
        prev_window = x[t-m:t].copy()
        x[t] = append_value_for_next_pattern(prev_window, pats_sub[next_s], noise=noise, margin=1e-2)

        # Registrar patrón que empieza en t-m+1
        patterns_seq.append(next_s)

        # Avanzar
        s = next_s
        remaining -= 1
        t += 1

    # Prepend patrón inicial para alinear longitud (opcional)
    patterns_seq = [s] + patterns_seq  # longitud N-m+1
    return x, patterns_seq, pats_sub


import matplotlib.pyplot as plt
import numpy as np
from funciones import permutation_entropy
import scienceplots
from PEs_PDFs import plot_patterns_to_pdf

# 1) Todos los patrones de m=4, duraciones medias = 5 ventanas, poco ruido:
x, seq, pats = simulate_op_smm(N=1000000, m=6, mean_durations=None, noise=0.0, seed=7)
res = plot_patterns_to_pdf(
arr=x, m=6, tau=1,
pdf_path=f"patrones.pdf",
show_top=None,          # o p.ej. 30 para las 30 más frecuentes
sort_by="freq",         # "freq" para ordenar por prob. desc, "code" por índice Lehmer
normalize=True,
title=None,
dpi=200
)
plt.plot(x, marker = '.')
plt.show()

for k in range(3,7):
    PEs = []
    print(k)
    for m in range(3,7):
        # plt.plot(x2, marker = '.')
        # plt.title(f"Simulación OP-SMM (N={len(x2)}, m={m})")
        # plt.show()
        # print(m)
        PE = permutation_entropy(x2, m=m, tau = 1)
        # Supón que ya tienes tu serie 'x' en un np.array
        if k == 3:
            res = plot_patterns_to_pdf(
                arr=x2, m=m, tau=1,
                pdf_path=f"patrones_m5_tau2{m}.pdf",
                show_top=None,          # o p.ej. 30 para las 30 más frecuentes
                sort_by="freq",         # "freq" para ordenar por prob. desc, "code" por índice Lehmer
                normalize=True,
                title=None,
                dpi=200
            )
        PEs.append(PE)
    np.save(f"PEs/PEs_op_smm{k}.npy", np.array(PEs))

plt.style.use(['science', 'notebook', 'grid'])

plt.plot(range(3,7),PEs)
# plt.title(f"Simulación OP-SMM (N={len(x)}, m=4) PE = {permutation_entropy(x2, m =4)}")
plt.show()

