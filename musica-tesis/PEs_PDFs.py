import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit

# ====== TUS UTILIDADES (sin cambios) ======

@njit
def lehmer_code(perm):
    m = len(perm)
    code = 0
    factor = 1
    for i in range(m-1, -1, -1):
        c = 0
        for j in range(i+1, m):
            if perm[j] < perm[i]:
                c += 1
        code += c * factor
        factor *= (m - i)
    return code

@njit
def stable_argsort_by_value_then_index(x):
    m = x.shape[0]
    idx = np.arange(m)
    # insertion sort por clave (valor, índice)
    for i in range(1, m):
        key = idx[i]
        j = i - 1
        while j >= 0:
            a = x[idx[j]]
            b = x[key]
            if (a > b) or (a == b and idx[j] > key):  # (valor) y luego (índice)
                idx[j+1] = idx[j]
                j -= 1
            else:
                break
        idx[j+1] = key
    return idx

# ====== NÚCLEO: histograma de patrones ======

@njit
def ordinal_pattern_counts(arr, m=3, tau=1):
    """
    Devuelve:
      counts: array de tamaño m! con el conteo de cada patrón (indexado por código Lehmer)
      denom: número de ventanas válidas
    """
    n = len(arr)
    if n < m or tau <= 0:
        return np.zeros(1, np.int64), 0

    # m!:
    fact = 1
    for k in range(2, m+1):
        fact *= k

    counts = np.zeros(fact, dtype=np.int64)
    denom = n - (m-1)*tau
    if denom <= 0:
        return counts, 0

    for i in range(denom):
        subseq = np.empty(m, np.float64)
        for j in range(m):
            subseq[j] = arr[i + j*tau]
        idx = stable_argsort_by_value_then_index(subseq)
        code = lehmer_code(idx)
        counts[code] += 1
    return counts, denom

# ====== UTILIDADES fuera de Numba ======

def lehmer_decode(code, m):
    """Invierte el código de Lehmer → permutación (0,1,...,m-1)."""
    # obtener factores factoriales
    factors = [1]*(m)
    for i in range(1, m):
        factors[i] = factors[i-1]*(i+0)  # no se usa, pero dejamos estructura

    # vector factorial base: [ (m-1)!, (m-2)!, ..., 1!, 0! ]
    fact = [math.factorial(k) for k in range(m)]
    # representación factorial mixta
    digits = [0]*m
    x = code
    for i in range(1, m+1):
        digits[-i] = x % i
        x //= i

    # reconstrucción de la permutación
    elems = list(range(m))
    perm = []
    for d in digits:
        perm.append(elems.pop(d))
    return tuple(perm)

def patterns_by_code(m):
    """Lista patterns[code] = perm (tupla), para code=0..m!-1"""
    fact = math.factorial(m)
    return [lehmer_decode(code, m) for code in range(fact)]

def ordinal_pattern_distribution(arr, m=3, tau=1, normalize=True):
    counts, denom = ordinal_pattern_counts(arr, m=m, tau=tau)
    if denom == 0:
        probs = np.zeros_like(counts, dtype=float)
    else:
        probs = counts / float(denom) if normalize else counts.astype(float)
    pats = patterns_by_code(m)
    return counts, probs, pats, denom

# ====== IDENTIFICACIÓN Y REPORTE ======

def describe_patterns(counts, probs, pats, top_k=10):
    idx_desc = np.argsort(-probs)  # descendente por prob
    top = [(pats[i], counts[i], probs[i]) for i in idx_desc[:top_k]]
    bottom = [(pats[i], counts[i], probs[i]) for i in idx_desc[::-1][:top_k]]
    forbidden = [pats[i] for i, c in enumerate(counts) if c == 0]
    return top, bottom, forbidden

# ====== GRÁFICA A PDF ======

def plot_patterns_to_pdf(arr, m=3, tau=1, pdf_path="patrones_ordinales.pdf",
                         show_top=None, sort_by="freq", normalize=True,
                         title=None, dpi=200):
    """
    Crea un PDF con:
      - Barra(s) de distribución de patrones
      - Resumen textual (top, proibidos, etc.)
    Parámetros:
      show_top: None o int → si int, muestra solo las top-K por frecuencia
      sort_by: "freq" (desc) o "code" (por índice de Lehmer)
    """
    counts, probs, pats, denom = ordinal_pattern_distribution(arr, m=m, tau=tau, normalize=True)

    if denom == 0:
        raise ValueError("No hay ventanas válidas (denom <= 0). Revisa n, m y tau.")

    # Orden para graficar
    idx = np.arange(len(counts))
    if sort_by == "freq":
        idx = np.argsort(-probs)
    elif sort_by == "code":
        idx = np.arange(len(counts))

    if show_top is not None:
        idx = idx[:show_top]

    labels = ['-'.join(map(str, pats[i])) for i in idx]
    yvals = probs[idx] if normalize else counts[idx]

    # Texto del título
    if title is None:
        title = f"Distribución de patrones ordinales (m={m}, τ={tau}) — ventanas={denom}"

    # Páginas del PDF
    with PdfPages(pdf_path) as pdf:
        # Página 1: barras
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(idx)), yvals)
        plt.xticks(np.arange(len(idx)), labels, rotation=90)
        plt.ylabel("Probabilidad" if normalize else "Conteo")
        plt.title(title)
        plt.tight_layout()
        pdf.savefig(dpi=dpi)
        plt.close()

        # Página 2: resumen textual
        top, bottom, forbidden = describe_patterns(counts, probs, pats, top_k=min(10, len(counts)))

        # Construimos una página con texto
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle("Resumen de patrones", fontsize=14)

        txt = []
        txt.append(f"Total de patrones posibles: {len(counts)} (m!); ventanas contadas: {denom}")
        txt.append("")
        txt.append("Top frecuentes:")
        for p, c, pr in top:
            txt.append(f"  {p}  | conteo={c:6d}  prob={pr:.6f}")
        txt.append("")
        txt.append("Menos frecuentes:")
        for p, c, pr in bottom:
            txt.append(f"  {p}  | conteo={c:6d}  prob={pr:.6f}")
        txt.append("")
        txt.append(f"Patrones prohibidos ({len(forbidden)}):")
        if len(forbidden) == 0:
            txt.append("  Ninguno")
        else:
            # en bloques para no hacer la línea interminable
            block = []
            for k, perm in enumerate(forbidden, 1):
                block.append(str(perm))
                if k % 8 == 0:
                    txt.append("  " + ", ".join(block))
                    block = []
            if block:
                txt.append("  " + ", ".join(block))

        text = "\n".join(txt)

        # Ponemos el texto ocupando la mayor parte de la página
        fig.text(0.05, 0.05, text, family="monospace", fontsize=10, va="bottom", ha="left")
        pdf.savefig(dpi=dpi)
        plt.close()

    return {
        "pdf_path": pdf_path,
        "counts": counts,
        "probs": probs,
        "patterns": pats,
        "denom": denom
    }

# ====== OPCIONAL: versión de tu entropía que también devuelve counts ======

def permutation_entropy_with_counts(arr, m=3, tau=1):
    counts, denom = ordinal_pattern_counts(arr, m=m, tau=tau)
    fact = len(counts)
    if denom == 0 or fact == 0:
        return np.nan, counts, denom
    probs = counts[counts > 0] / float(denom)
    H = -np.sum(probs * np.log(probs))
    return H / np.log(fact), counts, denom

