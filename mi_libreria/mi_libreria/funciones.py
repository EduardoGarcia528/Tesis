import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
from tqdm import tqdm


### INDICE J

@njit
def distancia(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

@njit
def mejor_vector(p1, p2):
    # Precomputar diferencias en los 9 cuadrantes
    diffs = [
        [p2[0], p2[1]],
        [p2[0], p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1]],
        [p2[0] + 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0], p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1]],
        [p2[0] - 2 * np.pi, p2[1] + 2 * np.pi],
    ]
    # Encontrar el índice con menor distancia
    d_og = distancia(p1,p2)
    min_idx = 0
    for i in range(9):
        d = distancia(p1, diffs[i])
        if d < d_og:
            min_idx = i
            d_og = d
    p2 = diffs[min_idx]
    return [p2[0] - 2*p1[0], p2[1] - 2*p1[1]]


@njit
def calcular_angulos(vectores):
    n = len(vectores) - 1
    angulos = np.empty(n, dtype=np.float64)
    for i in range(n):
        v1 = vectores[i]
        v2 = vectores[i + 1]
        norm_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if norm_v1 == 0 or norm_v2 == 0:
            angulo = 0.0
        else:
            v1n0 = v1[0] / norm_v1
            v1n1 = v1[1] / norm_v1
            v2n0 = v2[0] / norm_v2
            v2n1 = v2[1] / norm_v2
            dot = v1n0 * v2n0 + v1n1 * v2n1
            if dot > 1.0: dot = 1.0
            if dot < -1.0: dot = -1.0
            angulo = np.arccos(dot)
            cruz = v1[0] * v2[1] - v1[1] * v2[0]
            if cruz > 0:
                angulo = np.pi - angulo
            elif cruz == 0 and angulo < 0:
                angulo = np.pi
            elif cruz < 0:
                angulo += np.pi
        angulos[i] = angulo
    return angulos

def caminata_univariante(X, tau, bivariante):
    if bivariante is False:
        x1 = X[tau:]
        y1 = X[:-tau]
    else:
        x1 = X
        y1 = bivariante
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    n = len(ff1) - 1
    vectores = np.empty((n,2),dtype=np.float64) #(n,2)
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def indice_J(seriex, seriey, tau = 1):
    vectores = caminata_univariante(seriex,tau,bivariante=seriey)
    angulos = calcular_angulos(vectores)
    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1)
    return J

def angulos_alpha(seriex, seriey, tau = 1):
    vectores = caminata_univariante(seriex,tau,bivariante=seriey)
    angulos = calcular_angulos(vectores)
    return angulos

# Entropia de Shannon

def entropia_shannon(x, discreto, bins=None):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    if discreto:
        # Caso discreto: cada valor entero o categoría tiene su probabilidad exacta
        valores_unicos, cuentas = np.unique(x, return_counts=True)
        p = cuentas / cuentas.sum()
        if len(p) <= 1:
            return 0.0
        H = -np.sum(p * np.log2(p))
        H_norm = H / np.log2(len(valores_unicos))
    else:
        # Caso continuo: estimar densidad mediante histograma
        try:
            if bins is None:
                bins = 1 + int(np.log2(len(x)))  # Sturges
            hist, _ = np.histogram(x, bins=bins, density=True)
        except Exception:
            print("error: retorna nan")
            return np.nan
        hist = hist[hist > 0]
        if hist.size == 0:
            return np.nan
        p = hist / hist.sum()
        H = -np.sum(p * np.log2(p))
        H_norm = H / np.log2(bins)
    
    return H_norm

#Ruido de Colores 

def colored_noise(N, color="white", fs=1.0, seed=None, normalize=True):
    if seed is not None:
        np.random.seed(seed)

    # Mapeo color → beta
    beta_map = {
        "violet": -2, "violeta": -2,
        "blue": -1, "azul": -1,
        "white": 0, "blanco": 0,
        "pink": 1, "rosa": 1,
        "brown": 2, "cafe": 2, "café": 2
    }

    if color not in beta_map:
        raise ValueError("Color no reconocido")

    beta = beta_map[color]

    # Frecuencias positivas
    freqs = np.fft.rfftfreq(N, d=1/fs)
    freqs[0] = freqs[1]  # evitar división por cero

    # Ruido blanco en frecuencia (fase aleatoria)
    phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))

    # Amplitud espectral ∝ 1/f^{beta/2}
    amplitude = 1.0 / (freqs ** (beta / 2.0))

    spectrum = amplitude * phases

    # Transformada inversa
    x = np.fft.irfft(spectrum, n=N)

    if normalize:
        x = (x - np.mean(x)) / np.std(x)

    return x

def random_array(vocabulario, N, q, m, seed=None):
    """
    Genera una serie de tiempo discreta de longitud N a partir de un vocabulario finito.
    
    Dinámica:
    - Con probabilidad q: repite m veces el último valor agregado (persistencia).
    - Con probabilidad (1-q): elige un valor distinto al último del vocabulario.
    
    Parámetros
    ----------
    vocabulario : array-like
        Conjunto finito de valores posibles (por ejemplo notas MIDI, enteros, etc.)
    N : int
        Longitud deseada de la serie
    q : float
        Probabilidad de repetir el último valor m veces (0 <= q <= 1)
    m : int
        Número de repeticiones cuando ocurre persistencia (m >= 1)
    seed : int o None
        Semilla para reproducibilidad
    
    Retorna
    -------
    np.ndarray
        Serie de tiempo discreta de longitud N
    """
    rng = np.random.default_rng(seed)
    vocabulario = np.array(vocabulario)
    
    if len(vocabulario) == 0:
        raise ValueError("El vocabulario no puede estar vacío.")
    if not (0 <= q <= 1):
        raise ValueError("q debe estar entre 0 y 1.")
    if m < 1:
        raise ValueError("m debe ser >= 1.")
    
    # Inicialización: primer símbolo completamente aleatorio
    serie = [rng.choice(vocabulario)]
    
    while len(serie) < N:
        u = rng.random()
        
        if u < q:
            # Repetir el último valor m veces (truncando si se excede N)
            ultimo = serie[-1]
            repeticiones = min(m, N - len(serie))
            serie.extend([ultimo] * repeticiones)
        else:
            # Elegir un valor distinto al último
            ultimo = serie[-1]
            candidatos = vocabulario[vocabulario != ultimo]
            
            # Si el vocabulario tiene un solo elemento, necesariamente se repite
            if len(candidatos) == 0:
                serie.append(ultimo)
            else:
                nuevo = rng.choice(candidatos)
                serie.append(nuevo)
    
    return np.array(serie)



