import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit

def interpolador(subject, method, size):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    data = subject
    x = np.arange(len(data))
    
    # Crear 'size' puntos equidistantes
    x_new = np.linspace(0, len(data) - 1, size*(len(data)-1) + len(data))
    
    if method == 'lineal':
        data_interp = np.interp(x_new, x, data)
    elif method == 'herm':
        interpolator = PchipInterpolator(x, data)
        data_interp = interpolator(x_new)
    
    return data_interp

def interpolador_constante(subject):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    x = np.arange(2)
    data_interp = np.array([])
    diferencias =np.abs(np.diff(subject))
    if (np.max(diferencias) - np.min(diferencias)) <= 1e-5:
        return subject
    diferencias_sin_0 = diferencias[diferencias != 0]
    diferencias = np.round(diferencias/np.min(diferencias_sin_0))
    for i in range(len(subject)-1):
        data = subject[i:i+2]
        if diferencias[i] == 0:
            data_interp = np.concatenate((data_interp,data[:-1]))
            continue
        size = int(diferencias[i])- 1
        # Crear 'size' puntos equidistantes
        x_new = np.linspace(0, 1, size + 2)
        interp_step = np.interp(x_new, x, data)
        data_interp = np.concatenate((data_interp,interp_step[:-1]))
    data_interp = np.concatenate((data_interp,np.array([data[-1]])))

    return data_interp


def brownian_bridge(t0, tT, x0, xT, n_steps):
    t = np.linspace(t0, tT, n_steps)
    #Generar movimiento browniano con media cero
    W = np.random.normal(0, np.sqrt(t[1] - t[0]), size=n_steps-1)
    W = np.insert(np.cumsum(W), 0, 0)  # Inserta W(0) = 0 y suma acumulativa
    # Interpolación del Brownian Bridge
    X = x0 + (t - t0) / (tT - t0) * (xT - x0) + W - (t - t0) / (tT - t0) * W[-1]
    return t, X

def interpolador_estocastico(s_0_discreto, n_steps):
    t_list, X_list = [], []
    for i in range(len(s_0_discreto)- 1):
        t0 = i
        tT = i+1
        x0 = s_0_discreto[i]
        xT = s_0_discreto[i+1]
        t, X = brownian_bridge(t0, tT, x0, xT, n_steps)
        t_list = np.concatenate((t_list, t))
        X_list = np.concatenate((X_list, X))
    return X_list

def remove_consecutive_duplicates(data, tolerance=1e-1):
    result = [data[0]]  # Comenzamos con el primer elemento
    for i in range(1, len(data)):
        if abs(data[i] - data[i - 1]) > tolerance:
            result.append(data[i])

    return np.array(result)


# PE

@njit
def lehmer_code(perm):
    """Codifica una permutación en un índice único usando Lehmer code"""
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

@njit
def permutation_entropy(arr, m=3, tau=1):
    n = len(arr)
    if n < m:
        return np.nan
    # m!:
    fact = 1
    for k in range(2, m+1):
        fact *= k
    counts = np.zeros(fact, dtype=np.int64)
    denom = n - (m-1)*tau
    for i in range(denom):
        subseq = np.empty(m, np.float64)
        for j in range(m):
            subseq[j] = arr[i + j*tau]
        idx = stable_argsort_by_value_then_index(subseq)
        code = lehmer_code(idx)      # tu misma función
        counts[code] += 1
    # entropía normalizada (independiente de base)
    probs = counts[counts > 0] / denom
    n_prohibidos = fact - len(probs)
    H = -np.sum(probs * np.log(probs))
    Hnorm = H / np.log(fact)
    return Hnorm


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
    angulos = np.empty(n)
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
    vectores = np.empty((n,2)) #(n,2)
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
    J = 1.0 - np.abs(e1.real)
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
