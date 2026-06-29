import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
from tqdm import tqdm


### INDICE J

# ============================================================
# 1. Envolver diferencias angulares al intervalo [-pi, pi)
# ============================================================

@njit
def wrap_pi(a):
    """
    Lleva una diferencia angular al intervalo [-pi, pi).
    
    Esto representa la geodésica corta en una coordenada angular.
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# 2. Construir puntos en el toro: P_i = (f1[i], f2[i])
# ============================================================

def obtener_fases_fourier(seriex, seriey=None, tau=1):
    """
    Construye las dos secuencias de fases f1 y f2.

    Caso univariante:
        f1 = fases de X[tau:]
        f2 = fases de X[:-tau]

    Caso bivariante:
        f1 = fases de X
        f2 = fases de Y
    """

    seriex = np.asarray(seriex, dtype=np.float64)

    if seriey is None:
        if tau <= 0:
            raise ValueError("tau debe ser mayor que cero.")

        if tau >= len(seriex):
            raise ValueError("tau debe ser menor que la longitud de la serie.")

        x1 = seriex[tau:]
        y1 = seriex[:-tau]

    else:
        seriey = np.asarray(seriey, dtype=np.float64)

        if len(seriex) != len(seriey):
            raise ValueError("En el caso bivariante, seriex y seriey deben tener la misma longitud.")

        x1 = seriex
        y1 = seriey

    f1 = np.angle(np.fft.rfft(x1))
    f2 = np.angle(np.fft.rfft(y1))

    # Quitar frecuencia cero, cuya fase usualmente no es informativa.
    f1 = f1[1:]
    f2 = f2[1:]

    # Para señales reales de longitud par, la frecuencia de Nyquist también es real.
    if len(x1) % 2 == 0:
        f1 = f1[:-1]
        f2 = f2[:-1]

    return f1, f2


def construir_puntos_toro(f1, f2):
    """
    Construye los puntos:
    
        P_i = (f1[i], f2[i])
    """

    if len(f1) != len(f2):
        raise ValueError("f1 y f2 deben tener la misma longitud.")

    puntos = np.column_stack((f1, f2))

    return puntos


# ============================================================
# 3. Construir vectores geodésicos: v_i = P_i - P_{i-1}
# ============================================================

@njit
def construir_vectores_geodesicos(puntos):
    """
    Construye los vectores geodésicos entre puntos consecutivos.

    puntos[i] representa:
        P_i = (f1[i], f2[i])

    vectores[i] representa:
        v_{i+1} = P_{i+1} - P_i

    usando la diferencia angular mínima en cada coordenada.
    """

    n_puntos = len(puntos)

    if n_puntos < 2:
        return np.empty((0, 2), dtype=np.float64)

    n_vectores = n_puntos - 1
    vectores = np.empty((n_vectores, 2), dtype=np.float64)

    for i in range(n_vectores):
        dx = wrap_pi(puntos[i + 1, 0] - puntos[i, 0]) - puntos[i, 0] 
        dy = wrap_pi(puntos[i + 1, 1] - puntos[i, 1]) - puntos[i, 1]

        vectores[i, 0] = dx
        vectores[i, 1] = dy

    return vectores


# ============================================================
# 4. Calcular ángulos entre vectores consecutivos
# ============================================================

@njit
def calcular_angulos_entre_vectores(vectores):
    """
    Calcula los ángulos entre vectores consecutivos.

    Si:
        vectores[i]     = v_i
        vectores[i + 1] = v_{i+1}

    entonces calcula:

        alpha_i = angle(v_i, v_{i+1})

    usando:

        alpha_i = atan2(v_i x v_{i+1}, v_i · v_{i+1})

    El resultado se devuelve en [0, 2pi).
    """

    n_vectores = len(vectores)

    if n_vectores < 2:
        return np.empty(0, dtype=np.float64)
    temp = np.empty(n_vectores - 1, dtype=np.float64)
    count = 0

    for i in range(n_vectores - 1):
        v1x = vectores[i, 0]
        v1y = vectores[i, 1]

        v2x = vectores[i + 1, 0]
        v2y = vectores[i + 1, 1]

        norm1 = np.sqrt(v1x*v1x + v1y*v1y)
        norm2 = np.sqrt(v2x*v2x + v2y*v2y)

        # Si un vector tiene norma cero, el ángulo no está definido.
        # En lugar de asignar alpha = 0, lo excluimos.
        if norm1 > 0.0 and norm2 > 0.0:
            dot = v1x*v2x + v1y*v2y
            cross = v1x*v2y - v1y*v2x

            alpha = np.arctan2(cross, dot)

            if alpha < 0.0:
                alpha += 2.0 * np.pi

            temp[count] = alpha
            count += 1

    return temp[:count]


# ============================================================
# 5. Calcular índice J
# ============================================================

def indice_J(seriex, seriey=None, tau=1):
    # 1. Fases de Fourier
    f1, f2 = obtener_fases_fourier(
        seriex,
        seriey=seriey,
        tau=tau
    )

    # 2. Puntos en el toro
    puntos = construir_puntos_toro(f1, f2)

    # 3. Vectores geodésicos
    vectores = construir_vectores_geodesicos(puntos)

    # 4. Ángulos entre vectores consecutivos
    angulos = calcular_angulos_entre_vectores(vectores)

    if len(angulos) == 0:
        J = np.nan
    else:
        e = np.exp(1j * angulos)
        J = 1.0 - np.abs(np.mean(e))
    return J

def angulos_alpha(seriex, seriey, tau = 1):
    f1, f2 = obtener_fases_fourier(
        seriex,
        seriey=seriey,
        tau=tau,
    )
    puntos = construir_puntos_toro(f1, f2)
    vectores = construir_vectores_geodesicos(puntos)
    angulos = calcular_angulos_entre_vectores(vectores)
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



import numpy as np

def autocorrelacion_tau(serie, tau=1, n_shuffles=10, seed=None):
    """
    Calcula la autocorrelación temporal con retraso tau:

        C(tau) = corr(x[tau:], x[:-tau])

    También calcula el mismo índice para versiones random shuffle
    de la serie original.

    Parámetros
    ----------
    serie : array-like
        Serie temporal unidimensional.
    tau : int
        Retraso temporal.
    n_shuffles : int
        Número de realizaciones shuffle.
    seed : int o None
        Semilla aleatoria.

    Regresa
    -------
    resultado : dict
        {
            "tau": tau,
            "autocorr": autocorr_original,
            "autocorr_shuffle": array con valores shuffle,
            "media_shuffle": promedio del shuffle,
            "std_shuffle": desviación estándar del shuffle,
            "z_score": z-score respecto al shuffle
        }
    """

    x = np.asarray(serie, dtype=float)

    if x.ndim != 1:
        raise ValueError("La serie debe ser unidimensional.")

    if tau <= 0:
        raise ValueError("tau debe ser un entero positivo.")

    if tau >= len(x):
        raise ValueError("tau debe ser menor que la longitud de la serie.")

    rng = np.random.default_rng(seed)

    def corr_lag(y, tau):
        y1 = y[tau:]
        y2 = y[:-tau]

        # eliminar NaN si existen
        mask = np.isfinite(y1) & np.isfinite(y2)
        y1 = y1[mask]
        y2 = y2[mask]

        if len(y1) < 2:
            return np.nan

        std1 = np.std(y1)
        std2 = np.std(y2)

        if std1 == 0 or std2 == 0:
            return np.nan

        return np.corrcoef(y1, y2)[0, 1]

    autocorr_original = corr_lag(x, tau)

    autocorr_shuffle = np.empty(n_shuffles)

    for i in range(n_shuffles):
        xs = rng.permutation(x)
        autocorr_shuffle[i] = corr_lag(xs, tau)

    media_shuffle = np.nanmean(autocorr_shuffle)
    std_shuffle = np.nanstd(autocorr_shuffle, ddof=1) if n_shuffles > 1 else np.nan

    if n_shuffles > 1 and std_shuffle > 0:
        z_score = (autocorr_original - media_shuffle) / std_shuffle
    else:
        z_score = np.nan

    return autocorr_original, media_shuffle

import matplotlib.pyplot as plt
from scipy.signal import welch


def graficar_espectro_beta(
    x,
    fs=1.0,
    nperseg=None,
    fit_range=None,
    detrend=True,
    window="hann",
    ax=None,
    label=None,
    plot_fit=True
):
    """
    Grafica el espectro de potencias de una serie y estima el exponente beta
    de una ley tipo P(f) ~ f^(-beta).

    Parámetros
    ----------
    x : array-like
        Serie temporal.
    fs : float, opcional
        Frecuencia de muestreo. Si la serie está indexada por pasos discretos,
        usar fs=1.
    nperseg : int or None
        Tamaño de ventana para Welch. Si None, se usa min(256, len(x)).
    fit_range : tuple or None
        Rango de frecuencias para estimar beta: (f_min, f_max).
        Si None, usa todo el rango válido excepto f=0.
    detrend : bool
        Si True, resta la media antes de calcular el espectro.
    window : str
        Ventana usada en Welch.
    ax : matplotlib axis or None
        Eje donde graficar. Si None, crea uno nuevo.
    label : str or None
        Etiqueta para la serie.
    plot_fit : bool
        Si True, grafica la recta ajustada.

    Regresa
    -------
    beta : float
        Exponente estimado.
    resultado : dict
        Diccionario con frecuencias, PSD, máscara del ajuste, pendiente,
        intercepto y objeto de figura/eje.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x debe ser una serie unidimensional.")

    x = x[np.isfinite(x)]

    if len(x) < 4:
        raise ValueError("La serie es demasiado corta.")

    if detrend:
        x = x - np.mean(x)

    if nperseg is None:
        nperseg = min(256, len(x))

    # Espectro de potencias con Welch
    f, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        detrend=False,
        scaling="density"
    )

    # Quitar frecuencia cero y potencias no positivas
    mask = (f > 0) & (Pxx > 0)

    if fit_range is not None:
        f_min, f_max = fit_range
        mask &= (f >= f_min) & (f <= f_max)

    if np.sum(mask) < 2:
        raise ValueError("No hay suficientes puntos válidos para ajustar beta.")

    log_f = np.log10(f[mask])
    log_P = np.log10(Pxx[mask])

    # Ajuste lineal: log P = slope * log f + intercept
    slope, intercept = np.polyfit(log_f, log_P, 1)

    # Como P(f) ~ f^(-beta), entonces beta = -slope
    beta = -slope

    # Crear gráfica
    if plot_fit:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        ax.loglog(f[1:], Pxx[1:], marker="o", linestyle="-", label=label)

        f_fit = f[mask]
        P_fit = 10**intercept * f_fit**slope
        ax.loglog(
            f_fit,
            P_fit,
            linestyle="--",
            linewidth=2,
            label=rf"Ajuste: $\beta = {beta:.3f}$"
        )

        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Densidad espectral de potencia")
        ax.set_title("Espectro de potencias")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        resultado = {
            "frecuencias": f,
            "psd": Pxx,
            "mask_fit": mask,
            "slope": slope,
            "intercept": intercept,
            "beta": beta,
            "fig": fig,
            "ax": ax,
        }
        plt.plot()
        return beta, resultado
    else:
        return beta