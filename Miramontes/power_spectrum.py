import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def power_spectrum(x, dt=1.0, log_scale='loglog', smooth=False, Nsmooth=10):
    """
    Calcula y grafica el espectro de potencia de una serie temporal.
    
    Parámetros:
    ------------
    x : array_like
        Serie temporal (1D)
    dt : float
        Intervalo de muestreo (por defecto 1.0)
    log_scale : str
        Tipo de escala: 'linear', 'semilogx', 'loglog'
    smooth : bool
        Si True, aplica suavizado por ventana móvil
    Nsmooth : int
        Tamaño de la ventana de suavizado
    """

    # Eliminar tendencia lineal
    x = detrend(np.asarray(x))

    # FFT y frecuencias
    N = len(x)
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)

    # Potencia (normalizada)
    Pxx = (np.abs(fft_vals)**2) / N

    # Suavizado opcional
    if smooth:
        kernel = np.ones(Nsmooth) / Nsmooth
        Pxx = np.convolve(Pxx, kernel, mode='same')

    # Graficar
    plt.figure(figsize=(8, 5))
    if log_scale == 'linear':
        plt.plot(freqs, Pxx, lw=1)
        plt.xlabel("Frecuencia")
        plt.ylabel("Potencia")
    elif log_scale == 'semilogx':
        plt.semilogx(freqs, Pxx, lw=1)
        plt.xlabel("Frecuencia (log)")
        plt.ylabel("Potencia")
    elif log_scale == 'loglog':
        plt.loglog(freqs[1:], Pxx[1:], lw=1)
        plt.xlabel("Frecuencia (log)")
        plt.ylabel("Potencia (log)")
    else:
        raise ValueError("log_scale debe ser 'linear', 'semilogx' o 'loglog'")

    plt.title("Espectro de potencia")
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    return freqs, Pxx

# Ejemplo con una serie caótica del mapa logístico
def logistic_map(r, x):
    return r * x * (1 - x)

N = 5000
r = 4.0
x = np.zeros(N)
x[0] = 0.5
for i in range(N-1):
    x[i+1] = logistic_map(r, x[i])

x = np.random.rand(5000)
x = np.loadtxt('series/temp_madison.txt')

# Calcular espectro de potencia
freqs, Pxx = power_spectrum(x, dt=1.0, log_scale='linear', smooth=True)
