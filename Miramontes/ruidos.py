import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch, get_window
from scipy.stats import linregress

# 1) Ruido de colores (shaping en frecuencia)

def _noise_psd(N, psd=lambda f: 1.0):
    """
    Genera una señal 1D de longitud N con densidad espectral ~ psd(f).
    """
    # FFT unilateral real
    Xw = np.fft.rfft(np.random.randn(N))
    f = np.fft.rfftfreq(N)

    S = psd(f).astype(np.float64)
    # normalización suave para no disparar amplitudes
    S = S / (np.sqrt(np.mean(S**2)) + 1e-15)

    Xs = Xw * S
    x = np.fft.irfft(Xs, n=N)
    return x

def white_noise(N):    return _noise_psd(N, lambda f: 1.0)
def blue_noise(N):     return _noise_psd(N, lambda f: np.sqrt(np.where(f==0, 0.0, f)))
def violet_noise(N):   return _noise_psd(N, lambda f: np.where(f==0, 0.0, f))
def brownian_noise(N): return _noise_psd(N, lambda f: 1.0/np.where(f==0, np.inf, f))
def pink_noise(N):     return _noise_psd(N, lambda f: 1.0/np.where(f==0, np.inf, np.sqrt(f)))

def generate_colored_noise(kind: str, N: int):
    kind = kind.lower()
    if   kind in ("white","blanco"):   return white_noise(N)
    elif kind in ("pink","rosa"):      return pink_noise(N)
    elif kind in ("brown","marron","brownian"): return brownian_noise(N)
    elif kind in ("blue","azul"):      return blue_noise(N)
    elif kind in ("violet","violeta"): return violet_noise(N)
    else:
        raise ValueError("kind debe ser: white/pink/brown/blue/violet")




# Binning por cuartiles y Juego del Caos (4 vértices)

def bins_equal_freq_4(arr):

    n = len(arr)
    qs = np.quantile(arr, [0.25, 0.5, 0.75], method="linear")
    labels = np.searchsorted(qs, arr, side="left")  # 0,1,2,3
    return labels

def chaos_game_4(labels, alpha=0.5, vertices=None, start=(0.5, 0.5)):
    """
    labels: array de enteros en {0,1,2,3}
    alpha: fracción de avance hacia el vértice
    vertices: 4x2, si None usa cuadrado unitario en sentido horario
    """
    if vertices is None:
        # Asociaremos: 0->(0,0), 1->(1,0), 2->(1,1), 3->(0,1)
        vertices = np.array([[0.0, 1.0],
                             [1.0, 0.0],
                             [1.0, 1.0],
                             [0.0, 0.0]], dtype=np.float64)

    pts = np.empty((len(labels), 2), dtype=np.float64)
    x, y = float(start[0]), float(start[1])
    for i, lab in enumerate(labels):
        vx, vy = vertices[lab]
        x = alpha * x + alpha * vx
        y = alpha * y + alpha * vy
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


# 4) PSD y estimación de beta (Welch + ajuste log–log)

def plot_psd_beta(time_series, fs,
                  fmin=None, fmax=None,
                  nperseg=2**14, noverlap=None, window='hann',
                  bins_per_decade=None, detrend='constant', show=True):
    """
    Grafica PSD (Welch) y ajusta recta en log10-log10 para obtener beta en S(f) ~ f^{-beta}.
    Devuelve: beta, R2, slope, intercept, (f, Pxx)
    """
    x = np.asarray(time_series, dtype=np.float64)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)

    if noverlap is None:
        noverlap = nperseg // 2

    f, Pxx = welch(x, fs=fs, window=get_window(window, nperseg),
                   nperseg=nperseg, noverlap=noverlap,
                   detrend=detrend, return_onesided=True,
                   scaling='density', average='mean')

    valid = (f > 0) & np.isfinite(f) & np.isfinite(Pxx) & (Pxx > 0)
    if fmin is not None: valid &= (f >= fmin)
    if fmax is not None: valid &= (f <= fmax)

    f_raw = f[valid]
    P_raw = Pxx[valid]
    if f_raw.size < 3:
        raise ValueError("Muy pocos puntos válidos para ajustar beta.")

    if bins_per_decade and bins_per_decade > 0:
        logf = np.log10(f_raw)
        lo, hi = logf.min(), logf.max()
        nb = max(5, int((hi - lo) * bins_per_decade))
        edges = np.linspace(lo, hi, nb + 1)
        f_fit, P_fit = [], []
        for i in range(nb):
            m = (logf >= edges[i]) & (logf < edges[i+1])
            if m.sum() >= 3:
                f_fit.append(10**(logf[m].mean()))
                # promedio geométrico de potencia
                P_fit.append(10**(np.log10(P_raw[m]).mean()))
        f_fit = np.array(f_fit)
        P_fit = np.array(P_fit)
    else:
        f_fit, P_fit = f_raw, P_raw

    X = np.log10(f_fit)
    Y = np.log10(P_fit)
    slope, intercept, r, _, _ = linregress(X, Y)
    beta = -slope
    r2 = r**2

    # Curva del ajuste sobre el rango usado
    y_fit = 10**(intercept + slope * np.log10(f_fit))

    # Gráfica
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(f, Pxx, lw=1.2, label='PSD (Welch)')
    ax.loglog(f_fit, y_fit, '--', lw=2.0,
              label=f"Ajuste: β = {beta:.3f}  (R² = {r2:.3f})")
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('PSD [unidad²/Hz]')
    ax.set_title("Espectro de Potencias y Ajuste log–log")
    rtxt = f"Rango ajuste: [{f_fit.min():.3g}, {f_fit.max():.3g}] Hz"
    ax.text(0.02, 0.02, rtxt, transform=ax.transAxes)
    ax.grid(True, which='both', ls=':')
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()

    return beta, r2, slope, intercept, (f, Pxx)

def plot_PSD(time_series):
    amplitude = np.abs(np.fft.rfft(time_series))
    power = amplitude**2
    freq = np.fft.rfftfreq(len(time_series), d=1)
    plt.loglog(freq[1:], power[1:])  # Evitar f=0
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [unidad²/Hz]')
    plt.title('Espectro de Potencias (FFT)')
    plt.grid(True, which='both', ls=':')
    plt.show()



if __name__ == "__main__":

    source = "pink"     
    N = 300_000         
    fs = 1.0            
    file_txt = None     

    # Parámetros juego del caos
    alpha = 0.5

    # Parámetros de la estimación de beta
    fmin = None         
    fmax = None         
    bins_per_decade = 12  
    nperseg = 2**14

    # ---------- GENERAR / CARGAR SERIE ----------
    if source == "white":
        x = np.random.rand(N)
    else:
        x = generate_colored_noise(source, N)
    # ---------- JUEGO DEL CAOS ----------
    labels = bins_equal_freq_4(x)
    pts = chaos_game_4(labels, alpha=alpha)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2, linewidths=0.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Juego del caos (4 vértices) — fuente: {source}")
    V = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    ax.scatter(V[:,0], V[:,1], s=30, marker="s", edgecolor="k", facecolor="none")
    plt.tight_layout()
    plt.show()

    # ---------- PSD + beta ----------
    plot_PSD(x)
    beta, r2, slope, intercept, _ = plot_psd_beta(
        x, fs=fs,
        fmin=fmin, fmax=fmax,
        nperseg=nperseg, noverlap=None, window='hann',
        bins_per_decade=bins_per_decade, detrend='constant', show=True
    )

    print(f"\n► Resultado beta: β = {beta:.4f}  (R² = {r2:.4f})")
