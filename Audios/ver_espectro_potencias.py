import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, get_window
from scipy.stats import linregress
import argparse
import os

def compute_psd(signal, fs, nperseg=2**14, noverlap=None, window='hann', detrend='constant'):
    """
    PSD via Welch. Devuelve frecuencias f y densidad de potencia Pxx (unidades/Hz).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(
        signal,
        fs=fs,
        window=get_window(window, nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True,
        scaling='density',
        average='mean'
    )
    return f, Pxx

def fit_loglog(f, psd, fmin=None, fmax=None):
    """
    Ajuste lineal en espacio log–log: log10(PSD) = a * log10(f) + b
    Devuelve: (pendiente a, intercepto b, r2, idx_fit)
    """
    # Filtrado de NaNs, ceros y rango
    valid = (f > 0) & np.isfinite(f) & np.isfinite(psd) & (psd > 0)
    if fmin is not None:
        valid &= (f >= fmin)
    if fmax is not None:
        valid &= (f <= fmax)

    f_fit = f[valid]
    p_fit = psd[valid]

    if len(f_fit) < 2:
        raise ValueError("Muy pocos puntos en el rango seleccionado para ajustar.")

    x = np.log10(f_fit)
    y = np.log10(p_fit)

    slope, intercept, r, _, _ = linregress(x, y)
    r2 = r**2
    return slope, intercept, r2, valid

def plot_psd_with_fit(f, psd, slope, intercept, valid_mask, title="", outfile=None):
    plt.figure(figsize=(9,5.5))
    # PSD
    plt.loglog(f, psd, lw=1.4, label='PSD (Welch)')
    # Recta ajustada (solo sobre el rango usado)
    f_fit = f[valid_mask]
    y_fit = 10**(intercept + slope*np.log10(f_fit))
    plt.loglog(f_fit, y_fit, '--', lw=2, label=f"Fit: log10(PSD)= {slope:.3f}·log10(f)+ {intercept:.3f}")

    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [unidad²/Hz]')
    plt.title(title)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=200)
    plt.show()

def main(
    wav_path,
    channel='auto',
    nperseg=2**14,
    fmin=None,
    fmax=None,
    window='hann',
    detrend='constant'
):
    # Leer WAV
    fs, data = wavfile.read(wav_path)

    # Convertir a float si está en entero
    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_int
    else:
        data = data.astype(np.float64)

    # Manejo de canales
    if data.ndim == 2:
        if channel == 'left':
            x = data[:,0]
            used_ch = 'L'
        elif channel == 'right':
            x = data[:,1]
            used_ch = 'R'
        else:
            # mezcla a mono (promedio)
            x = data.mean(axis=1)
            used_ch = 'mono (promedio L+R)'
    else:
        x = data
        used_ch = 'mono'

    # Remover media para evitar pico DC
    x = x - np.mean(x)

    # PSD
    f, Pxx = compute_psd(x, fs, nperseg=nperseg, window=window, detrend=detrend)

    # Ajuste log–log
    slope, intercept, r2, valid_mask = fit_loglog(f, Pxx, fmin=fmin, fmax=fmax)

    # Etiquetas y guardado
    base = os.path.splitext(os.path.basename(wav_path))[0]
    title = f"PSD de '{base}' @ {fs} Hz | Canal: {used_ch}\nAjuste log–log en [{fmin or f[1]:.3g}, {fmax or f[-1]:.3g}] Hz  →  pendiente = {slope:.3f},  R² = {r2:.3f}"
    outfile = f"{base}_PSD_fit.png"


    # Graficar
    plot_psd_with_fit(f, Pxx, slope, intercept, valid_mask, title=title, outfile=outfile)

    print(f"Pendiente (exponente): {slope:.6f}")
    print(f"Intercepto (log10):   {intercept:.6f}")
    print(f"R^2 del ajuste:        {r2:.6f}")
    print(f"Figura guardada como:  {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSD y ajuste lineal en escala log–log para un .wav")
    parser.add_argument("wav", help="archivo.wav")
    parser.add_argument("--channel", choices=["auto", "left", "right"], default="auto",
                        help="Canal a usar: auto (promedia si es estéreo), left, right")
    parser.add_argument("--nperseg", type=int, default=2**14, help="Tamaño de segmento para Welch (potencia de 2 recomendada)")
    parser.add_argument("--fmin", type=float, default=None, help="Frecuencia mínima (Hz) para el ajuste")
    parser.add_argument("--fmax", type=float, default=None, help="Frecuencia máxima (Hz) para el ajuste")
    parser.add_argument("--window", type=str, default="hann", help="Ventana para Welch (hann, hamming, blackman, etc.)")
    parser.add_argument("--detrend", type=str, default="constant", help="Detrending para Welch ('constant' o 'linear')")
    args = parser.parse_args()

    main(
        wav_path=args.wav,
        channel=args.channel,
        nperseg=args.nperseg,
        fmin=args.fmin,
        fmax=args.fmax,
        window=args.window,
        detrend=args.detrend
    )
