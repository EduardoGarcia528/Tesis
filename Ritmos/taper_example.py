from multitaper_spectrogram_python import multitaper_spectrogram  # import multitaper_spectrogram function from the multitaper_spectrogram_python.py file
import numpy as np  # import numpy
from scipy.signal import chirp  # import chirp generation function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def estimar_pendiente_spectral(spect, sfreqs, fmin=0.01, fmax=1.0, graficar=True):
    """
    Calcula la pendiente del espectro promedio en escala log-log.
    
    Parámetros:
    - spect: espectrograma multitaper (freq x time)
    - sfreqs: frecuencias asociadas al espectrograma
    - fmin, fmax: rango de frecuencias para el ajuste
    - graficar: si True, muestra la gráfica

    Retorna:
    - pendiente negativa de la recta ajustada (β)
    """
    # 1. Promediar sobre el tiempo (colapsar dimensión temporal)
    psd_mean = spect

    # 2. Filtrar frecuencias en la región de interés
    mask = (sfreqs > 0)
    log_freqs = np.log10(sfreqs[mask])
    log_psd = np.log10(psd_mean[mask])
    
    roi = (sfreqs[mask] >= fmin) & (sfreqs[mask] <= fmax)

    # 3. Ajuste lineal en log-log
    slope, intercept, r_value, p_value, std_err = linregress(log_freqs[roi], log_psd[roi])

    # 4. Graficar
    if graficar:
        plt.figure(figsize=(8, 6))
        plt.plot(sfreqs, psd_mean, label='PSD promedio', color='C0')
        plt.plot(
            sfreqs[mask][roi],
            10**(intercept + slope * log_freqs[roi]),
            'r--',
            label=f'Ajuste: β = {-slope:.2f}'
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Densidad espectral de potencia')
        plt.title('Espectro multitaper promedio (log-log)')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    return round(-slope, 2)

import numpy as np
from scipy.signal.windows import dpss
from scipy.signal import detrend

def multitaper_full(data, fs=10, time_bandwidth=2, num_tapers=3, nfft=None, detrend_opt='constant', weighting='unity'):
    """
    Aplica multitaper sin ventanas: a toda la serie completa.
    """
    N = len(data)

    # Detrend (si se desea)
    if detrend_opt == 'constant':
        data = detrend(data, type='constant')
    elif detrend_opt == 'linear':
        data = detrend(data, type='linear')

    # DPSS tapers
    tapers, eigen = dpss(N, time_bandwidth, num_tapers, return_ratios=True)

    # Define NFFT
    if nfft is None:
        nfft = max(2 ** int(np.ceil(np.log2(N))), N)

    # Ponderaciones
    if weighting == 'eigen':
        weights = eigen[:, np.newaxis] / np.sum(eigen)
    else:  # 'unity'
        weights = np.ones((num_tapers, 1)) / num_tapers

    # Calcular espectros
    spectra = []
    for k in range(num_tapers):
        tapered = data * tapers[k]
        fft_vals = np.fft.fft(tapered, n=nfft)
        power = np.abs(fft_vals) ** 2
        spectra.append(power)

    # Promedio ponderado
    spectra = np.array(spectra)
    spectrum = np.average(spectra, axis=0, weights=weights.flatten())

    # Frecuencias
    freqs = np.fft.fftfreq(nfft, d=1/fs)

    # Devolver solo mitad positiva (one-sided)
    half = slice(0, nfft // 2 + 1)
    return freqs[half], spectrum[half]


def multitaper(onsets):
    # Set spectrogram params
    fs = 10  # Sampling Frequency
    frequency_range = [0.0001, 100]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = 2  # Set time-half bandwidth
    num_tapers = 3  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [10, 1]  # Window size is 4s with step size of 1s
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = False  # use multiprocessing
    n_jobs = 3  # use 3 cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = True  # plot spectrogram
    return_fig = False  # do not return plotted spectrogram
    clim_scale = False # do not auto-scale colormap
    verbose = True  # print extra info
    xyflip = False  # do not transpose spect output matrix


    interonset_intervals = np.diff(onsets)

    if False:
        interonset_intervals = np.random.permutation(interonset_intervals)
        onsets = np.concatenate([[onsets[0]], onsets[0] + np.cumsum(interonset_intervals)])

    # Crear spike train a 10 Hz (10 muestras por segundo)
    dt = 1 / fs

    # Eje temporal
    t_max = np.max(onsets) + dt
    t_axis = np.arange(0, t_max, dt)

    # Inicializar spike train
    spike_train = np.zeros_like(t_axis)

    # Marcar spikes en el bin más cercano
    for onset in onsets:
        idx = np.argmin(np.abs(t_axis - onset))
        spike_train[idx] = 1

    # Compute the multitaper spectrogram
    spect, stimes, sfreqs = multitaper_spectrogram(interonset_intervals, fs, frequency_range, time_bandwidth, num_tapers, window_params, min_nfft, detrend_opt, multiprocess, n_jobs,
                                                weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
    return spect, sfreqs

onsets = np.load('onsets.npy')
interonset_intervals = np.diff(onsets)

if True:
    interonset_intervals = np.random.permutation(interonset_intervals)
    onsets = np.concatenate([[onsets[0]], onsets[0] + np.cumsum(interonset_intervals)])
# spect, sfreqs = multitaper(onsets)

fs = 10
dt = 1 / fs

# Eje temporal
t_max = np.max(onsets) + dt
t_axis = np.arange(0, t_max, dt)

# Inicializar spike train
spike_train = np.zeros_like(t_axis)

# Marcar spikes en el bin más cercano
for onset in onsets:
    idx = np.argmin(np.abs(t_axis - onset))
    spike_train[idx] = 1
sfreqs, spect = multitaper_full(spike_train, fs=10, time_bandwidth=2, num_tapers=3)
print(sfreqs)
beta = estimar_pendiente_spectral(spect, sfreqs, fmin=0.01, fmax=1.0, graficar=True)
print(f"Pendiente espectral (β): {beta}")
