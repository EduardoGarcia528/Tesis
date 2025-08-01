"""IMPORTAR"""

import pandas as pd
import os
from scipy.interpolate import PchipInterpolator
import numpy as np
import copy
from multitaper_spectrogram_python import multitaper_spectrogram  # import multitaper_spectrogram function from the multitaper_spectrogram_python.py file
from numba import njit
from scipy.signal import chirp  # import chirp generation function
import multiprocessing as mp
from scipy.signal.windows import dpss
from scipy.signal import detrend
from mne.time_frequency import psd_array_multitaper
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import multitaper_spectrogram_python as taper

import importlib
import funciones
importlib.reload(funciones)
print()
from funciones import J_univariante, generar_uniforme_centrada, interpolador, remover_duplicados

from music21 import *
environment.set('musescoreDirectPNGPath', r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")


def combine_matrix(path):
    voices = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".npy"):
            try:
                arr = np.load(f"{path}/{filename}")
            except:
                try:
                    arr = np.load(f"{path}/{filename}",allow_pickle=True)
                except:
                    print("No se pudo cargar archivo: ", path)
                    return np.array([0])
            voices.append(arr)
    combined = np.vstack([voices[i] for i in range(len(voices))])
    try:
        matriz = combined[combined[:, -1].argsort()]       
    except Exception as e: 
        print(f"Errir al combinar matriz {path}: {e}")
        return np.array([0])
    return matriz


def onsets_function(matriz, shuffle):
    data = matriz # la matriz
    onsets_negras = np.array([]) # iniciamos array de oonsets 
    repeticiones = np.array([])   # array con número de repeticiones por onset
    cum = 0 # iniciamos la suma acumulativa
    for i in np.unique(data[:,4]): # iteramos compases
        array = data[np.where(data[:,4] == i),:][0] #seleccionamos array del compas
        if len(onsets_negras) > 0: # el primer compas cum = 0
            cum = np.max(array[:,1]) + cum # el segundo compas es cum = offset más alto
        # array = array[~np.any(array[:,:] == -1, axis=1)] # para eliminar silencios
        array =  array[array[:, 0].argsort()]    # ordenar por onsets
        array,counts = np.unique(array[:,0], return_counts=True) # eliminar onsets repetidos
        array = array + cum # sumamos el cum
        repeticiones = np.concatenate((repeticiones,counts)) # lo añadimos 
        onsets_negras = np.concatenate((onsets_negras,array)) # lo añadimos 

    """onsets está listo para ser analizado por multitapers"""
    interonset_intervals = np.abs(np.diff(onsets_negras))
    # print(interonset_intervals)
    if shuffle:
        interonset_intervals = np.random.permutation(interonset_intervals)
        onsets_negras = np.concatenate([[onsets_negras[0]], onsets_negras[0] + np.cumsum(interonset_intervals)])
    
    # Crear spike train a 10 Hz (10 muestras por segundo)
    fs = 10  # frecuencia de muestreo (Hz)
    # dt = min(interonset_intervals)
    dt = 1 /fs
    # print(dt)
    # fs = 1 /dt
    # Eje temporal
    t_max = np.max(onsets_negras) + dt
    t_axis = np.arange(0, t_max, dt)

    # Inicializar spike train
    spike_train = np.zeros_like(t_axis)

    # Marcar spikes en el bin más cercano
    for onset,weight in zip(onsets_negras,repeticiones):
        idx = np.argmin(np.abs(t_axis - onset))
        spike_train[idx] = weight

    return onsets_negras, interonset_intervals, spike_train, fs

def multitaper_full(data, fs, time_bandwidth, num_tapers, nfft=None, detrend_opt='constant', weighting='unity'):
    """
    Aplica multitaper sin ventanas: a toda la spike train.
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

    # n = len(data)
    # fft_vals = np.fft.fft(data)
    # frq = np.fft.fftfreq(n, d=1/fs)

    # # Espectro de potencias (PSD)
    # psd = (np.abs(fft_vals) ** 2) / n

    # # Solo tomar la mitad positiva
    # half_n = n // 2
    # frq = frq[:half_n]
    # psd = psd[:half_n]
    # return frq,psd
    return freqs[half], spectrum[half]


def estimar_pendiente_spectral(spect, sfreqs, fmin, fmax, graficar):
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
    # psd_mean = np.mean(spect, axis=1)
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


def main(df,ruta_base = "data\humdrum-data-numpy", profundidad_max=8):
    lineas_unicas = set()
    for raiz, dirs, archivos in os.walk(ruta_base):
        # Calcular profundidad relativa desde la carpeta base
        profundidad = os.path.relpath(raiz, ruta_base).count(os.sep)
        if profundidad > profundidad_max:
            # No descender más en esta rama
            dirs[:] = []
            continue
        for filename in archivos:
            if filename.endswith('.npy'):
                matriz = combine_matrix(raiz)
                if np.shape(matriz) == (1,):
                    print("matriz no permitida")
                    break
                onsets_negras, interonset_intervals, spike_train,fs = onsets_function(matriz, shuffle=False)
                # try:
                #     sfreqs, spect = multitaper_full(spike_train, fs=fs, time_bandwidth=2, num_tapers=3)
                #     beta = estimar_pendiente_spectral(spect, sfreqs, fmin=0.01, fmax=1.0, graficar=False)
                # except:
                #     break
                if len(onsets_negras) > 200:
                    
                    J = J_univariante(spike_train)
                    # J_interpolado = J_univariante(interpolador(spike_train,'lineal',10))
                    etiqueta_csv = str(os.path.basename(raiz)) + ".krn"
                    print(etiqueta_csv)
                    # df.loc[df['Archivo'] == etiqueta_csv, 'Beta_new'] = beta
                    # df.loc[df['Archivo'] == etiqueta_csv, 'Beta_multi'] = beta
                    df.loc[df['Archivo'] == etiqueta_csv, 'Jmulti'] = J
                    # df.loc[df['Archivo'] == etiqueta_csv, 'J_interp10'] = J_interpolado
                    
                    break
    df.to_csv("salida_multi.csv", index=False)  # sin escribir el índice como columna

df = pd.read_csv("salida_multi.csv")
# df['Beta_multi'] = 'NaN'
df['Jmulti'] = 'NaN'
# df['J_interp10'] = 'NaN'
main(df)