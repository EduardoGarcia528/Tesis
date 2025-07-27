"""IMPORTAR"""

import pandas as pd
import os
from scipy.interpolate import PchipInterpolator
import numpy as np
import copy
from numba import njit
import multiprocessing as mp
from mne.time_frequency import psd_array_multitaper
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def onsets_function(matriz):
    data = matriz # la matriz
    onsets_negras = np.array([]) # iniciamos array de oonsets 
    cum = 0 # iniciamos la suma acumulativa
    for i in np.unique(data[:,4]): # iteramos compases
        array = data[np.where(data[:,4] == i),:][0] #seleccionamos array del compas
        if len(onsets_negras) > 0: # el primer compas cum = 0
            cum = np.max(array[:,1]) + cum # el segundo compas es cum = offset más alto
        array = array[~np.any(array[:,:] == -1, axis=1)] # para eliminar silencios
        array =  array[array[:, 0].argsort()]    # ordenar por onsets
        array = np.unique(array[:,0]) # eliminar onsets repetidos
        array = array + cum # sumamos el cum
        onsets_negras = np.concatenate((onsets_negras,array)) # lo añadimos 

    """onsets está listo para ser analizado por multitapers"""
    return onsets_negras

def spike_train_function(onsets_negras,shuffle, fs = 10):
    # Calcular interonset intervals
    interonset_intervals = np.diff(onsets_negras)

    if shuffle:
        interonset_intervals = np.random.permutation(interonset_intervals)
        onsets_negras = np.concatenate([[onsets_negras[0]], onsets_negras[0] + np.cumsum(interonset_intervals)])


    # Crear spike train a 10 Hz (10 muestras por segundo)
    dt = 1 / fs

    # Eje temporal
    t_max = np.max(onsets_negras) + dt
    t_axis = np.arange(0, t_max, dt)

    # Inicializar spike train
    spike_train = np.zeros_like(t_axis)

    # Marcar spikes en el bin más cercano
    for onset in onsets_negras:
        idx = np.argmin(np.abs(t_axis - onset))
        spike_train[idx] = 1
    return spike_train, interonset_intervals

def multitaper(spike_train, fs = 10, graficar = False):
    psd, freqs = psd_array_multitaper(
        spike_train,
        sfreq=fs,
        fmin=0.01,
        fmax=1.0,
        adaptive=True,
        normalization='full',
        verbose=False
    )

    # 3. Transformación a escala log-log
    mask = (freqs > 0)  # Evitar log(0)
    log_freqs = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])

    # 4. Ajuste lineal en la región de interés (región de potencias tipo ley de potencias)
    roi = (freqs[mask] >= 0.01) & (freqs[mask] <= 1.0)
    slope, intercept, r_value, p_value, std_err = linregress(log_freqs[roi], log_psd[roi])

    # 5. Gráfica del espectro y la ley de potencias ajustada
    if graficar:
        plt.figure(figsize=(8, 6))
        plt.plot(freqs, psd, 'o-', label='PSD', markersize=4)
        plt.plot(
            freqs[mask][roi],
            10**(intercept + slope * log_freqs[roi]),
            'r--',
            label=f'Ajuste: β = {slope:.2f}'
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Densidad espectral de potencia')
        plt.title('Espectro multitaper en escala log-log')
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
                    print("HAHA")
                    break
                onsets_negras = onsets_function(matriz)
                if len(onsets_negras) > 200:
                    spike_train,interonset_intervals = spike_train_function(onsets_negras, shuffle=True)
                    beta = multitaper(spike_train)
                    J = J_univariante(spike_train)
                    J_interpolado = J_univariante(interpolador(spike_train,'lineal',10))
                    etiqueta_csv = str(os.path.basename(raiz)) + ".krn"
                    # print(etiqueta_csv)
                    df.loc[df['Archivo'] == etiqueta_csv, 'Beta'] = beta
                    df.loc[df['Archivo'] == etiqueta_csv, 'J'] = J
                    df.loc[df['Archivo'] == etiqueta_csv, 'J_interp10'] = J_interpolado
                    
                    break
    df.to_csv("salida3.csv", index=False)  # sin escribir el índice como columna

df = pd.read_csv("data/metadatos_piezas.csv")
# df['Beta'] = 'NaN'
df['J'] = 'NaN'
df['J_interp10'] = 'NaN'
main(df)