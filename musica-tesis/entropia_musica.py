import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import permutations
import os
import pandas as pd
import copy

def extraer_partitura_npy(carpeta):
    arrays = []

    # iteramos sobre los archivos
    for i, archivo in enumerate(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta) and archivo.endswith(".npy"):
            # cargamos el array
            array = np.load(ruta)
            arrays.append(array)

    array_complete = np.array(arrays, dtype='object')

    return array_complete

def extraer_dataset_musica():

    datos_composers = {}
    carpeta = r'data\Sequences\labels'
    archivos_en_carpeta = os.listdir(carpeta)
    index0 = 0
    indice = 0

    for archivo in archivos_en_carpeta:
        ruta_completa = os.path.join(carpeta, archivo)
        serie = pd.read_csv(ruta_completa, header = None)
        composer = archivo.split('-')[1].capitalize() # nombre compositor
        datos_composers[composer] = {} #genero bibio para composer
        datos_composers[composer]['Birth_year'] = archivo.split('-')[0] #año de nacimiento
        index1 = serie.iloc[0, 0].split('\t')[0] #el # del primer serie del composer
        index2 = int(serie.iloc[len(serie)-3, 0].split('\t')[0]) - index0 # # Piezas
        index0 = index2 + index0 # numero total de piezas anteriores
        datos_composers[composer]['# Piezas'] = index2 # Piezas
        datos_composers[composer]['Indice'] = indice
        indice += 1

    composers = {}
    M = 0
    carpeta = r'data\Sequences\Series'
    archivos_en_carpeta = os.listdir(carpeta)

    for archivo in archivos_en_carpeta:
        ruta_completa = os.path.join(carpeta, archivo)
        serie = pd.read_csv(ruta_completa)
        # escoge una serie
        composer = archivo.split('-')[1].capitalize() # nombre compositor
        composers[composer] = {}

        for pieza in range( datos_composers[composer]['# Piezas'] ):
            N = serie.iloc[0, 0].split('\t')[1] # # de elementos por pieza
            M = int(N) + M
            index_n1 = 0 
            index_n2 = int(N)+2 
            serie_n = serie[index_n1 + 2:index_n2].reset_index(drop=True) # resetear index
            serie = serie[index_n2 +1:] # recortar serie Original
            serie_n.index += 1 # que index empiece desde 1
            num_serie_T = serie.columns[0]  # numero de serie de todo el dataset
            num_serie = pieza + 1
            composers[composer]['Serie_'+str(num_serie)] = serie_n.squeeze().to_numpy().astype(float) # agregamos pieza al dicc composer con key como # serie

    ###
    ###

    composers_depurado = copy.deepcopy(composers)
    datos_composers_depurado = copy.deepcopy(datos_composers)

    for i,composer in enumerate(composers.keys()):
        d = 0
        for pieza in composers[composer].keys():
            if len(composers[composer][pieza])//2 < 400:
                del composers_depurado[composer][pieza]
                d = d + 1
        datos_composers_depurado[composer]['# Piezas'] = datos_composers[composer]['# Piezas'] - d


    # 40 promedio de numero de piezas por compositor
    composers_depurado_v2 = copy.deepcopy(composers_depurado)
    composers_depurado_v2_keychange = copy.deepcopy(composers_depurado_v2)
    datos_composers_depurado_v2 = copy.deepcopy(datos_composers_depurado)

    for composer in composers.keys():
        if datos_composers_depurado[composer]['# Piezas'] < 30:
            del composers_depurado_v2[composer]
            del datos_composers_depurado_v2[composer]
        
    for i,composer in enumerate(composers_depurado_v2.keys()):
        datos_composers_depurado_v2[composer]['Indice'] = i 

    for composer in composers_depurado_v2.keys():
        for i,serie in enumerate(composers_depurado_v2[composer].keys()):
            composers_depurado_v2_keychange[composer]['Serie_' + str(i+1)] = composers_depurado_v2_keychange[composer].pop(serie)

    print(" # de compositores restantes: ", len(composers_depurado_v2))

    return composers_depurado_v2, datos_composers_depurado_v2


def permutation_entropy(arr, m=3, tau=1, base=np.e):
    n = len(arr)
    if n < m:
        return np.nan
    embedded = np.array([arr[i:i + m*tau:tau] for i in range(n - m*tau + 1)])
    ranks = np.argsort(embedded, axis=1)
    counts = Counter(tuple(r) for r in ranks)
    probs = np.array(list(counts.values())) / (n - m*tau + 1)
    PE = -np.sum(probs * np.log(probs) / np.log(base))
    return PE

def conditional_entropy(arr, bins=None):
    arr = np.asarray(arr)
    if bins is None:
        bins = len(np.unique(arr))
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist>0]
    Hx = -np.sum(hist*np.log(hist))
    pairs = np.array(list(zip(arr[:-1], arr[1:])))
    joint_hist = np.histogram2d(pairs[:,0], pairs[:,1], bins=bins, density=True)[0]
    joint_hist = joint_hist[joint_hist>0]
    Hxy = -np.sum(joint_hist*np.log(joint_hist))
    return Hxy - Hx

def predictability_metrics(arr, m=3, tau=1, n_surr=100):
    arr = np.asarray(arr)
    states = len(np.unique(arr))
    
    # Observado
    PE = permutation_entropy(arr, m=m, tau=tau)
    rate = PE / np.log(states)
    CE = conditional_entropy(arr, bins=states)
    
    # Sustitutos
    PE_surr = []
    rate_surr = []
    CE_surr = []
    
    for _ in range(n_surr):
        surr = np.random.permutation(arr)
        PE_surr.append(permutation_entropy(surr, m=m, tau=tau))
        rate_surr.append(PE_surr[-1] / np.log(states))
        CE_surr.append(conditional_entropy(surr, bins=states))
    
    results = {
        'observed': {'PE': PE, 'rate': rate, 'CE': CE},
        'surrogates': {'PE': PE_surr, 'rate': rate_surr, 'CE': CE_surr}
    }
    return results

def plot_metrics(metrics):
    observed = metrics['observed']
    surrogates = metrics['surrogates']
    
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    
    metric_names = ['PE', 'rate', 'CE']
    
    delta = []
    for name in metric_names:
        delta.append(np.max(surrogates[name]) - np.min(observed[name]))
    ylim_delta = np.max(delta)

    for i, name in enumerate(metric_names):
        # Valor observado
        ax[i].scatter(0, observed[name], color='blue', label='Observed')
        # Surrogates
        ax[i].scatter(np.random.normal(0, 0.05, len(surrogates[name])), surrogates[name], color='red', alpha=0.5, label='Surrogates')
        ax[i].set_title(name)
        ax[i].set_xticks([])
        ylim1 = np.max(surrogates[name]) +0.05*ylim_delta
        ax[i].set_ylim(ylim1-ylim_delta-0.1*ylim_delta, ylim1)
        ax[i].legend()
    
    plt.tight_layout()
    plt.show()




# ======================================================
# 4. Ejemplo de uso
# ======================================================

if __name__ == "__main__":
    # Ejemplo: melodía simple en MIDI
    composers, datos_composers = extraer_dataset_musica()
    
    for notes in composers['Bach'].values():
    # notes = composers['Bach']['Serie_1'] 
        notes = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60] * 30  # Escala ascendente y descendente repetida
    # Evaluar notas
# Ejemplo de uso
        metrics = predictability_metrics(np.array(notes))
        plot_metrics(metrics)
    
    # Evaluar intervalos
    intervals = np.diff(notes)
    metrics = predictability_metrics(np.array(notes))
    plot_metrics(metrics)