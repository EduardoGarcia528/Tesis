import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import permutations
import os
import seaborn as sns
from numba import njit
from funciones import remove_consecutive_duplicates, permutation_entropy
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

def predictability_metrics(method,arr, m, tau, n_surr=100):
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
        if method == 'PE':
            PE_surr.append(permutation_entropy(surr, m=m, tau=tau))
        elif method == 'CE':
            CE_surr.append(conditional_entropy(surr, bins=states))
        elif method == 'rate':
            rate_surr.append(PE_surr[-1] / np.log(states))
    
    results = {
        'observed': {'PE': PE, 'rate': rate, 'CE': CE},
        'surrogates': {'PE': PE_surr, 'rate': rate_surr, 'CE': CE_surr}
    }
    return results

def plot_metrics(metrics):
    observed = metrics['observed']
    surrogates = metrics['surrogates']
    
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    
    metric_names = ['PE']
    
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


def test_hypothesis(null_distribution, observed_value, alpha=0.00, two_tailed=False, graficar=False):
    """
    Calcula p-value dado un evento observado y una distribución nula.
    Grafica la distribución con el punto observado y el umbral de significancia.
    
    Parámetros:
    null_distribution : array-like
        Muestras simuladas bajo H0
    observed_value : float
        Valor observado a comparar
    alpha : float
        Nivel de significancia
    two_tailed : bool
        Si True, usa prueba bilateral; si False, unilateral
    """
    null_distribution = np.array(null_distribution)
    
    # Calcular p-value
    if two_tailed:
        extreme_count = np.sum(np.abs(null_distribution - np.mean(null_distribution)) 
                               >= np.abs(observed_value - np.mean(null_distribution)))
    else:
        extreme_count = np.sum(null_distribution <= observed_value)
        
    p_value = extreme_count / len(null_distribution)
    
    # Decisión
    reject = p_value <= alpha
    
    # Umbral crítico para graficar
    lower_crit = np.percentile(null_distribution, 100 * (alpha/2)) if two_tailed else np.percentile(null_distribution, 100 * (alpha))
    upper_crit = np.percentile(null_distribution, 100 * (1 - alpha/2)) if two_tailed else None
    
    # Gráfica
    if graficar is True:
        plt.figure(figsize=(8,5))
        sns.histplot(null_distribution, kde=True, bins=30, color="skyblue", stat="density")
        
        # Región crítica
        if two_tailed:
            plt.axvline(lower_crit, color="red", linestyle="--", label=f"α/2 = {alpha/2}")
            plt.axvline(upper_crit, color="red", linestyle="--")
        else:
            plt.axvline(lower_crit, color="red", linestyle="--", label=f"α = {alpha}")
        
        # Valor observado
        plt.axvline(observed_value, color="black", linewidth=2, label=f"Valor observado = {observed_value:.3f}")
        
        plt.title("Test de Hipótesis con Distribución Nula")
        plt.legend()
        plt.show()
    
    return lower_crit, reject


# ======================================================
# 4. Ejemplo de uso
# ======================================================

if __name__ == '__main__':
    # mp.freeze_support()

    composers, datos_composers = extraer_dataset_musica()

    for i, serie in enumerate(composers['Bach'].keys()):
        if i != 18:
            continue
        print(serie)
        f = composers['Bach'][serie]
        # f = remove_consecutive_duplicates(f, tolerance=0)
        metrics = predictability_metrics('PE',np.array(np.abs(np.diff(f))),m=3, tau=1)
        print("PE observado:", metrics['observed']['PE'])
        null_dist = metrics['surrogates']['PE']
        lower_crit, reject = test_hypothesis(null_dist, metrics['observed']['PE'], alpha=0.0, two_tailed=False, graficar=True)
        print(f"Rechazar H0: {reject} (umbral crítico: {lower_crit})")
    """
    PEs = np.full((19,2160), np.nan)
    PEs_null = np.full((19,2160), np.nan)
    for x, composer in enumerate(composers.keys()):
        if composer != 'Bach':
            continue
        birth_year = datos_composers[composer]['Birth_year']
        print(composer)
        for y, serie in enumerate(composers[composer]):
            f = composers[composer][serie]
            f= remove_consecutive_duplicates(f, tolerance=0)
            metrics = predictability_metrics('PE',np.array(f),m=10, tau=1)
            PEs[x,y] = metrics['observed']['PE']
            null_dist = metrics['surrogates']['PE']
            PEs_null[x,y] = test_hypothesis(null_dist, PEs[x,y], alpha=0.0, two_tailed=False, graficar=True)[0]
            # plot_metrics(metrics)
            lenght = y
        print(composer)
        np.save(f'new_data/PEs_null_10/{birth_year}_{composer}_PEs_null.npy', PEs_null[x,:lenght+1])
        np.save(f'new_data/PEs_absdiff/{birth_year}_{composer}_PEs.npy', PEs[x,:lenght+1])
    # np.save('J_composers_Hz_depurado.npy', Js)
    
    """
