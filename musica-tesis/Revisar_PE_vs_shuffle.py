import numpy as np
import matplotlib.pyplot as plt
import mi_libreria as ml
import pandas as pd
import copy
import os


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

def plot_pe_vs_shuffle(array, m, tau, n_shuffles=1000, random_state=None, bins=30):
    """
    Grafica el PE observado de un array junto con la distribución de PE
    de versiones shuffle del mismo array. También calcula Z-score y p-value
    empírico.

    Parámetros
    ----------
    array : array-like
        Serie original.
    m : int
        Dimensión de embedding.
    tau : int
        Retardo.
    n_shuffles : int, opcional
        Número de shuffles.
    random_state : int o None, opcional
        Semilla para reproducibilidad.
    bins : int, opcional
        Número de bins del histograma.

    Regresa
    -------
    resultados : dict
        Diccionario con:
        - 'pe_obs'
        - 'pe_shuffles'
        - 'mu_shuffle'
        - 'sigma_shuffle'
        - 'z_score'
        - 'p_value_left'
        - 'p_value_right'
        - 'p_value_two_sided'
    """
    rng = np.random.default_rng(random_state)
    array = np.asarray(array)

    # PE observado
    # pe_obs = modified_permutation_entropy(array, m, tau)
    # pe_obs= 1 - ml.gamma_index_rank_ties(array,m,tau)[1][-1]
    pe_obs = ml.H_orbit(array, m = 3)
    beta, res = ml.graficar_espectro_beta(array, fs=1.0)
    plt.show()

    # PE de los shuffles
    pe_shuffles = np.empty(n_shuffles)
    shuffled = ml.iaaft(array,n_shuffles)
    for i in range(n_shuffles):
        # shuffled = rng.permutation(array)
        # pe_shuffles[i] = modified_permutation_entropy(shuffled[i,:], m, tau)
        # _, g = ml.gamma_index_rank_ties(shuffled,m,tau)
        # pe_shuffles[i] = 1-g[-1]
        beta, res = ml.graficar_espectro_beta(shuffled[i,:], fs=1.0)
        plt.show()
        pe_shuffles[i] = ml.H_orbit(shuffled[i,:], m = 3)


    # Estadísticos
    mu_shuffle = np.mean(pe_shuffles)
    sigma_shuffle = np.std(pe_shuffles, ddof=1)

    if sigma_shuffle > 0:
        z_score = (pe_obs - mu_shuffle) / sigma_shuffle
    else:
        z_score = np.nan

    # p-values empíricos
    # corrección +1 para evitar p=0 exacto
    p_value_left = (np.sum(pe_shuffles <= pe_obs) + 1) / (n_shuffles + 1)
    p_value_right = (np.sum(pe_shuffles >= pe_obs) + 1) / (n_shuffles + 1)
    p_value_two_sided = 2 * min(p_value_left, p_value_right)
    p_value_two_sided = min(p_value_two_sided, 1.0)

    # Gráfica
    plt.figure(figsize=(8, 5))
    plt.hist(pe_shuffles, bins=bins, alpha=0.75, edgecolor='black', label='PE shuffle')
    plt.axvline(pe_obs, linestyle='--', linewidth=2, label=f'PE observado = {pe_obs:.5f}')
    plt.axvline(mu_shuffle, linestyle=':', linewidth=2, label=f'Media shuffle = {mu_shuffle:.5f}')

    texto = (
        f"Z = {z_score:.3f}\n"
        f"p izq = {p_value_left:.4g}\n"
        f"p der = {p_value_right:.4g}\n"
        f"p 2 colas = {p_value_two_sided:.4g}"
    )

    plt.text(
        0.98, 0.98, texto,
        transform=plt.gca().transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    plt.xlabel('Permutation Entropy')
    plt.ylabel('Frecuencia')
    plt.title(f'PE observado vs PE shuffle\nm={m}, tau={tau}, n_shuffles={n_shuffles}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    resultados = {
        'pe_obs': pe_obs,
        'pe_shuffles': pe_shuffles,
        'mu_shuffle': mu_shuffle,
        'sigma_shuffle': sigma_shuffle,
        'z_score': z_score,
        'p_value_left': p_value_left,
        'p_value_right': p_value_right,
        'p_value_two_sided': p_value_two_sided
    }

    return resultados

composers, datos_composers = extraer_dataset_musica()

composer = 'Beethoven'
labels = list(composers[composer].keys())
for i, label in enumerate(labels):
    serie = composers[composer][label]

    resultados = plot_pe_vs_shuffle(serie, m=2, tau=2, n_shuffles=500, random_state=123)
        
    print("PE observado:", resultados['pe_obs'])
    print("Media shuffle:", resultados['mu_shuffle'])
    print("Sigma shuffle:", resultados['sigma_shuffle'])
    print("Z-score:", resultados['z_score'])
    print("p-value izquierda:", resultados['p_value_left'])
    print("p-value derecha:", resultados['p_value_right'])
    print("p-value dos colas:", resultados['p_value_two_sided'])