import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from functools import partial
import multiprocessing as mp
import copy
import pandas as pd
import os
import math
from collections import Counter
from funciones import interpolador


@njit
def distancia(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

@njit
def mejor_vector(p1, p2):
    # Precomputar diferencias en los 9 cuadrantes
    diffs = [
        [p2[0], p2[1]],
        [p2[0], p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1]],
        [p2[0] + 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0], p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1]],
        [p2[0] - 2 * np.pi, p2[1] + 2 * np.pi],
    ]
    # Encontrar el índice con menor distancia
    d_og = distancia(p1,p2)
    min_idx = 0
    for i in range(9):
        d = distancia(p1, diffs[i])
        if d < d_og:
            min_idx = i
            d_og = d
    p2 = diffs[min_idx]
    return [p2[0] - 2*p1[0], p2[1] - 2*p1[1]]



@njit
def isaac_vector(p1, p2):
    # Precomputar diferencias en los 9 cuadrantes
    cuadrante = [[p2[0]-p1[0], p2[1]-p1[1]], [p2[0]-p1[0], p2[1]+2*np.pi-p1[1]],
        [p2[0]+2*np.pi-p1[0],p2[1]+2*np.pi-p1[1]],[p2[0]+2*np.pi-p1[0],p2[1]-p1[1]],
        [p2[0]+2*np.pi-p1[0],p2[1]-2*np.pi-p1[1]],[p2[0]-p1[0],p2[1]-2*np.pi-p1[1]],
        [p2[0]-2*np.pi-p1[0],p2[1]-2*np.pi-p1[1]],[p2[0]-2*np.pi-p1[0],p2[1]-p1[1]],
        [p2[0]-2*np.pi-p1[0],p2[1]+2*np.pi-p1[1]]]
    # Encontrar el índice con menor distancia
    d_og = distancia(p1,p2)
    min_idx = 0
    for i in range(9):
        d = distancia(p1, cuadrante[i])
        if d < d_og:
            min_idx = i
            d_og = d
    p2 = cuadrante[min_idx]
    return [p2[0]-p1[0],p2[1]-p1[1]]

@njit
def calcular_angulos(vectores):
    n = len(vectores) - 1
    angulos = np.empty(n)
    for i in range(n):
        v1 = vectores[i]
        v2 = vectores[i + 1]
        norm_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if norm_v1 == 0 or norm_v2 == 0:
            angulo = 0.0
        else:
            v1n0 = v1[0] / norm_v1
            v1n1 = v1[1] / norm_v1
            v2n0 = v2[0] / norm_v2
            v2n1 = v2[1] / norm_v2
            dot = v1n0 * v2n0 + v1n1 * v2n1
            if dot > 1.0: dot = 1.0
            if dot < -1.0: dot = -1.0
            angulo = np.arccos(dot)
            cruz = v1[0] * v2[1] - v1[1] * v2[0]
            if cruz > 0:
                angulo = np.pi - angulo
            elif cruz == 0 and angulo < 0:
                angulo = np.pi
            elif cruz < 0:
                angulo += np.pi
        angulos[i] = angulo
    return angulos

def caminata_univariante(X, tau, bivariante):
    if bivariante is False:
        x1 = X[tau:]
        y1 = X[:-tau]
    else:
        x1 = X
        y1 = bivariante
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    n = len(ff1) - 1
    vectores = np.empty((n,2)) #(n,2)
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def indice_J(angulos):
    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    return J

def entropia_shannon(x, bins=100):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    try:
        hist, _ = np.histogram(x, bins=bins, density=True)
    except: 
        return np.nan
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    p = hist / hist.sum()
    H = -np.sum(p * np.log2(p))
    return H / np.log2(bins)

def entropia_permutacion(x, m=4, tau=1):
    n = len(x)
    if n < (m - 1) * tau + 1:
        return np.nan

    patrones = []
    for i in range(n - (m - 1) * tau):
        ventana = x[i:i + tau * m:tau]
        orden = tuple(np.argsort(ventana))
        patrones.append(orden)

    cuenta = Counter(patrones)
    total = sum(cuenta.values())
    p = np.array(list(cuenta.values())) / total
    H = -np.sum(p * np.log2(p))
    H_norm = H / np.log2(math.factorial(m))  # normalización
    return H_norm

def diff_S(d, angulos):
    if d == 0:
        return entropia_shannon(angulos)
    for _ in range(d):
        angulos = np.diff(angulos)

    return entropia_shannon(angulos)

def main(angulos,d):
    # angulos = np.load('henon_C_1000diff.npy')
    entropia = diff_S(d,angulos=angulos)
    J = indice_J(angulos)
    if not np.isfinite(entropia):
        entropia = 0
    return J, entropia

#dataframe datos de compositores 
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


if __name__ == '__main__':
    # mp.freeze_support()

    composers, datos_composers = extraer_dataset_musica()

    """"""
    Js = np.full((19,2160), np.nan)
    for x, composer in enumerate(composers):
        birth_year = datos_composers[composer]['Birth_year']
        for y, serie in enumerate(composers[composer]):
            f = composers[composer][serie]
            random = np.random.uniform(0,1, size = len(f))
            vectores = caminata_univariante(interpolador(f,'lineal',2),tau = 1,bivariante=False)
            angulos = calcular_angulos(vectores)
            J, S = main(angulos=angulos, d=1)
            Js[x,y] = J
            lenght = y
        print(composer)
        np.save(f'new_data/Js_interp2/{birth_year}_{composer}_Js.npy', Js[x,:lenght])
    # np.save('J_composers_Hz_depurado.npy', Js)
    
    
    """"""

    num_compositores = 19
    Ns = np.load(r'data/Ns_depurado.npy') 
    J_null_matrix = np.load(r'new_data\J_null_continuo.npy')
    J_minus = J_null_matrix[:2,:]
    # J_minus = np.load('data/J_minus_continuo.npy')

    pts_interp = 1
    carpeta2 = 'data/J_lineal_sincorte_depurado/interp_'+str(pts_interp)
    # carpeta2 = 'data/J_hermite_sincorte_depurado/interp_'+str(pts_interp) 

    archivos_en_carpeta2 = os.listdir(carpeta2)

    data2 = [1 - np.sort(np.load(os.path.join(carpeta2, array)))[:] for array in archivos_en_carpeta2]
    Ns_data = [Ns[array] for array in range(np.shape(Ns)[0])]
    J_OG = [1-np.load(r'data\J_composers_Hz_depurado.npy')[array] for array in range(np.shape(Ns)[0])]
    J_new = [Js[array] for array in range(np.shape(Ns)[0])]

    data2 = [array[~np.isnan(array)] for array in data2]
    Ns_data = [array[~np.isnan(array)] for array in Ns_data]
    J_OG = [array[~np.isnan(array)] for array in J_OG]
    data2 = [array[~np.isnan(array)] for array in J_new]

    # for j in range(num_compositores):
    #     for i in range(len(Ns_data[j])):
    #         Ns_data[j][i] = (Ns_data[j][i] - 1)*pts_interp + Ns_data[j][i]
    #         if Ns_data[j][i]//2 >= 13920:
    #             Ns_data[j][i] = 27840

    umbral = [[J_minus[1, np.where(J_minus[0] == (i)//2)[0]][0] for i in Ns_data[j]] for j in range(num_compositores)]
    # Aquí, asume que `puntos` es la lista con los num_compositores elementos
    # puntos1 = [J_minus[int(np.mean(Ns_data[j])) - 20] for j in range(num_compositores)]
    puntos2 = [np.mean(np.array([J_minus[1, np.where(J_minus[0] == i//2)[0]] for i in Ns_data[j]])) for j in range(num_compositores)]
    mediana = [np.median(array) for array in data2]
    print('mean',np.mean(puntos2))
    fig, ax = plt.subplots(figsize=(15, 10))
    # box2 = ax.boxplot(data2, patch_artist=True)

    total_red_points = 0
    total_points = 0
    porcentajes = []
    for i, (d, u) in enumerate(zip(data2, umbral)):
        # Añadir dispersión para evitar superposición
        x = np.random.normal(i + 1, 0.04, size=len(d))
        colors = ['blue' if d_val < u_val else 'red' for d_val, u_val in zip(d, [x for x in u])]
        plt.scatter(x, d, alpha=0.5, color='none', edgecolors=colors)
        # Calcular porcentaje de puntos rojos
        num_red = sum(c == 'red' for c in colors)
        total_points += len(d)
        total_red_points += num_red
        percentage_red = (num_red / len(d)) * 100
        porcentajes.append(percentage_red)
        # Calcular la posición del texto sin exceder el margen superior
        y_text = min(max(d) + 0.05, ax.get_ylim()[1] - 0.05)  # Limitar la posición de texto al margen superior
        
        # Añadir texto con el porcentaje
        ax.text(i + 1, 0.08, f'{percentage_red:.1f}%', ha='center', fontsize=13, color='black')

    # np.save('porcentajes_run_avg_50.npy', np.array(porcentajes))
    total_percentage_red = (total_red_points / total_points) * 100

    # Añadir texto con el porcentaje total
    ax.text(0.2, 0.7, f'Total: {total_percentage_red:.1f}% puntos rojos', 
            ha='center', va='bottom', fontsize=13, transform=ax.transAxes, color='black')


    # ax.set_xlabel(f'Distribución del índice (1-J) con interpolación lineal de {pts_interp} pts', fontsize=13)
    # ax.set_xlabel(f'Distribución del índice (1-J) de las melodias de larga duración', fontsize=13)
    ax.set_ylabel('1 - J', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xticks(np.arange(1, num_compositores+1))

    # ax.scatter(np.arange(1, 78), puntos1, color='red', s=50, zorder=3, marker = '*', label=r'$U\{\overline{N_J}\}$')
    ax.scatter(np.arange(1, num_compositores+1), puntos2, color='#87CEEB', s=25, zorder=3, marker = 'o', label=r'Umbral $\overline{U\{ N_J \}}$')
    # ax.axhline(y=0.029071461469613136)
    ax.scatter(np.arange(1, num_compositores+1), mediana, color='black',s=25,zorder=3,marker='o',label='Mediana')
    ax.plot(np.arange(1, num_compositores+1), puntos2, color='blue', linestyle='-', linewidth=0.8)
    # ax.plot(np.arange(1, 18), mediana, color='red', linestyle='-', linewidth=0.8)
    ax.plot([],[], marker = 'o',color='none', markeredgecolor='red', label = 'Determinista')
    ax.plot([],[], marker='o', color='none', markeredgecolor='blue', label= 'Aleatoria')
    ax.plot([],[], marker='none', color='none', label = '%: Porcentajes de puntos rojos')

    # Agregar la leyenda al gráfico
    # ax.legend(handles=legend_elements)
    ax.legend(loc='upper left',fontsize=12)
    # ax.xaxis.set_label_position('top')  # Movemos la etiqueta del eje x
    # ax.xaxis.tick_top()  # Movemos los ticks del eje x a la parte superior
    # ax.spines['top'].set_position(('axes', 1.0))  # Movemos la espina superior al tope de la figura
    # ax.spines['bottom'].set_position(('axes', -0.1))  # Ocultamos la espina inferior
    ax.set_xticklabels([f"{composer} {datos_composers[composer]['Birth_year']} " for i, composer in enumerate(datos_composers.keys())], rotation=90,fontsize=12)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()