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
from funciones import interpolador, entropia_shannon, indice_J, angulos_alpha, permutation_entropy


def diff_S(d, angulos):
    if d == 0:
        return entropia_shannon(angulos,discreto=False, bins = 100)
    for _ in range(d):
        angulos = np.diff(angulos)

    return entropia_shannon(angulos)

def main(array,d):
    angulos = angulos_alpha(array,False)
    # angulos = np.load('henon_C_1000diff.npy')
    entropia = diff_S(d,angulos=angulos)
    J = indice_J(array,False)
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
            J = entropia_shannon(f,discreto=True)
            Js[x,y] = J
            lenght = y
        print(composer)
        np.save(f'new_data/shannon/{birth_year}_{composer}_shannon.npy', Js[x,:lenght+1])
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