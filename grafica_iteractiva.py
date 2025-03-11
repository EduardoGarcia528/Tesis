import pandas as pd
import os
from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

def interpolador(subject, method, size):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    data = subject
    x = np.arange(len(data))
    
    # Crear 'size' puntos equidistantes
    x_new = np.linspace(0, len(data) - 1, size*(len(data)-1) + len(data))
    
    if method == 'lineal':
        data_interp = np.interp(x_new, x, data)
    elif method == 'herm':
        interpolator = PchipInterpolator(x, data)
        data_interp = interpolator(x_new)
    
    return x_new, data_interp

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import pickle

Ns = np.load('Ns.npy')
Js = np.load('J_composers.npy')
Jotas2 = np.load('Random/Jotas2.npy')
Jotas3 = np.load('Random/Jotas3.npy') #Jotas.npy contiene los 100 vectores de J hipotesis nula
Jotas4 = np.load('Random/Jotas4.npy')
Jotas5 = np.load('Random/Jotas5.npy')
Jotas6 = np.load('Random/Jotas6.npy')
Jotas7 = np.load('Random/Jotas7.npy')
# Jotas = np.zeros((100,len(range(20,14000,100))))
# print(np.shape(Jotas), len(range(20,14000,100)))
randoms = np.random.uniform(0, 1, (100, 14000))
# print(np.shape(np.mean([Jotas2,Jotas],axis=0)))
stack = np.vstack((Jotas2,Jotas3,Jotas4,Jotas5,Jotas6,Jotas7))
Jotas = np.mean([Jotas2,Jotas3,Jotas4,Jotas5,Jotas6,Jotas7],axis=0)
# for n,i in enumerate(range(20, 14000, 100)):
    # J_univariante_parcial = partial(J_univariante,tau=1,corte=False, j = i)
    # J_por_i = np.apply_along_axis(J_univariante_parcial, axis =1 , arr=randoms[:,:i])
    # Jotas[:,n] = J_por_i
    # print(i)
# # print(J_univariante(np.random.uniform(0.0,1.0, 5000), 1, False))
# np.save('Random/Jotas7.npy', Jotas)
x = np.arange(20, 14000, 100)
xx = np.arange(20, 13921, 1)
J_mean = np.mean(stack, axis=0)
J_std = np.std(stack, axis = 0)
J_minus = np.mean([np.min(Jotas2,axis=0),np.min(Jotas3,axis=0),np.min(Jotas4,axis=0),
 np.min(Jotas5,axis=0),np.min(Jotas6,axis=0),np.min(Jotas7,axis=0)], axis=0)
# J_minus = np.min(Jotas, axis=0)
# print(J_minus_intr[1,2000-20])
J_minus = interpolador(J_minus, 'lineal', 99)
J_minus_intr = np.zeros((2, 13901))
J_minus_intr[1,:] = J_minus[1]
J_minus_intr[0,:] = xx

import matplotlib.pyplot as plt
import mplcursors

# Tu código de datos aquí
# Graficar barras de error
plt.errorbar(x=x, y=J_mean, yerr=J_std, fmt='none', ecolor='black', capsize=2, label='std. dev.')

# Graficar curva promedio
plt.plot(x, J_mean, color='red', label='Promedio')

# Graficar curva J mínima
plt.plot(xx, J_minus[1], 'green', label='J mínima')

# Definir puntos personalizados con color y label
puntos_x = [260, 778, 506, 1516,5395, 16183,69,205,165,493,197,589]  # Ejemplo de coordenadas x
puntos_y = [0.94785, 0.86114, 0.97517, 0.94704, 0.9784986342991956,0.9522797435374722,0.8472476329920281,
            0.8552133766900794,0.9240593229803462, 0.8943252815067188,0.9951701389408945,0.9961072339787621]  #y
print(len(puntos_x),len(puntos_y))
puntos_labels = ['Detroit become human', 'Detroit become human', 'Webern_-_Variationen_Op._27',
                 'Webern_-_Variationen_Op._27','Concerto No. 2 in E flat major','Concerto No. 2 in E flat major',
                 'Elf_Kurze_Stcke_No._4','Elf_Kurze_Stcke_No._4','Op.4, No.3','Op.4, No.3',
                 'Entflieht_auf_leichten_Kaehnen__Op_2','Entflieht_auf_leichten_Kaehnen__Op_2']
puntos_colors = ['blue', 'blue', 'orange','orange','red','red','black','black','green','green','gray','gray']

# Agregar los puntos a la gráfica
sc = plt.scatter(puntos_x, puntos_y, c=puntos_colors, label='Puntos etiquetados', zorder=3)

# Crear cursores interactivos con labels
cursor = mplcursors.cursor(sc, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(puntos_labels[sel.index]))

# Configurar ejes y leyenda
plt.xlabel('N/2 tamaño de la serie')
plt.ylabel('J')
plt.xlim(0, 16183)
plt.legend()
plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)

# Mostrar la gráfica
plt.show()