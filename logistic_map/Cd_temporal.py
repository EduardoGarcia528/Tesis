"""
MULTIPLE C
""" 

import numpy as np
import matplotlib.pyplot as plt
from funciones import indice_J, permutation_entropy, angulos_alpha, entropia_shannon
from circular_gamma import gamma_index_jacobs_circular
from gamma_4 import gamma_index_jacobs

def logistic_map(r, x):
    return r * x * (1 - x)


def ruido_uniforme(size, var):
    # Ruido blanco uniforme centrado con varianza ~ var
    a = np.sqrt(var)
    return np.random.uniform(-a, a, size)


def bifurcacion_con_J(
    r_min=3.0,
    r_max=4.0,
    max_gamma=4,
    mu=20,
    resolucion_r=300,
    longitud_serie=2000,
    iter_descartar=4000,
    graficar=True
):
    colores =['#2ECC71','purple', '#FFD700','#1ABC9C','#E91E63']
    r_vals_plot = []
    x_vals_plot = []

    r_vals_J = []
    J_vals = np.zeros((max_gamma+2,resolucion_r+1))

    for r in np.sort(np.concatenate((np.linspace(r_min, r_max, resolucion_r), np.array([3.569945672])))):
        if r == 3.0:
            continue
        x = 0.6

        # --- Transitorio ---
        for _ in range(iter_descartar):
            x = np.clip(x, 0, 1)
            x = logistic_map(r, x)

        # --- Serie ---
        serie = np.empty(longitud_serie)
        for i in range(longitud_serie):
            x = np.clip(x, 0, 1)
            x = logistic_map(r, x)
            serie[i] = x

  
        # Guardar para bifurcación
        r_vals_plot.extend([r] * longitud_serie)
        x_vals_plot.extend(serie)

        # # MEDIDA
        angulos = angulos_alpha(serie,False)
        C, J = gamma_index_jacobs_circular(angulos,max_gamma,mu)
        J_vals[:,len(r_vals_J)] = C
        r_vals_J.append(r)

    if graficar:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bifurcación
        ax1.plot(r_vals_plot, x_vals_plot, ',', alpha=0.5)
        ax1.axvline(x=3.569945672, color='gray', linestyle='--', label=r'$r_\infty = 3.56994...$')
        ax1.set_xlabel('r')
        ax1.set_ylabel('x')
        ax1.set_xlim(r_min, r_max)
        ax1.set_ylim(0, 1)

        fig.tight_layout()  
        major_ticks_x = np.linspace(r_min, r_max, 6)  # 0.0, 0.2, ..., 1.0
        minor_ticks_x = np.linspace(r_min, r_max, 21)  # Ticks secundarios

        major_ticks_y = np.linspace(0, 1, 6)  # -1, -0.5, 0, 0.5, 1.0
        minor_ticks_y = np.linspace(0, 1, 21)  # Ticks secundarios

        # Configurar los ticks del eje X
        plt.xticks(major_ticks_x)  # Solo etiquetar los ticks principales
        plt.gca().set_xticks(minor_ticks_x, minor=True)  # Agregar ticks menores sin etiquetas

        # Configurar los ticks del eje Y
        plt.yticks(major_ticks_y)  # Solo etiquetar los ticks principales
        plt.gca().set_yticks(minor_ticks_y, minor=True)  # Agregar ticks menores sin etiquetas

        # Activar la cuadrícula
        plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks principales
        plt.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks secundarios


        # Índice J
        ax2 = ax1.twinx()
        ax2.invert_yaxis()
        for i in range(2,max_gamma+2):
            ax2.plot(r_vals_J, J_vals[i,:], color=colores[i-2] , alpha = 1, label= r'$C_d^\alpha$ = '+f'{i}')

        ax1.set_ylabel('x', rotation = 360)
        ax2.set_ylabel(r'$C_d^\alpha$', rotation= 360)
        ax2.set_ylim(1.0,0.0)

        ax2.legend(loc='lower left')
        plt.title(
            fr"Mapa logístico + $C_d^\alpha$"
        )
        plt.show()

    return np.array(r_vals_J), np.array(J_vals), np.array(r_vals_plot), np.array(x_vals_plot)


r_J, J, r_bif, x_bif = bifurcacion_con_J(
    r_min=3.4,
    r_max=4.0,
    max_gamma=5,
    mu=5,
    resolucion_r=300,
    longitud_serie=4000,
    iter_descartar=1000,
    graficar=True)