import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit
import mi_libreria as ml
from scipy.stats import pearsonr, spearmanr

@njit
def logistic_map(r, x):
    return r * x * (1 - x)


def ruido_uniforme(size, var):
    # Ruido blanco uniforme centrado con varianza ~ var
    a = np.sqrt(var)
    return np.random.uniform(-a, a, size)


def bifurcacion_con_J(
    r_min=3.0,
    r_max=4.0,
    max_gamma=5,
    mu=3,
    resolucion_r=300,
    longitud_serie=20_000,
    iter_descartar=1000,
    var_ruido=1e-7,
    tipo_ruido="no",  # "iterativo" o "aditivo"
    graficar=True
):
    r_vals_J = np.sort(np.concatenate((np.linspace(r_min, r_max, resolucion_r), np.array([3.569945672]))))
    if r_vals_J[0] == 3.0:
        r_vals_J = r_vals_J[1:]
    r_vals_plot = []
    x_vals_plot = []
    
    J_vals = np.zeros((len(r_vals_J)))
    J_vals_null = np.zeros((len(r_vals_J)))
    for V in range(1):
        print(V)
        for q,r in enumerate(r_vals_J):
            x = 0.6

            # --- Transitorio ---
            for _ in range(iter_descartar):
                x = np.clip(x, 0, 1)
                x = logistic_map(r, x)
                if tipo_ruido == "iterativo":
                    x += ruido_uniforme(1, var_ruido)[0]

            # --- Serie ---
            serie = np.empty(longitud_serie)
            for i in range(longitud_serie):
                x = np.clip(x, 0, 1)
                x = logistic_map(r, x)
                if tipo_ruido == "iterativo":
                    x += ruido_uniforme(1, var_ruido)[0]
                serie[i] = x

            if tipo_ruido == "aditivo":
                serie += ruido_uniforme(longitud_serie, var_ruido)

            # Guardar para bifurcación
            if V == 0:
                r_vals_plot.extend([r] * longitud_serie)
                x_vals_plot.extend(serie)

            # # Índice J
            J = ml.indice_H(serie,None,tau=3)
            # _,J = gamma_index_jacobs(serie, max_gamma, mu=mu)
            # _, J = gamma_index_rank(serie, max_gamma, mu=mu)
            J_vals[q] = J_vals[q] + J  
            J_vals_null[q] = J_vals_null[q] + ml.indice_H(serie,None,tau=3,null="shuffle")
    # J_vals = J_vals / 500.0  
        
    # np.save("J_transicion5.npy", np.array([r_vals_J, J_vals]))
    print("Pearson:", pearsonr(J_vals_null, J_vals))
    print("Spearman:", spearmanr(J_vals_null, J_vals))
    if graficar:
        fig, ax1 = plt.subplots(figsize=(10, 6))


        # Índice J
        ax1.plot(r_vals_plot, x_vals_plot, '.', ms=0.1, alpha=0.2)
        ax1.set_xlabel('r')
        ax1.set_ylabel('x',rotation=360)
        ax1.set_xlim(r_min, r_max)
        ax1.set_ylim(0, 1)
        
        fig.tight_layout()  
        major_ticks_x = np.linspace(r_min, r_max, 6)  # 0.0, 0.2, ..., 1.0
        minor_ticks_x = np.linspace(r_min, r_max, 21)  # Ticks secundarios

        major_ticks_y = np.linspace(0.0, 1.0, 6)  # -1, -0.5, 0, 0.5, 1.0
        minor_ticks_y = np.linspace(0.0, 1.0, 21)  # Ticks secundarios

        # Configurar los ticks del eje X
        plt.xticks(major_ticks_x)  # Solo etiquetar los ticks principales
        plt.gca().set_xticks(minor_ticks_x, minor=True)  # Agregar ticks menores sin etiquetas

        # Configurar los ticks del eje Y
        plt.yticks(major_ticks_y)  # Solo etiquetar los ticks principales
        plt.gca().set_yticks(minor_ticks_y, minor=True)  # Agregar ticks menores sin etiquetas

        # Activar la cuadrícula
        plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks principales
        plt.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks secundarios



        # Bifurcación
        ax2 = ax1.twinx()
        ax2.axvline(x=3.569945672, color='gray', linestyle='--', label=r'$r_\infty = 3.56994...$')
        ax2.plot(r_vals_J, J_vals, color='red',marker='.', lw=0.5, ms = 2.0,label=r'$J_{\theta}$')
        ax2.plot(r_vals_J, J_vals_null, color='blue',marker='.', lw=0.5, ms = 2.0,label=r'$J_{\theta ,null}$')
        # ax2.invert_yaxis()
        # U = np.log(longitud_serie-(mu-1)*max_gamma)/np.log(math.factorial(mu))
        # ax2.axhline(y=U, color='red', linestyle='--', label=r'$U(N,m)$') 
        ax2.set_xlabel('r')
        ax2.set_ylabel('J',rotation=360)
        ax2.legend(loc='lower left')
        ax2.set_xlim(r_min, r_max)
        ax2.set_ylim(0, 1)

        # plt.title(
            # f"Mapeo logístico +"+ r"$\gamma^\alpha$"
            # f"Ruido {tipo_ruido}, varianza = {var_ruido}"
        # )

        plt.tight_layout()
        plt.show()

    return np.array(r_vals_J), np.array(J_vals), np.array(r_vals_plot), np.array(x_vals_plot)


r_J, J, r_bif, x_bif = bifurcacion_con_J(
    r_min=3.45, 
    r_max=4.0,
    # r_min=3.5695, 
    # r_max=3.5702,
    max_gamma=1,
    mu=4.0,  
    resolucion_r=300, 
    longitud_serie=10000,  
    iter_descartar=1000,
    var_ruido=1e-05,
    tipo_ruido="no",  # o "aditivo"
    graficar=True)