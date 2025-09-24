import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def henon_map(a, b, x0=0.1, y0=0.1, n_trans=200, n_points=10000):
    x, y = x0, y0
    # Transitorio
    for _ in range(n_trans):
        x, y = 1 - a * x * x + y, b * x

    # Iteraciones para graficar
    xs = []
    ys = []
    for _ in range(n_points):
        x, y = 1 - a * x * x + y, b * x
        xs.append(x)
        ys.append(y)

    return xs, ys


def plot_henon_map(x, y):
    # plt.figure(figsize=(8, 8))
    plt.plot(x, y,',')
    plt.title("Atractor de Henon a=1.4, b=0.3")
    plt.xlabel(r"$x_{n}$")
    plt.ylabel(r"$x_{n+1}$")
    plt.grid(True)
    plt.show()

def lyapunov_exponent_from_henon_orbit(xs, ys, a, b):
    # Vectores de perturbación iniciales
    v1, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])  

    sum_ln1, sum_ln2 = 0.0, 0.0  # Sumas para los exponentes de Lyapunov
    n = len(xs)
    
    for i in range(n):
        # Derivadas parciales del mapa de Henon
        dfdx = -2 * a * xs[i]
        dfdy = 1
        dgdx = b
        dgdy = 0
        
        # Matriz jacobiana del mapa de Henon
        J = np.array([[dfdx, dfdy],
                      [dgdx, dgdy]])
        
        # Aplicar la matriz jacobiana a los vectores de perturbación
        v1 = np.dot(J, v1)
        v2 = np.dot(J, v2)

        # Gram-Schmidt ortogonalización
        v1_norm = np.linalg.norm(v1)
        v1 = v1 / v1_norm
        
        v2_proj = np.dot(v1, v2) * v1  # Proyección de v2 sobre v1
        v2 = v2 - v2_proj
        v2_norm = np.linalg.norm(v2)
        v2 = v2 / v2_norm
        
        # Actualizar las sumas de los logaritmos de los cocientes
        sum_ln1 += np.log(v1_norm)
        sum_ln2 += np.log(v2_norm)

    # Cálculo de los exponentes de Lyapunov
    le1 = sum_ln1 / n
    le2 = sum_ln2 / n
    return max(le1, le2)


def plot_orbit_diagram(graficar, a_min, a_max,num_points_per_a):
    
    a_array = np.linspace(a_min, a_max, num_points_per_a)
    orbit_ejex= []
    x_array = []
    y_array = []

    for a in a_array:
        x0 = 0.1
        y0 = 0.1
        x, y = henon_map(a, b=0.3, x0=x0, y0=y0,n_trans=10000, n_points=1000)
        x_array.extend(x)
        y_array.extend(y)
        orbit_ejex.extend([a] * len(x))
        
    #A partir de aqui, orbita continua de logistica completada
        
    z = 0
    lyapunov_values = []

    for i in range(0, len(orbit_ejex) - 1):
        if orbit_ejex[i] != orbit_ejex[i+1]:
            print(orbit_ejex[i])
            a_single_orbitx = x_array[z:i+1]
            a_single_orbity = y_array[z:i+1]
            z = i+1

            lyapunov = lyapunov_exponent_from_henon_orbit(a_single_orbitx, a_single_orbity, orbit_ejex[i], 0.3)
            lyapunov_values.append(lyapunov)

    #ultimo valor de r
    lyapunov = lyapunov_exponent_from_henon_orbit(x_array[z:],y_array[z:], orbit_ejex[len(orbit_ejex) - 1],0.3)  
    lyapunov_values.append(lyapunov)

    
    if graficar == True:
        fig, ax1 = plt.subplots(figsize=(10,6))#figsize=(10,6)
        ax1.plot(orbit_ejex, x_array, ',', label='Bifurcación de la órbita', alpha=1)
                            
        ax1.set_xlabel('a') #.9873
        ax1.set_ylabel('x', color='blue', rotation = 360)
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='center left', bbox_to_anchor=(0.1, 0.25), framealpha=0.5)
        ax1.set_xlim(a_min, a_max)

        
        major_ticks_x = np.linspace(a_min, a_max, 6)  # 0.0, 0.2, ..., 1.0
        minor_ticks_x = np.linspace(a_min, a_max, 21)  # Ticks secundarios

        major_ticks_y = np.linspace(np.min(x_array), np.max(x_array), 6)  # -1, -0.5, 0, 0.5, 1.0
        minor_ticks_y = np.linspace(np.min(x_array),np.max(x_array) , 21)  # Ticks secundarios

        # Configurar los ticks del eje X
        plt.xticks(major_ticks_x)  # Solo etiquetar los ticks principales
        plt.gca().set_xticks(minor_ticks_x, minor=True)  # Agregar ticks menores sin etiquetas

        # Configurar los ticks del eje Y
        plt.yticks(major_ticks_y)  # Solo etiquetar los ticks principales
        plt.gca().set_yticks(minor_ticks_y, minor=True)  # Agregar ticks menores sin etiquetas

        # Activar la cuadrícula
        plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks principales
        plt.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5)         

        if True:
            ax2 = ax1.twinx()
            ax2.axhline(y=0, color='black', linestyle='--', alpha =0.55)
            ax2.plot(a_array, lyapunov_values, 'black', label = 'λ')
            ax2.set_ylabel('λ', color='black', rotation = 360)
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='center right',framealpha=0.5)
            # ax2.set_xlim(3.571906, 4.0)
        
        fig.tight_layout()  
        plt.show()

    return a_array, lyapunov_values



def feigenbaum(x):
    for n in range(1, 2):
        delta_n = (x[n] - x[n-1]) / (x[n+1] - x[n])
        print(f"Feigenbaum ratio for n={n}: {delta_n}")


if __name__ == "__main__":

    #Graficar atractor de Henon
    x, y = henon_map(a=1.4, b=0.3, x0=0.1, y0=0.1,n_trans=10000, n_points=100_000)
    plot_henon_map(x[1:], x[:-1])

    # #Diagrama de bifurcacion
    a_array, lyapunov_values = plot_orbit_diagram(graficar=True, a_min = 1.0, a_max = 1.4,num_points_per_a=1000)


    # Feigenbaum constants
    # a_array, lyapunov_values = plot_orbit_diagram(graficar=True, a_min = 0.35, a_max = 1.03,num_points_per_a=100_000)
    # np.save('henon_bifurcation_data.npy', (a_array, lyapunov_values))
    a_array, lyapunov_values = np.load('henon_bifurcation_data.npy', allow_pickle=True)
    arr = np.array(lyapunov_values)
    peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1
    x = np.array(a_array)
    peaks = peaks[arr[peaks] >= -0.3]
    # # print("Indices de los picos:", peaks)  
    # # print("Valores de re los picos:", x[peaks])
    feigenbaum(x[peaks])

"""
Feigenbaum ratio for n=1: 4.807236354655469
Real value: 4.6692016
"""