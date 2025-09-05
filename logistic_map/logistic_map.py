import matplotlib.pyplot as plt
import os
import numpy as np
from numba import njit

@njit
def logistic_map(r, x):
    return r * x * (1 - x)


@njit
def lyapunov_exponent_from_orbit(orbit, r):
    lyapunov_sum = 0

    for x in orbit:
        # Derivada de la función logística
        derivative = abs(r * (1 - 2 * x))
        if derivative == 0:
            lyapunov_sum += 0
        else:
            lyapunov_sum += np.log(derivative)

    # Calcular el exponente de Lyapunov
    lyapunov_exponent = lyapunov_sum / len(orbit)
    if lyapunov_exponent == 0:
        return np.nan
    return lyapunov_exponent


def plot_orbit_diagram(graficar=True, r_min = 2.99, r_max = 3.571, num_points_per_r=50000,
 num_iterations_discard=300, num_iterations_display=5000):
    
    r_values = []
    orbit_values = []
    
    for r in np.linspace(r_min, r_max, num_points_per_r):
        x = 0.1
        for _ in range(num_iterations_discard): 
            x = logistic_map(r, x) 
        for _ in range(num_iterations_display):
            x = logistic_map(r, x) 
            r_values.append(r)
            orbit_values.append(x)
        
    #A partir de aqui, orbita continua de logistica completada
        
    a = 0
    J_index=[]
    lyapunov_values = []
    b=0

    for i in range(0, len(r_values) - 1):
        if r_values[i] != r_values[i+1]:
            r_single_orbit = orbit_values[a:i+1]
            a = i+1

            lyapunov = lyapunov_exponent_from_orbit(r_single_orbit, r_values[i])
            lyapunov_values.append(lyapunov)
            J_index.append(r_values[i])
            b += 1

    #ultimo valor de r
    lyapunov = lyapunov_exponent_from_orbit(orbit_values[a:], r_values[len(r_values) - 1])  
    lyapunov_values.append(lyapunov)
    J_index.append(r_values[len(r_values) - 1])

    
    if graficar == True:
        fig, ax1 = plt.subplots(figsize=(10,6))#figsize=(10,6)
        ax1.plot(r_values, orbit_values, ',', label='Bifurcación de la órbita', alpha=1)
                            
        ax1.set_xlabel('r') #.9873
        ax1.set_ylabel('x', color='blue', rotation = 360)
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='center left', bbox_to_anchor=(0.1, 0.25), framealpha=0.5)
        ax1.set_xlim(1.0, 4.0)

        
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
        plt.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5)         

        if True:
            ax2 = ax1.twinx()
            ax2.axhline(y=0, color='black', linestyle='--', alpha =0.55)
            ax2.plot(J_index, lyapunov_values, 'black', label = 'λ')
            ax2.set_ylabel('λ', color='black', rotation = 360)
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='center right',framealpha=0.5)
            # ax2.set_xlim(3.571906, 4.0)
        
        fig.tight_layout()  
        plt.show()

    return J_index, lyapunov_values

def feigenbaum(x):
    for n in range(2, len(x)-1):
        if x[n+1] > 3.5699:
            continue
        delta_n = (x[n] - x[n-1]) / (x[n+1] - x[n])
        print(f"Feigenbaum ratio for n={n}: {delta_n}")

# Diagrama de órbitas y exponente de Lyapunov
plot_orbit_diagram(graficar=True, r_min = 1.0, r_max = 4.0, num_points_per_r=1000)

# Espacio de fases (Cobweb Plot)
r = 4.0
x = np.linspace(0, 1, 1000)
plt.plot(x, logistic_map(4.0, x))
plt.plot(x, x, '--', label=r'$x(n+1)=x(n)$')
xn = 0.1
for i in range(100):
    x_next = logistic_map(r,xn)
    # línea vertical: (xn, xn) -> (xn, x_next)
    plt.plot([xn, xn], [xn, x_next], color='red', linewidth=1)
    # línea horizontal: (xn, x_next) -> (x_next, x_next)
    plt.plot([xn, x_next], [x_next, x_next], color='red', linewidth=1)
    xn = x_next
plt.title("Logistic Map: Cobweb Plot")
plt.xlabel("x(n)")
plt.ylabel("x(n+1)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()


# Sensibilidad a las condiciones iniciales
r = 4.0
for i in [0,10e-2,10e-3,10e-6,10e-8]:
    orbit = []
    x0 = 0.1 + i
    for _ in range(1000):
        orbit.append(x0)
        x0 = logistic_map(4.0, x0)
    if i == 0:
        orbit0 = orbit
    else:
        plt.plot(range(len(orbit)), np.abs(np.array(orbit)-np.array(orbit0)), label=r"$\Delta x$" + f"= {i}")
plt.title("Logistic Map: x(n) vs n")
plt.xlabel("n")
plt.ylabel("x(n)")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()


# Feigenbaum constants
x, lambd = plot_orbit_diagram(graficar=False, r_min = 2.99, r_max = 3.571, num_points_per_r=50000,
 num_iterations_discard=300, num_iterations_display=5000)

arr = np.array(lambd)
peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1
x = np.array(x)

# # print("Indices de los picos:", peaks)  
# # print("Valores de re los picos:", x[peaks])

feigenbaum(x[peaks])

"""
Feigenbaum ratio for n=2: 4.655803316180737
Feigenbaum ratio for n=3: 4.664000000000044
Feigenbaum ratio for n=4: 4.687499999998268
Feigenbaum ratio for n=5: 4.705882352943028
"""
