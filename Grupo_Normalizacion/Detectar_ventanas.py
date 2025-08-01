import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, cpu_count
from numba import njit
from scipy.signal import periodogram

def generar_uniforme_centrada(n, varianza):
    # Calcular el límite superior e inferior de la distribución uniforme
    limite = np.sqrt(varianza) # varianza*3
    # Generar n números aleatorios con distribución uniforme entre -limite y limite
    return np.random.uniform(-limite, limite, n)

def J_univariante(X, tau=1):
    X = np.array(X)
    x1 = X[tau:]
    y1 = X[:-tau]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))
    vectores = []
    for i in range(len(ff1) - 1):
        p1 = [ff1[i], ff2[i]]
        p2 = [ff1[i + 1], ff2[i + 1]]
        cuadrante = [
            [p2[0] - p1[0], p2[1] - p1[1]],
            [p2[0] - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
        ]
        distancias = np.array([np.linalg.norm(np.array(c) - np.array(p1)) for c in cuadrante])
        p2 = cuadrante[np.argmin(distancias)]
        vectores.append([p2[0] - p1[0], p2[1] - p1[1]])

    vectores = np.array(vectores)
    norms = np.linalg.norm(vectores, axis=1, keepdims=True)
    v_norm = np.where(norms == 0, vectores, vectores / norms)
    
    angulos = np.arccos(np.clip(np.einsum('ij,ij->i', v_norm[:-1], v_norm[1:]), -1.0, 1.0))
    cruces = np.cross(v_norm[:-1], v_norm[1:])
    angulos = np.where(cruces > 0, np.pi - angulos, angulos)
    angulos = np.where((cruces == 0) & (angulos < 0), np.pi, angulos)
    angulos = np.where(cruces < 0, angulos + np.pi, angulos)

    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    
    return J

def analizar_ventana(args):
    """Función auxiliar para procesar una ventana"""
    ventana, i, umbral = args
    if np.max(np.abs(np.gradient(ventana))) > umbral:
        if len(ventana) > 10:
            J = J_univariante(ventana, tau=1)
            return J
    return np.nan

def detector_ventanas_criticas_parallel(X, window_size, umbral_percentil, paso):
    X = np.array(X)
    grad = np.abs(np.gradient(X))
    umbral = np.percentile(grad, umbral_percentil)

    # Preparamos las ventanas
    tareas = [
        (X[i:i + window_size], i, umbral)
        for i in range(0, len(X) - window_size, paso)
    ]

    # Procesamos en paralelo
    with Pool(processes=cpu_count()) as pool:
        resultados = pool.map(analizar_ventana, tareas)

    resultados = np.array(resultados)
    return resultados[~np.isnan(resultados)]

def periodicidad(signal, fs = 1.0):
    f , Pxx = periodogram(signal, fs = fs)
    if len(Pxx) <= 1:
        return 0.0
    Pxx = Pxx[1:]
    total_energy = np.sum(Pxx)
    if total_energy == 0:
        return 0.0
    max_energy = np.max(Pxx)
    return max_energy/total_energy

def logistic_map(r, x):
    return r * x * (1 - x)

def plot_orbit_diagram(graficar=True, r_min = 3.45, r_max = 3.6, num_points_per_r=300,
 num_iterations_discard=1000, num_iterations_display=100_000):

    r_values = []
    orbit_values = []
    for r in np.concatenate((np.linspace(3.45,3.56994,200),np.linspace(3.56994,3.6,100)[1:])):
        
        x = 0.6

        for _ in range(num_iterations_discard):
            x = np.clip(x, 0.0, 1.0)
            x = logistic_map(r, x) + generar_uniforme_centrada(1, 1e-10)[0]
        for _ in range(num_iterations_display):
            x = np.clip(x, 0.0, 1.0)
            x = logistic_map(r, x) + generar_uniforme_centrada(1, 1e-10)[0]
            r_values.append(r)
            orbit_values.append(x)
    return orbit_values, r_values

    
def plotear(orbit_values, r_values):    
    #A partir de aqui, orbita continua de logistica completada
        
    a = 0
    J_values=[]
    J_index=[]
    lyapunov_values = []

    for i in range(0, len(r_values) - 1):
        if r_values[i] != r_values[i+1]:
            r_single_orbit = orbit_values[a:i+1]
            a = i+1
            J_ventanas = detector_ventanas_criticas_parallel(r_single_orbit,window_size=100, umbral_percentil=95, paso=100)
            # J = J_univariante(J_ventanas)
            J_values.append(periodicidad(J_ventanas))
            print(r_values[i])
            # J_values.append(J)
            J_index.append(r_values[i])
    
    #ultimo valor de r
    r_single_orbit = orbit_values[a:]
    
    J_ventanas = detector_ventanas_criticas_parallel(r_single_orbit, window_size=100, umbral_percentil=95, paso=100)
    # J = J_univariante(J_ventanas)
    J_values.append(periodicidad(J_ventanas))
    # J_values.append(J)
    J_index.append(r_values[len(r_values) - 1])

    #A partir de aqui, lyapunob y J fueron calculados
    np.save('J_ventanas_criticas_periodicidad.npy',J_values)
    # J_values = np.load('J_ventanas_criticas_logistic_r_critico_transit.npy')
    if True:
        fig, ax1 = plt.subplots(figsize=(10,6))
        
    
        ax1.plot(J_index, J_values, color='red', label='J' , alpha = 1)
        ax1.plot(r_values, orbit_values, ',', label='Bifurcación de la órbita', alpha=1)
        ax1.set_xlabel('r')
        ax1.set_ylabel('J', color='r', rotation = 360)
        ax1.tick_params(axis='y', labelcolor='r')
        # ax1.set_ylim(0,1)
        ax1.legend(loc = 'upper left')
        ax1.grid()
    
        
        fig.tight_layout()  
        major_ticks_x = np.linspace(3.45, 3.6, 6)  # 0.0, 0.2, ..., 1.0
        minor_ticks_x = np.linspace(3.45, 3.6, 21)  # Ticks secundarios

        major_ticks_y = np.linspace(0, 1, 6)  # -1, -0.5, 0, 0.5, 1.0
        minor_ticks_y = np.linspace(0, 1, 6)  # Ticks secundarios

        # Configurar los ticks del eje X
        plt.xticks(major_ticks_x)  # Solo etiquetar los ticks principales
        plt.gca().set_xticks(minor_ticks_x, minor=True)  # Agregar ticks menores sin etiquetas

        # Configurar los ticks del eje Y
        plt.yticks(major_ticks_y)  # Solo etiquetar los ticks principales
        plt.gca().set_yticks(minor_ticks_y, minor=True)  # Agregar ticks menores sin etiquetas

        # Activar la cuadrícula
        plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks principales
        plt.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5) 
        
        # if True:
        #     ax2 = ax1.twinx()
        #     ax2.axhline(y=0, color='black', linestyle='--', alpha =0.55)
        #     ax2.plot(J_index, lyapunov_values, 'black', label = 'λ')
        #     ax2.set_ylabel('λ', color='black', rotation = 360)
        #     ax2.tick_params(axis='y', labelcolor='black')
        #     ax2.legend(loc='center right',framealpha=0.5)

        plt.show()

    
    return J_values
    

# --- Parámetros ---
if __name__ == '__main__':

    # --- Simular ---
    # M_series = np.load('magnetization_time_series.npy')
    # M_series = np.load('M_series_Tc.npy')

    # --- Graficar serie ---
    # plt.figure(figsize=(8, 4))
    # plt.plot(M_series, lw=0.7)
    # np.save('M_series_Tc.npy',M_series)
    # plt.xlabel('Paso Monte Carlo')
    # plt.ylabel('Magnetización')
    # plt.title(f'Ising 2D L={32}, beta={0.4407}')
    # plt.grid(True)
    # plt.show()

    # # --- (Opcional) Mostrar configuración final ---
    # plt.imshow(red_final, cmap='coolwarm')
    # plt.title('Configuración final de espines')
    # plt.colorbar(label='Espín')
    # plt.show()

    # --- Detector con multiprocess ---
    # r = 3.56994
    # x = 0.5
    # orbit_values = []
    # # for _ in range(0): # Converger
    # #         # x = np.clip(x, 0.0, 1.0)
    # #     x = logistic_map(r, x) 
    # for _ in range(1000000):
    #         # x = np.clip(x, 0.0, 1.0)
    #     x = logistic_map(r, x) 
    #     orbit_values.append(x)
    orbit_values, r_values = plot_orbit_diagram()
    print('ya')
    J_values = plotear(orbit_values,r_values)

    # Agrupamiento para graficar
    # n_prom = 100
    # prom_J = resultado["J_s0"].groupby(np.arange(len(resultado)) // n_prom).mean()
    # prom_indices = resultado["inicio_ventana"].groupby(np.arange(len(resultado)) // n_prom).mean()

    # plt.figure(figsize=(10, 5))
    # plt.plot(prom_indices, prom_J, label="Promedio J (s=0)", marker='o')
    # plt.xlabel("Índice promedio de ventana")
    # plt.ylabel("J")
    # plt.title("J promedio en ventanas críticas")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()