import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, cpu_count

def inicializar_ising(L):
    """Inicializa una red LxL de espines +1 o -1 aleatorios"""
    return np.random.choice([1, -1], size=(L, L))

def energia_local(red, i, j):
    """Calcula la energía local de un espín en (i,j)"""
    L = red.shape[0]
    arriba = red[(i-1)%L, j]
    abajo = red[(i+1)%L, j]
    izquierda = red[i, (j-1)%L]
    derecha = red[i, (j+1)%L]
    return -red[i,j] * (arriba + abajo + izquierda + derecha)

def paso_monte_carlo(red, T):
    """Un paso de Monte Carlo: recorrer cada sitio una vez en promedio"""
    L = red.shape[0]
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = -2 * energia_local(red, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            red[i,j] *= -1
    return red

def simular_ising(L, T, pasos):
    """Corre la simulación y regresa la serie de magnetización"""
    red = inicializar_ising(L)
    M = []
    octavos = 1
    for paso in range(pasos):
        if paso == octavos*pasos//8:
            print("Octavo!")
            octavos += 1
        red = paso_monte_carlo(red, T)
        magnetizacion = np.sum(red) / (L * L)
        M.append(magnetizacion)
    return np.array(M), red

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
            return (i, J)
    return None

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

    # Filtramos resultados válidos
    indices_activas = []
    J_local = []
    for r in resultados:
        if r is not None:
            i, J = r
            indices_activas.append(i)
            J_local.append(J)

    df_J = pd.DataFrame({
        "inicio_ventana": indices_activas,
        "J_s0": J_local
    })

    return df_J

def logistic_map(r, x):
    return r * x * (1 - x)

# --- Parámetros ---
if __name__ == '__main__':
    L = 128
    T = 2.269185
    pasos = 1000000

    # --- Simular ---
    # M_series, red_final = simular_ising(L, T, pasos)
    M_series = np.load('M_series_Tc.npy')

    # --- Graficar serie ---
    plt.figure(figsize=(8, 4))
    plt.plot(M_series, lw=0.7)
    np.save('M_series_Tc.npy',M_series)
    plt.xlabel('Paso Monte Carlo')
    plt.ylabel('Magnetización')
    plt.title(f'Ising 2D L={L}, T={T}')
    plt.grid(True)
    plt.show()

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
    resultado = detector_ventanas_criticas_parallel(M_series, window_size=500, umbral_percentil=95, paso=100)

    # Agrupamiento para graficar
    n_prom = 100
    prom_J = resultado["J_s0"].groupby(np.arange(len(resultado)) // n_prom).mean()
    prom_indices = resultado["inicio_ventana"].groupby(np.arange(len(resultado)) // n_prom).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(prom_indices, prom_J, label="Promedio J (s=0)", marker='o')
    plt.xlabel("Índice promedio de ventana")
    plt.ylabel("J")
    plt.title("J promedio en ventanas críticas")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()