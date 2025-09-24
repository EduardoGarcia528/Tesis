import numpy as np
import matplotlib.pyplot as plt
import time

# ===================================================================
# PARÁMETROS DE LA SIMULACIÓN
# ===================================================================
a = 1.0       # Altura del objetivo para el movimiento browniano
T = 10000.0    # Tiempo total en segundos
dt = 0.01     # Paso de tiempo en segundos
n_steps = int(T / dt)
m = 10000   # Número de caminantes
times = np.arange(0, n_steps) * dt

# ===================================================================
# INICIALIZACIÓN
# ===================================================================
# Posiciones de los caminantes en la red 1D
x = np.zeros(m, dtype=int)
# Proceso browniano subyacente para cada caminante
y = np.zeros(m)
# Lista para guardar el historial del MSD
msd_record = np.zeros(n_steps)

# Historial de sitios visitados: una lista de sets.
# Cada caminante empieza con el sitio '0' en su historial.
#history = [set([0]) for _ in range(m)]
history = [[0] for _ in range(m)]
print(np.shape(history))

brownian_step = np.sqrt(2*dt)


start_time = time.time()

# ===================================================================
# BUCLE DE SIMULACIÓN PRINCIPAL
# ===================================================================
for t in range(n_steps):
    # 1. Actualizar el proceso browniano subyacente para todos
    y += brownian_step * np.random.randn(m)

    # 2. Encontrar qué caminantes deben moverse (índices)
    mask = (y >= a)
    mover_indices = np.where(mask)[0]

    # 3. Procesar cada caminante que debe moverse individualmente
    if len(mover_indices) > 0:
        for i in mover_indices:
            # --- NUEVA LÓGICA DE DECISIÓN ---
            # Lanzar una moneda: 50% regresar, 50% movimiento normal
            if np.random.rand() < 0.5:
                # ACCIÓN 1: Regresar a un sitio visitado
                # Convertimos el set a una lista para poder elegir un elemento
                #visited_sites = list(history[i])
                visited_sites = history[i]
                destination = np.random.choice(visited_sites)
                x[i] = destination
            else:
                # ACCIÓN 2: Movimiento normal (otro 50%)
                # Lanzar otra moneda: 50% derecha, 50% izquierda
                if np.random.rand() < 0.5:
                    x[i] += 1
                else:
                    x[i] -= 1
            
            # 4. Actualizar el historial y reiniciar el proceso 'y'
            history[i].append(x[i])
            y[i] = 0.0

    # 5. Guardar el MSD en el paso actual
    msd_record[t] = np.mean(x**2)

# ===================================================================
# FIN DE LA SIMULACIÓN Y ANÁLISIS
# ===================================================================
end_time = time.time()
print(f"Simulación completada en {end_time - start_time:.2f} segundos.")

# --- Calcular estadísticas finales ---
# Número promedio de sitios únicos que cada caminante visitó
avg_unique_sites = np.mean([len(h) for h in history])




print(f"Número promedio de sitios únicos visitados: {avg_unique_sites:.2f}")

# ===================================================================
# VISUALIZACIÓN DE RESULTADOS
# ===================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# --- Gráfico 1: Desplazamiento Cuadrático Medio (MSD) ---
axs[0].plot(times, msd_record, label='MSD Simulado')
# Para comparación, una línea recta que representa difusión normal (MSD ~ t)
axs[0].plot(times, times * msd_record[-1] / T, '--', color='red', alpha=0.7, label='Guía Difusión Normal (MSD ∝ t)')
axs[0].set_xlabel("Tiempo (s)")
axs[0].set_ylabel("Desplazamiento Cuadrático Medio (MSD)")
axs[0].set_title("Evolución del MSD (Escala Lineal)")
axs[0].legend()

# --- Gráfico 2: Distribución Final de Posiciones ---
axs[1].hist(x, bins=50, density=True, edgecolor='black', alpha=0.7)
axs[1].set_xlabel("Posición Final (x)")
axs[1].set_ylabel("Densidad de Probabilidad")
axs[1].set_title(f"Distribución de {m} Caminantes en t={T}s")

plt.tight_layout()
plt.show()