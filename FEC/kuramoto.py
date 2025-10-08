import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 500              # número de osciladores
dt = 0.01            # paso de integración
tmax = 200_000           # tiempo total
sigma = 1.0          # desviación estándar de frecuencias naturales
Kc = 2 * sigma * np.sqrt(2/np.pi)  # acoplamiento crítico teórico
K = Kc                # valor de K en el umbral

# Inicialización
omega = np.random.normal(0, sigma, N)      # frecuencias naturales
theta = np.random.uniform(0, 2*np.pi, N)   # fases iniciales
nsteps = int(tmax / dt)

# Guardar resultados
R_values = np.zeros(nsteps)

# Simulación
for step in range(nsteps):
    # Calcular parámetro de orden
    re = np.mean(np.cos(theta))
    im = np.mean(np.sin(theta))
    R = np.sqrt(re**2 + im**2)
    psi = np.arctan2(im, re)
    R_values[step] = R

    # Ecuación de Kuramoto
    theta += dt * (omega + K * R * np.sin(psi - theta))

# np.save("kuramoto_Rc.npy", R_values[100_000:])


t = np.linspace(0, tmax, nsteps)
plt.figure(figsize=(8,4))
plt.plot(t, R_values, lw=1)
plt.xlabel("Tiempo")
plt.ylabel("Parámetro de orden R(t)")
plt.title(f"Modelo de Kuramoto en Kc ≈ {Kc:.3f}")
plt.grid(True)
plt.show()
