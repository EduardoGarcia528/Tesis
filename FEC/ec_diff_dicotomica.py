import numpy as np
import matplotlib.pyplot as plt

# Parámetros del proceso
v = 1.0       # velocidad (magnitud)
lam = 0.5     # tasa de salto
Tmax = 10.0   # tiempo máximo
dt = 0.01     # paso temporal
Ntraj = 5000  # número de trayectorias para el promedio

# Número de pasos
Nsteps = int(Tmax / dt)

# Inicialización
x = np.zeros((Ntraj, Nsteps))
vstate = np.random.choice([+v, -v], size=Ntraj)  # estado inicial aleatorio

# Simulación del proceso dicotómico
for t in range(1, Nsteps):
    # Con probabilidad lam*dt, cambia el signo de la velocidad
    flip = np.random.rand(Ntraj) < lam * dt
    vstate[flip] *= -1
    # Actualizamos posición
    x[:, t] = x[:, t-1] + vstate * dt

# Calculamos <x^2(t)>
mean_x2 = np.mean(x**2, axis=0)
time = np.linspace(0, Tmax, Nsteps)

# Solución analítica
x2_theory = (v**2 / lam) * (time - (1 - np.exp(-2*lam*time)) / (2*lam))

# Graficamos
plt.figure(figsize=(8,5))
plt.plot(time, mean_x2, label="Simulación Monte Carlo", alpha=0.7)
plt.plot(time, x2_theory, 'r--', label="Solución analítica", linewidth=2)
plt.xlabel("t")
plt.ylabel(r"$\langle x^2(t) \rangle$")
plt.title("Segundo momento en el proceso dicotómico de difusión")
plt.legend()
plt.grid(True)
plt.show()
