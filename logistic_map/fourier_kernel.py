import matplotlib.pyplot as plt
import mi_libreria as nl
import numpy as np

# log = nl.logistic_map(4.0,10000,1000)
# log = nl.iaaft(log,1)[0,:]
log = nl.colored_noise(1_000_000,color="white")
theta1 = nl.angulos_alpha(log, False)
theta = np.mod(theta1, 2 * np.pi)
H2 = nl.entropia_shannon(theta, discreto=False) # bins regla de Sturges
print("Entropía de Shannon bins:", H2)



N = len(theta)
M = int(np.sqrt(N))


n = np.arange(1, M + 1)[:, None]
theta_row = theta[None, :]

c = np.mean(np.exp(1j * n * theta_row), axis=1)
sigma = M / 2
weights = np.exp(-(np.arange(1, M + 1) ** 2) / (2 * sigma**2))
c = c * weights


theta_grid = np.linspace(0, 2 * np.pi, 500)
theta_grid_row = theta_grid[None, :]

f_theta = np.ones_like(theta_grid) / (2 * np.pi)

f_theta += (1 / np.pi) * np.sum(
    np.real(c[:, None] * np.exp(-1j * n * theta_grid_row)), axis=0
)

f_theta = np.maximum(f_theta, 0)
f_theta /= np.trapz(f_theta, theta_grid)


plt.figure(figsize=(8, 5))
plt.hist(theta1, bins=60, density=True, alpha=0.3, label="Datos")
plt.plot(theta_grid, f_theta, "r", lw=2, label="Fourier")
plt.legend()
plt.xlabel("Ángulo (rad)")
plt.ylabel("Densidad")
plt.title("Reconstrucción con serie de Fourier")
plt.show()

power = np.abs(c) ** 2
p = power / np.sum(power)
H_spectral = -np.sum(p * np.log(p))

M = len(p)
H_norm = H_spectral / np.log(M)

print("Entropía espectral normalizada:", H_norm)
print("indiceJ: ", nl.indice_J(log, False))
print("SE de kernel: ", nl.entropia_shannon(f_theta, discreto=False))