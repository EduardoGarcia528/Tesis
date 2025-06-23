import numpy as np

def ising_2d(L, T, n_iter=1000):
    """Simula el modelo de Ising 2D con algoritmo de Metropolis."""
    # Inicialización aleatoria
    config = np.random.choice([-1, 1], size=(L, L))

    for _ in range(n_iter * L * L):
        i, j = np.random.randint(0, L, size=2)
        s = config[i, j]
        nb = config[(i+1)%L, j] + config[(i-1)%L, j] + config[i, (j+1)%L] + config[i, (j-1)%L]
        dE = 2 * s * nb
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            config[i, j] = -s
    return config

def fases_fourier(config):
    """Devuelve las fases de la transformada de Fourier 2D."""
    fft = np.fft.fft2(config)
    fases = np.angle(fft)
    return fases


import matplotlib.pyplot as plt

L = 64
temps = np.linspace(1.5, 3.5, 20)
Js = []

for T in temps:
    config = ising_2d(L, T, n_iter=200)
    fases = fases_fourier(config)
    J = J_univariante(fases.flatten())
    Js.append(J)
    print(f"T={T:.2f}, J={J:.4f}")

plt.plot(temps, Js, marker='o')
plt.xlabel("Temperatura T")
plt.yline(y=9.60745723e-01)
plt.ylabel("Índice J")
plt.title("Índice J vs Temperatura en el modelo de Ising 2D")
plt.grid(True)
plt.show()
