import numpy as np
import matplotlib.pyplot as plt

# Ejemplo: dos o más arrays
x1 = np.load("S_aleatorio5.npy")
x2 = np.load("S_aleatorio6.npy")
x3 = np.load("S_aleatorio7.npy")
x4 = np.load("S_aleatorio8.npy")

arrays = [x1, x2, x3, x4]
labels = [r"$S_{\theta}$", r"$S_{\theta}^{(\hat{\Pi \theta})}$", r"$ S_{\theta}^{(\Pi \theta)}$", r"$ S_{\Delta \theta}^{(\Pi \Delta \theta)}$"]

# Bines comunes para que la comparación sea justa
todos = np.concatenate(arrays)
bins = np.histogram_bin_edges(todos, bins=80)
# También puedes usar: bins=30, bins="fd", bins="auto"

plt.figure(figsize=(7, 4))

for arr, label in zip(arrays, labels):
    plt.hist(
        arr,
        bins=bins,
        alpha=0.45,        # transparencia para ver solapamiento
        density=True,      # normaliza para comparar distribuciones
        label=label,
        edgecolor="black"
    )

plt.xlabel(r"$S$",fontsize=12)
plt.ylabel("Densidad")
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()