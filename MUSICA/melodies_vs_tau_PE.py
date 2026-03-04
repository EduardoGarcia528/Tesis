import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

from funciones import angulos_alpha, permutation_entropy
from gamma_4 import gamma_index_jacobs

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

def segunda_diferencia(C, tol=1e-5):
    C = np.asarray(C, dtype=float)
    delta2_C = C[2:] - 2*C[1:-1] + C[:-2]
    delta2_C[np.abs(delta2_C) < tol] = 0.0
    return delta2_C

# =========================
# Parámetros
# =========================
melodies_folder = Path("melodies")
max_files = 23
max_tau = 15
m = 5

# Buscar archivos .npy
files = sorted(melodies_folder.glob("*.npy"))

if len(files) < max_files:
    raise ValueError(
        f"Se encontraron solo {len(files)} archivos .npy en '{melodies_folder}', "
        f"pero se esperaban al menos {max_files}."
    )

files = files[:max_files]

n_plots = len(files)
ncols = math.ceil(np.sqrt(n_plots))
nrows = math.ceil(n_plots / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
axes = axes.flatten()

for i, file_path in enumerate(files):
    # Cargar melodía
    serie = np.load(file_path, allow_pickle=True)
    serie = np.asarray(serie).squeeze()

    # Verificar que sea univariante
    if serie.ndim != 1:
        raise ValueError(f"El archivo {file_path.name} no contiene un array univariante.")

    # Calcular C(d) con entropía permutacional variando tau
    C = []
    for tau in range(1, max_tau + 1):
        pe = permutation_entropy(serie, m=m, tau=tau)
        C.append(pe)

    # Si quieres usar gamma en vez de PE, descomenta esto:
    # angulos = angulos_alpha(serie, False)
    # C, g = gamma_index_jacobs(angulos, max_gamma, mu)

    d = np.arange(1, len(C) + 1)
    ax = axes[i]

    ax.plot(d, C, marker='o', linestyle='-', linewidth=1.0, markersize=2.5)
    ax.set_title(file_path.stem, fontsize=9)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel(r"$PE$")
    ax.set_xlabel(r"$\tau$")

# Apagar ejes vacíos
for j in range(n_plots, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()