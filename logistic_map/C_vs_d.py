import numpy as np
import matplotlib.pyplot as plt
import math
from funciones import angulos_alpha
from gamma_4 import gamma_index_jacobs

def logistic_map(r, x0, n):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

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
    
    # valores muy pequeños se consideran cero
    delta2_C[np.abs(delta2_C) < tol] = 0.0
    return delta2_C
# =========================
# Parámetros
# =========================
x0 = 0.6
n_total = 21000
n_trans = 1000
max_gamma = 15
mu = 20
 
r_values = np.sort(np.append(np.linspace(3.5, 4.0, 24), np.array([3.569945672, 3.569949700])))    # puedes aumentar esto

n_plots = len(r_values)
ncols = math.ceil(np.sqrt(n_plots))
nrows = math.ceil(n_plots / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.8*nrows), squeeze=False)
axes = axes.flatten()

for i, r in enumerate(r_values):
    serie = logistic_map(r, x0, n_total)[n_trans:]
    angulos = angulos_alpha(serie,False)
    C, g = gamma_index_jacobs(angulos, max_gamma, mu)
    d = np.arange(len(C))
    ax = axes[i]

    ax.plot(d[2:], C[2:], marker='o', linestyle='-', linewidth=1.0, markersize=2.5)
    ax.set_title(rf"$r={r:.4f}$", fontsize=10)
    # ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$C(d)$")

for j in range(n_plots, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.8*nrows), squeeze=False)
# axes = axes.flatten()
# for i, r in enumerate(r_values):
#     serie = logistic_map(r, x0, n_total)[n_trans:]
#     C, g = gamma_index_jacobs(serie, max_gamma, mu)
#     ax = axes[i]
#     d = np.arange(len(C))
#     delta2_C = segunda_diferencia(C[2:])
#     d_delta2 = np.arange(3, len(C)-1)
#     ax.plot(d_delta2, delta2_C, marker='o', linestyle='-', linewidth=1.0, markersize=2.5)
#     ax.set_title(rf"$r={r:.4f}$", fontsize=10)
#     # ax.set_xlabel(r"$d$")
#     ax.set_ylabel(r"$\delta C(d)$")

# for j in range(n_plots, len(axes)):
#     axes[j].axis("off")

# plt.tight_layout()
# plt.show()