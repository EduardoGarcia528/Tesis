import numpy as np
import matplotlib.pyplot as plt

# ======= TUS FUNCIONES (tal cual) =======
def logistic_exponential_map(r, x0, n, l):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] * np.exp(r*(1 - x[i-1])) + l
    return x[3*n//4:]  # nos quedamos con el último cuarto (post-transitorio)

def detect_period(x, max_period=16, tol=1e-3):
    for p in range(1, max_period + 1):
        if np.all(np.abs(x[-p:] - x[-2*p:-p]) < tol):
            return p
    return None
# ========================================

# Parámetros del barrido
r_min, r_max, nr = 0.0, 5.0, 260
lam_min, lam_max, nlam = 0.00, 1.20, 320
iters = 1000      # iteraciones totales por punto
x0 = 0.3          # condición inicial
tol = 1e-3        # tolerancia para detectar periodo
max_p = 4        # periodo máximo a checar

r_vals = np.linspace(r_min, r_max, nr)
lam_vals = np.linspace(lam_min, lam_max, nlam)

# Matriz de clases: 1,2,4,8 y 99 (caos/alto periodo)
classes = np.empty((nlam, nr), dtype=int)

for i, lam in enumerate(lam_vals):
    for j, r in enumerate(r_vals):
        x = logistic_exponential_map(r, x0, iters, lam)
        p = detect_period(x, max_period=max_p, tol=tol)
        if p in (1, 2, 4, 8):
            classes[i, j] = p
        elif p is None:
            classes[i, j] = 99
        else:
            # si encontró p>8 (10, 12, 16, etc.), lo puedes agrupar como 8 o marcar 99
            classes[i, j] = 8  # o usa 99 si prefieres distinguirlo como "no simple"
            
# Paleta discreta
import matplotlib.colors as mcolors
labels = {1:'p1', 2:'p2', 4:'p4', 8:'p8', 99:'caótico/alto'}
key_order = [1,2,4,8,99]
colors = [(0.20,0.60,0.20),  # p1  verde
          (0.20,0.30,0.90),  # p2  azul
          (0.70,0.30,0.90),  # p4  violeta
          (1.00,0.60,0.20),  # p8  naranja
          (0.90,0.10,0.10)]  # 99  rojo

# Mapear clases a índices 0..4
lookup = {k:i for i,k in enumerate(key_order)}
plot_mat = np.vectorize(lambda v: lookup.get(int(v), lookup[99]))(classes)

cmap = mcolors.ListedColormap(colors)
bounds = np.arange(-0.5, len(key_order)+0.5, 1.0)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(9,6))
im = plt.imshow(plot_mat, origin='lower', aspect='auto',
                extent=[r_min, r_max, lam_min, lam_max],
                cmap=cmap, norm=norm, interpolation='nearest')
plt.xlabel(r'$r$ (tasa de crecimiento)')
plt.ylabel(r'$\lambda$ (inmigración/refugio)')
plt.title(r'Plano de estabilidad $(\lambda, r)$ — reversión de duplicación de período')

# Leyenda
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=colors[k], label=labels[key_order[k]]) for k in range(len(key_order))]
plt.legend(handles=legend_patches, loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
