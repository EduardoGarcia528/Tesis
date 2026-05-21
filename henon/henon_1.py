import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# f = ml.rossler_system(a=0.2,b=0.2,c=5.7,t_max=1000,dt=0.01,t_transient=0,x0=0.1,y0=0.1,z0=0.1)
# Parámetros
b = 0.3
a_values = np.linspace(1.0, 1.4, 300)
r_values = np.linspace(3.25, 4.0, 300)

n_iter = 10000
n_transient = 2000

# Arreglos para guardar datos
A_bif = []
X_bif = []
medida = []
medida_null = []
# t,x,y2,z = ml.lorenz_system(dt=0.1)

for r,a,tau in zip(r_values,a_values,range(1,301)):

    # Ojo: aquí debe ir a=a, no a=1.4
    x,y = ml.henon_map(a, b, x0=0.1, y0=0.1, n_trans=1000, n_points=10000)

    # Diagrama de bifurcación: guardamos x_n contra a
    A_bif.extend([a] * len(x))
    X_bif.extend(y)

    # Medida cualquiera: desviación estándar de x_n
    # tauu = 1
    # x = np.random.normal(0, 1, size=len(x))
    # y = np.random.normal(0, 1, size=len(y))
    # x = (1-r)*x + r*z[tauu:]
    # y = (1-r)*y + r*z[:-tauu]
    # y= ml.logistic_map(r=4.0,n_iter=n_iter,n_transient=n_transient)
    medida.append(ml.indice_S_eff_fast(x,seriey=y,tau=1))
    # x,y = np.random.permutation(x), np.random.permutation(y)
    medida_null.append(ml.indice_S_eff_fast(x,seriey=y,tau=1,null="shuffle2"))
medida = np.asarray(medida)
medida_null = np.asarray(medida_null)
print(np.mean(medida),np.mean(medida_null))

# Figura
fig, ax1 = plt.subplots(figsize=(8, 5))

# Diagrama de bifurcación
ax1.plot(
    A_bif,
    X_bif,
    ',k',
    alpha=0.35
)

ax1.set_xlabel(r"$a$")
ax1.set_ylabel(r"$x_n$")
ax1.set_title(r"Diagrama de bifurcación del mapa de Hénon")

# Curva roja de la medida
ax2 = ax1.twinx()

ax2.plot(
    a_values,
    medida,
    color="red",
    lw=2,
    label=r"$J_{\theta}$"
)

ax2.plot(
    a_values,
    medida_null,
    color="blue",
    lw=2,
    label=r"$J_{\theta}$ (null)"
)
ax2.set_ylim(0, 1)
ax2.legend(loc="lower left")
ax2.set_ylabel(r"Medida: $J_{\theta}$", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.tight_layout()
plt.show()

print("Pearson:", pearsonr(medida_null, medida))
print("Spearman:", spearmanr(medida_null, medida))