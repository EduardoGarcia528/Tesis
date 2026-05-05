import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
b = 0.3
a_values = np.linspace(1.0, 1.4, 300)

n_iter = 10000
n_transient = 2000

# Arreglos para guardar datos
A_bif = []
X_bif = []
medida = []
medida_null = []

for a in a_values:

    # Ojo: aquí debe ir a=a, no a=1.4
    x,y = ml.henon_map(a, b, x0=0.1, y0=0.1, n_trans=1000, n_points=10000)
    if a == 1.4:
        angulos = ml.angulos_alpha(np.random.permutation(x), np.random.permutation(y))
        plt.hist(angulos, bins=30, density=True)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Densidad")
        plt.title(r"Distribución de ángulos $\alpha$ para $a=1.4$")
        plt.show()

    # Diagrama de bifurcación: guardamos x_n contra a
    A_bif.extend([a] * len(x))
    X_bif.extend(y)

    # Medida cualquiera: desviación estándar de x_n
    medida.append(ml.indice_H(x,y))
    medida_null.append(ml.indice_H(np.random.permutation(x), np.random.permutation(y)))
medida = np.asarray(medida)
medida_null = np.asarray(medida_null)

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
ax2.set_ylabel(r"Medida: $J_{\theta}$", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.tight_layout()
plt.show()