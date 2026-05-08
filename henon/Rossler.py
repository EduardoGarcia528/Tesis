import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parámetros del sistema de Rössler
# ============================================================
a = 0.2
b = 0.2

# Variación de c: de régimen más regular a más caótico
c_values = np.array([4.0, 6.0, 8.5, 8.7, 9.0, 12.0, 12.6, 13.0, 18.0])

t_max = 1200
dt = 0.01
t_transient = 200

x0, y0, z0 = 0.1, 0.1, 0.1

# Para no usar una serie demasiado sobremuestreada en la medida
subsample = 10

rng = np.random.default_rng(123)

medida = []
medida_null = []



for i, c in enumerate(c_values):

    t, x, y, z = ml.rossler_system(
        a=a,
        b=b,
        c=c,
        t_max=t_max,
        dt=dt,
        t_transient=t_transient,
        x0=x0,
        y0=y0,
        z0=z0    )


    medida.append(
        ml.indice_H(x, y)
    )

    medida_null.append( 
        ml.indice_H(
            rng.permutation(x),
            rng.permutation(y)
        )
    )



medida = np.asarray(medida)
medida_null = np.asarray(medida_null)


fig, ax1 = plt.subplots(figsize=(8, 5))


ax1.set_xlabel(r"$c$")
ax1.set_ylabel(r"Máximos locales de $x(t)$")
ax1.set_title(
    rf"Sistema de Rössler: bifurcación al variar $c$ "
    rf"con $a={a}$, $b={b}$"
)


ax1.plot(
    c_values,
    medida,
    color="red",
    lw=2,
    label=r"$J_{\theta}$"
)

ax1.plot(
    c_values,
    medida_null,
    color="blue",
    lw=2,
    label=r"$J_{\theta}$ null"
)

ax1.set_ylim(0, 1)
ax1.set_ylabel(r"Medida: $J_{\theta}$")
ax1.legend(loc="upper right")

plt.tight_layout()
plt.show()