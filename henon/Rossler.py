import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parámetros del sistema de Rössler
# ============================================================


# Variación de c: de régimen más regular a más caótico
c_values = np.array([4.0, 6.0, 8.5, 8.7, 9.0, 12.0, 12.6, 13.0, 18.0])
c_values = np.array([4.0, 6.0, 12.0, 8.5,12.6,8.7, 13.0,9.0, 18.0])



rng = np.random.default_rng(123)

medida = []
medida_null = []



for i, c in enumerate(c_values):

    t, x, y, z = ml.rossler_system(c=c)
    x = x + np.random.normal(0,1,len(x))


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
c_pos = np.arange(len(c_values))
fig, ax1 = plt.subplots(figsize=(8, 5))


ax1.set_xlabel(r"$c$")
ax1.set_ylabel(r"Máximos locales de $x(t)$")
ax1.set_title(
    rf"Sistema de Rössler: bifurcación al variar $c$ "
    rf"con $a={0.1}$, $b={0.1}$"
)


ax1.plot(
    c_pos,
    medida,
    color="red",
    lw=2,
    label=r"$J_{\theta}$"
)

ax1.plot(
    c_pos,
    medida_null,
    color="blue",
    lw=2,
    label=r"$J_{\theta}$ null"
)

ax1.set_ylim(0, 1)
ax1.set_ylabel(r"Medida: $J_{\theta}$")
ax1.legend(loc="upper right")
plt.xticks(
    c_pos,
    [str(c) for c in c_values]
)

plt.tight_layout()
plt.show()