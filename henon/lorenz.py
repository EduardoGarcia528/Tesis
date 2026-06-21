import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parámetros del sistema de Lorenz
# ============================================================

sigma = 10
beta = 8/3

# Variación de rho en el orden indicado
# rho_values = np.array([28.0,99.65,100.5, 160.0, 350])
rho_values = np.array([350, 100.5, 160, 99.65, 28.0])

t_max = 1200
dt = 0.01
t_transient = 200

x0, y0, z0 = 0.1, 0.1, 0.1

# Para no usar una serie demasiado sobremuestreada en la medida
subsample = 10

rng = np.random.default_rng(123)

medida = []
medida_null = []

for i, rho in enumerate(rho_values):

    t, x, y, z = ml.lorenz_system(
        sigma=sigma,
        rho=rho,
        beta=beta,
        t_max=t_max,
        dt=dt,
        t_transient=t_transient,
        x0=x0,
        y0=y0,
        z0=z0
    )

    # Submuestreo opcional
    x_m = x[::subsample]
    y_m = y[::subsample]

    medida.append(
        ml.indice_S_eff_fast(x_m, seriey=None, tau=1, delta=False)
    )

    medida_null.append(
        ml.indice_S_eff_fast(
            ml.iaaft(x_m,1)[0],
            seriey=None,
            tau=1,
            null="no",
            delta=False
        )
    )

medida = np.asarray(medida)
medida_null = np.asarray(medida_null)

fig, ax1 = plt.subplots(figsize=(8, 5))

casos = np.arange(len(rho_values))

ax1.plot(
    casos,
    medida,
    color="red",
    marker="o",
    lw=2,
    label=r"$J_{\theta}$"
)

ax1.plot(
    casos,
    medida_null,
    color="blue",
    marker="o",
    lw=2,
    label=r"$J_{\theta}$ null"
)

ax1.set_xticks(casos)
ax1.set_xticklabels([str(rho) for rho in rho_values])

ax1.set_xlabel(r"$\rho$")
ax1.set_ylabel(r"Medida: $J_{\theta}$")
ax1.set_ylim(0, 1)

ax1.set_title(
    rf"Sistema de Lorenz: índice $J_\theta$ para distintos $\rho$ "
    rf"con $\sigma={sigma}$, $\beta={beta:.3g}$"
)

ax1.legend(loc="upper right")

plt.tight_layout()
plt.show()