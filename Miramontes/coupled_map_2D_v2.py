import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Definición del mapeo 2D
# ===============================
def map_step(x, y, mu, beta):
    """
    Un paso del mapeo:
        x_{n+1} = mu * x_n * (1 - x_n - y_n)
        y_{n+1} = beta * x_n * y_n
    """
    x_next = mu * x * (1.0 - x - y)
    y_next = mu * x * y
    return x_next, y_next


def bifurcation_diagram(mu_min, mu_max, n_mu, beta, n_transient, n_iter, x0, y0):
    mu_values = np.linspace(mu_min, mu_max, n_mu)

    mu_plot = []
    x_plot  = []

    for mu in mu_values:
        x, y = x0, y0

        # Transitorio
        for _ in range(n_transient):
            x, y = map_step(x, y, mu, beta)


        # Puntos en régimen estacionario
        for _ in range(n_iter):
            x, y = map_step(x, y, mu, beta)
            mu_plot.append(mu)
            x_plot.append(x)   # también podrías guardar y si quieres ver al depredador

    mu_plot = np.array(mu_plot)
    x_plot  = np.array(x_plot)
    return mu_plot, x_plot

