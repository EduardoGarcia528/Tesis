import numpy as np
import matplotlib.pyplot as plt

def map_step(x, y, mu, beta):
    x_next = mu * x * (1.0 - x - y)
    y_next = mu * x * y
    return x_next, y_next


def bifurcation_diagram(mu_min, mu_max, n_mu, beta, n_transient, n_iter, x0, y0):
    mu_values = np.linspace(mu_min, mu_max, n_mu)

    mu_plot = []
    x_plot  = []

    for mu in mu_values:
        x, y = x0, y0

        for _ in range(n_transient):
            x, y = map_step(x, y, mu, beta)


        for _ in range(n_iter):
            x, y = map_step(x, y, mu, beta)
            mu_plot.append(mu)
            x_plot.append(x)   

    mu_plot = np.array(mu_plot)
    x_plot  = np.array(x_plot)
    return mu_plot, x_plot

