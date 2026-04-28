import numpy as np
import matplotlib.pyplot as plt


def ajuste_lineal_doble(x, y, min_puntos=5, graficar=True):
    """
    Ajusta dos rectas a un conjunto de datos y calcula la intersección.

    Parámetros
    ----------
    x, y : array-like
        Datos a ajustar.
    min_puntos : int
        Número mínimo de puntos que debe tener cada región.
    graficar : bool
        Si True, grafica los datos y los ajustes.

    Regresa
    -------
    resultado : dict
        Diccionario con los parámetros de las rectas y la intersección.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Ordenar datos por x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    n = len(x)

    if n < 2 * min_puntos:
        raise ValueError("No hay suficientes puntos para hacer dos ajustes lineales.")

    mejor_error = np.inf
    mejor_resultado = None

    # Probar todos los posibles cortes
    for i in range(min_puntos, n - min_puntos):

        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        # Ajuste lineal: y = m x + b
        m1, b1 = np.polyfit(x1, y1, 1)
        m2, b2 = np.polyfit(x2, y2, 1)

        y1_fit = m1 * x1 + b1
        y2_fit = m2 * x2 + b2

        error = np.sum((y1 - y1_fit)**2) + np.sum((y2 - y2_fit)**2)

        if error < mejor_error:
            mejor_error = error
            mejor_resultado = {
                "indice_corte": i,
                "x_corte": x[i],
                "m1": m1,
                "b1": b1,
                "m2": m2,
                "b2": b2,
                "error": error
            }

    m1 = mejor_resultado["m1"]
    b1 = mejor_resultado["b1"]
    m2 = mejor_resultado["m2"]
    b2 = mejor_resultado["b2"]

    # Intersección entre las rectas:
    # m1*x + b1 = m2*x + b2
    if np.isclose(m1, m2):
        x_interseccion = np.nan
        y_interseccion = np.nan
        print("Las rectas son casi paralelas. No hay una intersección bien definida.")
    else:
        x_interseccion = (b2 - b1) / (m1 - m2)
        y_interseccion = m1 * x_interseccion + b1

        print(f"x en la intersección = {x_interseccion}")
        print(f"y en la intersección = {y_interseccion}")

    mejor_resultado["x_interseccion"] = x_interseccion
    mejor_resultado["y_interseccion"] = y_interseccion

    if graficar:
        x_fit = np.linspace(np.min(x), np.max(x), 500)

        y_fit_1 = m1 * x_fit + b1
        y_fit_2 = m2 * x_fit + b2

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=25, label="Datos")
        plt.plot(x_fit, y_fit_1, label="Recta 1")
        plt.plot(x_fit, y_fit_2, label="Recta 2")
        plt.axvline(3.569945672, color='gray', ls='--', label="Valor teórico")

        if not np.isnan(x_interseccion):
            plt.axvline(x_interseccion, linestyle="--", label=f"Intersección x = {x_interseccion}")
            plt.scatter(x_interseccion, y_interseccion, s=80, zorder=5)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return mejor_resultado


Rs = []
for i in ['', '2', '3', '4', '5']:
    x, y  = np.load(f"J_transicion{i}.npy")
    resultado = ajuste_lineal_doble(x, y, min_puntos=130, graficar=True)
    Rs.append(resultado["x_interseccion"])
    print("Intersección en x:", resultado["x_interseccion"])

print("promedio ", np.mean(Rs))