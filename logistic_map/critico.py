import numpy as np
import matplotlib.pyplot as plt

def logistic_bifurcation(r_values, n_transient=1500, n_keep=120, x0=0.5):
    """
    Genera datos del diagrama de bifurcación del mapa logístico:
        x_{n+1} = r x_n (1 - x_n)

    Parámetros
    ----------
    r_values : ndarray
        Valores de r.
    n_transient : int
        Iteraciones transitorias descartadas.
    n_keep : int
        Número de iteraciones finales que se conservan para el diagrama.
    x0 : float
        Condición inicial.

    Retorna
    -------
    rr : ndarray
        Valores de r repetidos.
    xx : ndarray
        Valores x del atractor para cada r.
    """
    r_values = np.asarray(r_values, dtype=float)
    x = np.full_like(r_values, x0, dtype=float)

    for _ in range(n_transient):
        x = r_values * x * (1.0 - x)

    xs = []
    for _ in range(n_keep):
        x = r_values * x * (1.0 - x)
        xs.append(x.copy())

    xx = np.array(xs)                 # shape (n_keep, len(r_values))
    rr = np.tile(r_values, n_keep)    # shape (n_keep * len(r_values),)
    return rr, xx.ravel()


def ajuste_doble_interseccion_con_bifurcacion(
    archivo='J_transicion.npy',
    x1_ini=3.56955, x1_fin=3.56985,
    x2_ini=3.56998, x2_fin=3.57018,
    n_transient=2000,
    n_keep=150
):
    """
    Carga J_transicion.npy, usa como eje x el arreglo r especificado,
    dibuja arriba el diagrama de bifurcación del mapa logístico y abajo
    la curva de J con dos ajustes lineales e intersección.

    Parámetros
    ----------
    archivo : str
        Archivo .npy. Puede tener shape (2, N) o (N,).
        - Si es (2, N), se usa data[1] como y y se ignora data[0].
        - Si es (N,), se usa directamente como y.
    x1_ini, x1_fin : float
        Rango en x del primer ajuste.
    x2_ini, x2_fin : float
        Rango en x del segundo ajuste.
    n_transient : int
        Transitorio para el diagrama de bifurcación.
    n_keep : int
        Puntos conservados por cada r en el diagrama de bifurcación.
    """

    # Eje x exacto pedido
    r_full = np.sort(np.concatenate((
        np.linspace(3.5695, 3.5702, 300),
        np.array([3.569945672])
    )))

    data = np.load(archivo)

    # Extraer y del archivo
    if data.ndim == 2:
        if data.shape[0] == 2:
            y = data[1]
        elif data.shape[1] == 2:
            y = data[:, 1]
        else:
            raise ValueError(
                f"No se pudo interpretar el archivo con shape {data.shape}. "
                "Se esperaba (2, N), (N, 2) o (N,)."
            )
    elif data.ndim == 1:
        y = data
    else:
        raise ValueError(f"Formato no soportado: shape={data.shape}")

    # Compatibilizar longitud de r con longitud de y
    if len(y) == len(r_full):
        x = r_full.copy()
    elif len(y) == 300:
        # quitamos el punto extra 3.569945672 para igualar 300 puntos
        x = np.linspace(3.5695, 3.5702, 300)
        print("Aviso: J_transicion.npy tiene 300 puntos; se usa r = linspace(3.5695, 3.5702, 300)")
    else:
        raise ValueError(
            f"La longitud de y es {len(y)}, pero el eje r tiene 301 puntos "
            "o 300 puntos sin el valor extra. No coinciden."
        )

    # Máscaras por valores de x
    mask1 = (x >= x1_ini) & (x <= x1_fin)
    mask2 = (x >= x2_ini) & (x <= x2_fin)

    x1, y1 = x[mask1], y[mask1]
    x2, y2 = x[mask2], y[mask2]

    if len(x1) < 2:
        raise ValueError("El primer intervalo no contiene suficientes puntos.")
    if len(x2) < 2:
        raise ValueError("El segundo intervalo no contiene suficientes puntos.")

    # Ajustes lineales
    p1 = np.polyfit(x1, y1, 1)
    p2 = np.polyfit(x2, y2, 1)
    m1, b1 = p1
    m2, b2 = p2

    if np.isclose(m1, m2):
        raise ValueError("Las rectas son paralelas o casi paralelas.")

    # Intersección
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1

    # Datos del diagrama de bifurcación
    rr, xx = logistic_bifurcation(x, n_transient=n_transient, n_keep=n_keep)

    # Rectas ajustadas para dibujar
    xfit = np.linspace(np.min(x), np.max(x), 600)
    yfit1 = m1 * xfit + b1
    yfit2 = m2 * xfit + b2

    # Figura
    fig, ax0 = plt.subplots(
        1, 1, figsize=(9, 8))


    # Panel superior: bifurcación
    ax0.plot(rr, xx, ',', alpha=0.6)
    ax0.axvline(x_int, ls=':', lw=1.5)
    ax0.set_ylabel(r'$x_n$')

    # Panel inferior: datos + ajustes
    ax0.plot(x, y, 'o-', ms=3, lw=1, label='J')
    ax0.plot(xfit, yfit1, '--', lw=2)
    ax0.plot(xfit, yfit2, '--', lw=2)
    ax0.plot(x_int, y_int, 'r*', ms=12, color='blue', label=fr'Intersección: $r_\infty ={x_int:.9f}$')
    ax0.axvline(x_int, ls=':', lw=1.5)

    ax0.set_xlabel('r')
    ax0.set_ylabel('J', rotation = 360)
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

    print(f"Ajuste 1: y = {m1:.12e} x + {b1:.12e}")
    print(f"Ajuste 2: y = {m2:.12e} x + {b2:.12e}")
    print(f"Intersección en x = {x_int:.12f}")
    print(f"Intersección en y = {y_int:.12e}")

    return x_int, y_int, p1, p2


# Ejemplo de uso
x_int, y_int, p1, p2 = ajuste_doble_interseccion_con_bifurcacion(
    archivo='J_transicion2.npy',
    x1_ini=3.57-0.0005, x1_fin=3.57-0.0001,
    x2_ini=3.57-0.00005, x2_fin=3.57+0.0002,
    n_transient=1000,
    n_keep=120
)