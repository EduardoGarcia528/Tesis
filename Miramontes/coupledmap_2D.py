import numpy as np


def heaviside_clamp(z):
    """
    Heaviside 'suave' usada por Solé y Valls:
    H(z) = z si z > 0, 0 en otro caso.
    """
    return np.where(z > 0.0, z, 0.0)


def laplacian_periodic(field):
    """
    Laplaciano discreto en 2D con condiciones de frontera periódicas.
    
    Δu(i,j) = u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4 u(i,j)
    
    field: array 2D (N x N)
    """
    up    = np.roll(field, -1, axis=0)
    down  = np.roll(field,  1, axis=0)
    right = np.roll(field, -1, axis=1)
    left  = np.roll(field,  1, axis=1)
    return up + down + right + left - 4.0 * field


def lv_cml_step(x, y, a=4.0, b=1.0, D1=0.10, D2=0.15):
    """
    Un paso temporal del mapeo acoplado Lotka–Volterra 2D.
    
    x, y: arrays 2D con las poblaciones en el tiempo n.
    a, b: parámetros del mapa LV local.
    D1, D2: coeficientes de difusión para x e y.
    
    Devuelve x_next, y_next.
    """
    lap_x = laplacian_periodic(x)
    lap_y = laplacian_periodic(y)

    # Parte local tipo Lotka–Volterra (versión discreta del artículo)
    x_local = a * x * (1.0 - x - y)
    y_local = b * x*y  # el depredador depende de la presa local

    # Añadimos difusión
    x_new = x_local + D1 * lap_x
    y_new = y_local + D2 * lap_y

    # Aplicamos Heaviside para evitar valores negativos
    x_new = heaviside_clamp(x_new)
    y_new = heaviside_clamp(y_new)

    return x_new, y_new


def inicializar_lattice(N=100, seed=None):
    """
    Inicializa x e y con condiciones aleatorias pequeñas (0 a 0.1).
    """
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 0.1, size=(N, N))
    y0 = rng.uniform(0.0, 0.1, size=(N, N))
    return x0, y0


def simular_lv_cml(
    N=100,
    pasos=10_000,
    a=2.5,
    b=4.0,
    D1=0.10,
    D2=0.15,
    seed=None,
    guardar_cada=None,
):
    """
    Simula el mapeo acoplado Lotka–Volterra 2D.
    
    Parámetros:
    - N: tamaño de la red (N x N).
    - pasos: número de iteraciones temporales.
    - a, b, D1, D2: parámetros del modelo.
    - seed: semilla para reproducibilidad.
    - guardar_cada: si es un entero, guarda un snapshot cada 'guardar_cada' pasos
      y devuelve una lista de (x_snap, y_snap). Si es None, solo devuelve el estado final.
    """
    x, y = inicializar_lattice(N=N, seed=seed)

    snapshots = []
    for n in range(pasos):
        x, y = lv_cml_step(x, y, a=a, b=b, D1=D1, D2=D2)

        if guardar_cada is not None and (n % guardar_cada == 0):
            snapshots.append((x.copy(), y.copy()))

    if guardar_cada is None:
        return x, y
    else:
        return x, y, snapshots


if __name__ == "__main__":
    # Ejemplo de uso:
    N = 200
    pasos = 25000

    x_final, y_final, snaps = simular_lv_cml(
        N=N,
        pasos=pasos,
        a=4.0,
        b=4.0,
        D1=0.099,
        D2=0.250,   # por ejemplo, parámetro donde suelen aparecer espirales
        seed=42,
        guardar_cada=1000,
    )

    # Aquí podrías graficar alguno de los snapshots con matplotlib, por ejemplo:
    import matplotlib.pyplot as plt
    plt.imshow(x_final, origin="lower")
    plt.colorbar()
    plt.show()
