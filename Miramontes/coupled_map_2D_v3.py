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
    y_local = b * x * y # el depredador depende de la presa local

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


def norm_state(dx, dy):
    """
    Norma euclidiana de la perturbación en el espacio de estados completo.
    dx, dy: arrays 2D con la diferencia en x e y.
    """
    return np.sqrt(np.sum(dx*dx + dy*dy))


def largest_lyapunov_cml(
    N=100,
    a=4.0,
    b=1.0,
    D1=0.099,
    D2=0.150,
    pasos_transitorio=5000,
    pasos_lyap=5000,
    delta0=1e-8,
    seed=None,
):
    """
    Calcula el exponente de Lyapunov máximo del CML Lotka–Volterra 2D
    usando el método de Wolf et al. (ec. (8) del artículo de Solé y Valls).
    
    Parámetros:
    - N: tamaño del lattice (N x N)
    - a, b, D1, D2: parámetros del modelo
    - pasos_transitorio: nº de pasos para llegar al atractor (no se mide Lyapunov)
    - pasos_lyap: nº de pasos durante los cuales se acumula el promedio
    - delta0: norma inicial de la perturbación
    - seed: semilla para reproducibilidad
    
    Devuelve:
    - lambda_max: estimación del exponente de Lyapunov máximo
    """
    rng = np.random.default_rng(seed)

    # 1. Estado inicial (trayectoria de referencia)
    x, y = inicializar_lattice(N=N, seed=seed)

    # 2. Evolucionar hasta el atractor (transitorio)
    for _ in range(pasos_transitorio):
        x, y = lv_cml_step(x, y, a=a, b=b, D1=D1, D2=D2)

    # 3. Definir perturbación inicial pequeña
    dx = rng.normal(0.0, 1.0, size=(N, N))
    dy = rng.normal(0.0, 1.0, size=(N, N))
    # Normalizar a norma delta0
    d_norm = norm_state(dx, dy)
    dx *= (delta0 / d_norm)
    dy *= (delta0 / d_norm)

    # 4. Segunda trayectoria: X' = X + deltaX
    x_p = x + dx
    y_p = y + dy

    # 5. Bucle principal para acumular el exponente
    suma_logs = 0.0

    for _ in range(pasos_lyap):
        # Evolucionar ambas trayectorias
        x,   y   = lv_cml_step(x,   y,   a=a, b=b, D1=D1, D2=D2)
        x_p, y_p = lv_cml_step(x_p, y_p, a=a, b=b, D1=D1, D2=D2)

        # Diferencia actual
        dx = x_p - x
        dy = y_p - y

        # Norma de la diferencia
        d_norm = norm_state(dx, dy)

        # Factor de crecimiento respecto a la norma de referencia
        g = d_norm / delta0
        if g <= 0.0:
            # En principio no debería ocurrir; por seguridad
            continue

        suma_logs += np.log(g)

        # Renormalizar la perturbación a delta0, manteniendo la dirección
        dx *= (delta0 / d_norm)
        dy *= (delta0 / d_norm)

        # Re-definir la trayectoria perturbada
        x_p = x + dx
        y_p = y + dy

    lambda_max = suma_logs / pasos_lyap
    return lambda_max


if __name__ == "__main__":
    lyaps = []
    for D2 in np.linspace(0.10, 0.25, 40):
        lambda_est = largest_lyapunov_cml(
            N=100,
            a=4.0,
            b=4.0,
            D1=0.099,
            D2=D2,
            pasos_transitorio=3000,
            pasos_lyap=6000,
            delta0=1e-10,
            seed=123,
        )
        lyaps.append((D2, lambda_est))

    import matplotlib.pyplot as plt
    np.save("coupledmap2d/lyap_cml_a4_b4_D2.npy", lyaps)
    D2_vals, lambda_vals = zip(*lyaps)
    plt.figure()
    plt.plot(D2_vals, lambda_vals, marker="o", linestyle="-")
    plt.xlabel("D2")
    plt.ylabel("Exponente de Lyapunov máximo")
    plt.title("Exponente de Lyapunov máximo vs D2 en CML Lotka–Volterra 2D")
    plt.show()