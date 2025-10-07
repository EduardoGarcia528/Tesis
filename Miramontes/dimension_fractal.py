import numpy as np

def box_counting_dimension_3d(x, y, z, 
                              n_scales=15, 
                              eps_range=None,
                              n_shifts=0,
                              fit_range=None,
                              return_all=False,
                              seed=0):
    """
    Estima la dimensión fractal (Minkowski/box-counting) de un conjunto 3D.

    Parámetros
    ----------
    x, y, z : array_like (1D)
        Coordenadas del atractor en R^3.
    n_scales : int
        Nº de escalas (eps) en malla logarítmica.
    eps_range : tuple or None
        (eps_min, eps_max). Si None, se estima automáticamente.
    n_shifts : int
        Nº de desplazamientos aleatorios de la rejilla para promediar (0 = sin promedio).
    fit_range : tuple or None
        Índices (i_min, i_max) para ajustar la recta solo en ese rango de escalas.
        Útil para evitar escalas muy grandes/pequeñas que rompen la ley de potencia.
    return_all : bool
        Si True, devuelve también epsilons y Ns (promediados).
    seed : int
        Semilla para los desplazamientos.

    Returns
    -------
    D : float
        Dimensión estimada (pendiente del ajuste log-log).
    fit_coeffs : np.ndarray
        Coeficientes del ajuste lineal [pendiente, intercepto] en log-log.
    (opcional) epsilons, Ns : arrays
        Escalas y número de cajas ocupadas (promedio en shifts si aplica).
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    assert x.ndim == y.ndim == z.ndim == 1 and len(x) == len(y) == len(z)

    # Encajonar en el paralelepípedo mínimo y normalizar a [0,1]^3 (mejora estabilidad numérica)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    Lx = xmax - xmin; Ly = ymax - ymin; Lz = zmax - zmin
    # Evitar divisiones por cero si hay coordenadas degeneradas
    Lx = Lx if Lx > 0 else 1.0
    Ly = Ly if Ly > 0 else 1.0
    Lz = Lz if Lz > 0 else 1.0

    X = (x - xmin) / Lx
    Y = (y - ymin) / Ly
    Z = (z - zmin) / Lz

    # Rango de eps: por defecto evita escalas donde N(eps) satura (muy grande) o es casi un punto por caja (muy pequeña)
    if eps_range is None:
        # Heurística: de ~10^{-1.5} a ~10^{-0.3} del tamaño normalizado (igual que tu 2D original)
        eps_min, eps_max = 10**(-1.8), 10**(-0.3)
    else:
        eps_min, eps_max = eps_range

    epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), n_scales)

    rng = np.random.default_rng(seed)

    Ns = np.zeros_like(epsilons, dtype=float)

    # Función para contar cajas ocupadas en una rejilla 3D de tamaño eps con posible desplazamiento
    def count_boxes_3d(X, Y, Z, eps, shift=None):
        if shift is None:
            sx = sy = sz = 0.0
        else:
            sx, sy, sz = shift  # desplazamiento en [0, eps)
        # Índices de caja
        ix = np.floor((X + sx) / eps).astype(np.int64)
        iy = np.floor((Y + sy) / eps).astype(np.int64)
        iz = np.floor((Z + sz) / eps).astype(np.int64)
        # Conjunto de cajas ocupadas
        # Usamos un hash de tuplas para evitar colisiones
        boxes = set(zip(ix, iy, iz))
        return len(boxes)

    for k, eps in enumerate(epsilons):
        if n_shifts <= 0:
            Ns[k] = count_boxes_3d(X, Y, Z, eps, shift=None)
        else:
            total = 0
            for _ in range(n_shifts):
                # Desplazamientos uniformes en [0, eps)
                shift = rng.uniform(0.0, eps, size=3)
                total += count_boxes_3d(X, Y, Z, eps, shift=shift)
            Ns[k] = total / n_shifts

    # Ajuste lineal en log-log: log N(eps) vs log(1/eps)
    log_inv_eps = np.log(1.0 / epsilons)
    log_N = np.log(Ns)

    # Elegir rango de ajuste
    if fit_range is None:
        # Por defecto: descartamos 2 puntos de cada extremo (ajústalo si lo necesitas)
        i0, i1 = 2, len(epsilons) - 2
    else:
        i0, i1 = fit_range
        i1 = min(i1, len(epsilons))

    coeffs = np.polyfit(log_inv_eps[i0:i1], log_N[i0:i1], 1)
    D = coeffs[0]  # pendiente = dimensión
    if return_all:
        return D, coeffs, epsilons, Ns
    else:
        return D, coeffs


import matplotlib.pyplot as plt

serie = np.loadtxt('series/El_nino.txt')
x = serie[:-90]
y = serie[45:-45]
z = serie[90:]

D, coeffs, eps, Ns = box_counting_dimension_3d(x, y, z, 
                                                n_scales=18, 
                                                n_shifts=5, 
                                                return_all=True)

log_inv_eps = np.log(1/eps); log_N = np.log(Ns)
yfit = np.polyval(coeffs, log_inv_eps)

plt.figure(figsize=(6,5))
plt.plot(log_inv_eps, log_N, 'o-', label='Datos')
plt.plot(log_inv_eps, yfit, 'r--', label=f'Ajuste (pendiente ≈ {coeffs[0]:.3f})')
plt.xlabel(r'$\log(1/\varepsilon)$')
plt.ylabel(r'$\log N(\varepsilon)$')
plt.legend()
plt.tight_layout()
plt.show()
