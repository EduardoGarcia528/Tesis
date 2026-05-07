import numpy as np
from numba import njit

"""LOGISTIC"""

def logistic_map(
    r,
    n_iter=1000,
    n_transient=0,
    ruido=None,
    sigma=0.0,
    x0=0.6,
    random_state=None,
    clip=True
):


    if ruido not in [None, "aditivo", "iterativo"]:
        raise ValueError("ruido debe ser None, 'aditivo' o 'iterativo'.")

    if n_iter <= 0:
        raise ValueError("n_iter debe ser mayor que cero.")

    if n_transient < 0:
        raise ValueError("n_transient debe ser mayor o igual que cero.")

    rng = np.random.default_rng(random_state)

    total_iter = n_iter + n_transient
    x = np.empty(total_iter + 1, dtype=float)
    x[0] = x0

    # Caso 1: ruido iterativo
    if ruido == "iterativo":
        for n in range(total_iter):
            eta = rng.normal(0.0, sigma)
            x[n + 1] = r * x[n] * (1.0 - x[n]) + eta

            if clip:
                x[n + 1] = np.clip(x[n + 1], 0.0, 1.0)

        return x[n_transient + 1:]

    # Caso 2: dinámica determinista
    for n in range(total_iter):
        x[n + 1] = r * x[n] * (1.0 - x[n])

        if clip:
            x[n + 1] = np.clip(x[n + 1], 0.0, 1.0)

    serie = x[n_transient + 1:]

    # Caso 3: ruido aditivo posterior
    if ruido == "aditivo":
        serie = serie + rng.normal(0.0, sigma, size=n_iter)

        if clip:
            serie = np.clip(serie, 0.0, 1.0)

    return serie

import numpy as np

"""HENON"""

@njit
def henon_map(a, b, x0=0.1, y0=0.1, n_trans=200, n_points=10000):
    x, y = x0, y0
    # Transitorio
    for _ in range(n_trans):
        x, y = 1 - a * x * x + y, b * x

    # Iteraciones para graficar
    xs = []
    ys = []
    for _ in range(n_points):
        x, y = 1 - a * x * x + y, b * x
        xs.append(x)
        ys.append(y)

    return xs, ys

import numpy as np
from scipy.integrate import solve_ivp

"""LORENZ"""

def lorenz_system(
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    t_max=100.0,
    dt=0.01,
    t_transient=0.0,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
    return_xyz=True
):
    """
    Integra el sistema de Lorenz:

        dx/dt = sigma (y - x)
        dy/dt = x (rho - z) - y
        dz/dt = x y - beta z

    usando un método numérico adaptativo de alta precisión.

    Parámetros
    ----------
    sigma : float
        Parámetro sigma del sistema de Lorenz.

    rho : float
        Parámetro rho del sistema de Lorenz.

    beta : float
        Parámetro beta del sistema de Lorenz.

    x0, y0, z0 : float
        Condiciones iniciales.

    t_max : float
        Tiempo total de integración.

    dt : float
        Paso temporal con el que se guarda la solución.
        El integrador internamente usa pasos adaptativos, no necesariamente dt.

    t_transient : float
        Tiempo inicial que se elimina como transitorio.

    method : str
        Método de integración. Recomendado: "DOP853".

    rtol : float
        Tolerancia relativa del integrador.

    atol : float
        Tolerancia absoluta del integrador.

    return_xyz : bool
        Si True, retorna t, x, y, z.
        Si False, retorna t, serie, donde serie tiene forma (N, 3).

    Retorna
    -------
    t : ndarray
        Tiempos después de eliminar el transitorio.

    x, y, z : ndarray
        Coordenadas del sistema de Lorenz.

    o bien

    serie : ndarray, shape (N, 3)
        Trayectoria tridimensional.
    """

    if t_max <= 0:
        raise ValueError("t_max debe ser mayor que cero.")

    if dt <= 0:
        raise ValueError("dt debe ser mayor que cero.")

    if t_transient < 0:
        raise ValueError("t_transient debe ser mayor o igual que cero.")

    if t_transient >= t_max:
        raise ValueError("t_transient debe ser menor que t_max.")

    def f(t, state):
        x, y, z = state

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return [dx, dy, dz]

    # Tiempos en los que se guarda la solución
    t_eval = np.arange(0.0, t_max + dt, dt)

    sol = solve_ivp(
        f,
        t_span=(0.0, t_max),
        y0=[x0, y0, z0],
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    if not sol.success:
        raise RuntimeError(f"La integración falló: {sol.message}")

    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    # Eliminación del transitorio
    mask = t >= t_transient

    t_out = t[mask]
    x_out = x[mask]
    y_out = y[mask]
    z_out = z[mask]

    if return_xyz:
        return t_out, x_out, y_out, z_out

    serie = np.column_stack((x_out, y_out, z_out))

    return t_out, serie

def local_maxima(y):
    """
    Detecta máximos locales simples de una serie y[n].

    Retorna
    -------
    maxima : ndarray
        Valores de y en los máximos locales.
    indices : ndarray
        Índices donde ocurren los máximos.
    """

    y = np.asarray(y)

    mask = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    indices = np.where(mask)[0] + 1
    maxima = y[indices]

    return maxima, indices

def lorenz_map(
    rho_values,
    sigma=10.0,
    beta=8.0 / 3.0,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    t_max=300.0,
    dt=0.01,
    t_transient=100.0,
    n_maximos_plot=200,
    rtol=1e-10,
    atol=1e-12
):
    """
    Construye un diagrama de bifurcación del sistema de Lorenz
    usando los máximos locales de z(t).

    Para cada valor de rho:
        1. Integra Lorenz.
        2. Elimina transitorio.
        3. Calcula máximos locales de z(t).
        4. Guarda los últimos máximos.

    Parámetros
    ----------
    rho_values : array_like
        Valores del parámetro rho.

    sigma, beta : float
        Parámetros del sistema de Lorenz.

    x0, y0, z0 : float
        Condición inicial.

    t_max : float
        Tiempo total de integración.

    dt : float
        Paso temporal de guardado.

    t_transient : float
        Tiempo eliminado como transitorio.

    n_maximos_plot : int
        Número de máximos finales que se guardan para cada rho.

    rtol, atol : float
        Tolerancias del integrador.

    Retorna
    -------
    RHO : ndarray
        Valores de rho repetidos.

    ZMAX : ndarray
        Máximos locales de z asociados a cada rho.
    """

    RHO = []
    ZMAX = []

    for rho in rho_values:

        t, x, y, z = lorenz_system(
            sigma=sigma,
            rho=rho,
            beta=beta,
            x0=x0,
            y0=y0,
            z0=z0,
            t_max=t_max,
            dt=dt,
            t_transient=t_transient,
            rtol=rtol,
            atol=atol,
            return_xyz=True
        )

        zmax, _ = local_maxima(z)

        if len(zmax) > n_maximos_plot:
            zmax = zmax[-n_maximos_plot:]

        RHO.extend([rho] * len(zmax))
        ZMAX.extend(zmax)

    return np.array(RHO), np.array(ZMAX)

"""ROSSLER"""

import numpy as np
from scipy.integrate import solve_ivp


def rossler_system(
    a=0.2,
    b=0.2,
    c=5.7,
    t_max=1000.0,
    dt=0.01,
    t_transient=0.0,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
    return_xyz=True
):
    """
    Integra el sistema de Rössler:

        dx/dt = -y - z
        dy/dt = x + a y
        dz/dt = b + z (x - c)

    usando un método numérico adaptativo de alta precisión.

    Parámetros
    ----------
    a, b, c : float
        Parámetros del sistema de Rössler.

    x0, y0, z0 : float
        Condiciones iniciales.

    t_max : float
        Tiempo total de integración.

    dt : float
        Paso temporal con el que se guarda la solución.
        El integrador usa pasos internos adaptativos.

    t_transient : float
        Tiempo inicial que se elimina como transitorio.

    method : str
        Método de integración. Recomendado: "DOP853".

    rtol : float
        Tolerancia relativa del integrador.

    atol : float
        Tolerancia absoluta del integrador.

    return_xyz : bool
        Si True, retorna t, x, y, z.
        Si False, retorna t, serie, donde serie tiene forma (N, 3).

    Retorna
    -------
    t : ndarray
        Tiempos después de eliminar el transitorio.

    x, y, z : ndarray
        Coordenadas del sistema de Rössler.

    o bien

    serie : ndarray, shape (N, 3)
        Trayectoria tridimensional.
    """

    if t_max <= 0:
        raise ValueError("t_max debe ser mayor que cero.")

    if dt <= 0:
        raise ValueError("dt debe ser mayor que cero.")

    if t_transient < 0:
        raise ValueError("t_transient debe ser mayor o igual que cero.")

    if t_transient >= t_max:
        raise ValueError("t_transient debe ser menor que t_max.")

    def f(t, state):
        x, y, z = state

        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        return [dx, dy, dz]

    # Tiempos donde se guarda la solución.
    # El integrador no usa necesariamente este paso internamente.
    t_eval = np.arange(0.0, t_max + dt, dt)

    # Evita que t_eval se pase ligeramente de t_max por redondeo numérico.
    t_eval = t_eval[t_eval <= t_max]

    sol = solve_ivp(
        f,
        t_span=(0.0, t_max),
        y0=[x0, y0, z0],
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    if not sol.success:
        raise RuntimeError(f"La integración falló: {sol.message}")

    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    # Eliminación del transitorio
    mask = t >= t_transient

    t_out = t[mask]
    x_out = x[mask]
    y_out = y[mask]
    z_out = z[mask]

    if return_xyz:
        return t_out, x_out, y_out, z_out

    serie = np.column_stack((x_out, y_out, z_out))

    return t_out, serie