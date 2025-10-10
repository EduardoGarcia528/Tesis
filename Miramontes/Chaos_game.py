import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# Logistic
def logistic_series(r, x0, N, burnin):
    x = np.empty(N + burnin, dtype=np.float64)
    x[0] = x0
    for i in range(N + burnin - 1):
        x[i+1] = r * x[i] * (1.0 - x[i])
    return x[burnin:]  

#Henon
@njit
def henon_map(a, n_points, n_trans=1000, b=0.3, x0=0.1, y0=0.1):
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

def lorenz_rhs(x, y, z, sigma=10.0, rho=30.0, beta=8/3):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def lorenz_rk4(N, dt=0.011, x0=1.0, y0=1.0, z0=1.0,
               sigma=10.0, rho=30.0, beta=8/3):
    """
    Integración del sistema de Lorenz con Runge-Kutta 4 (paso fijo).
    Devuelve: t, x, y, z (arrays de longitud N+1).
    """
    t = np.arange(N+1) * dt
    x = np.empty(N+1); y = np.empty(N+1); z = np.empty(N+1)
    x[0], y[0], z[0] = x0, y0, z0

    for i in range(N):
        k1x, k1y, k1z = lorenz_rhs(x[i], y[i], z[i], sigma, rho, beta)

        k2x, k2y, k2z = lorenz_rhs(
            x[i] + 0.5*dt*k1x,
            y[i] + 0.5*dt*k1y,
            z[i] + 0.5*dt*k1z,
            sigma, rho, beta
        )

        k3x, k3y, k3z = lorenz_rhs(
            x[i] + 0.5*dt*k2x,
            y[i] + 0.5*dt*k2y,
            z[i] + 0.5*dt*k2z,
            sigma, rho, beta
        )

        k4x, k4y, k4z = lorenz_rhs(
            x[i] + dt*k3x,
            y[i] + dt*k3y,
            z[i] + dt*k3z,
            sigma, rho, beta
        )

        x[i+1] = x[i] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        y[i+1] = y[i] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        z[i+1] = z[i] + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)

    return t, x, y, z

# Binning 
def bins_equal_freq_4(arr):

    n = len(arr)
    qs = np.quantile(arr, [0.25, 0.5, 0.75], method="linear")
    labels = np.searchsorted(qs, xs, side="left")  # 0,1,2,3
    return labels


# 3) Juego del caos para 4 vértices
def chaos_game_4(labels, alpha=0.5, vertices=None, start=(0.5, 0.5)):
    """
    labels: array de enteros en {0,1,2,3}
    alpha: fracción de avance hacia el vértice
    vertices: 4x2, si None usa cuadrado unitario en sentido horario
    """
    if vertices is None:
        # Asociaremos: 0->(0,0), 1->(1,0), 2->(1,1), 3->(0,1)
        vertices = np.array([[0.0, 1.0],
                             [1.0, 0.0],
                             [1.0, 1.0],
                             [0.0, 0.0]], dtype=np.float64)

    pts = np.empty((len(labels), 2), dtype=np.float64)
    x, y = float(start[0]), float(start[1])
    for i, lab in enumerate(labels):
        vx, vy = vertices[lab]
        x = alpha * x + alpha * vx
        y = alpha * y + alpha * vy
        pts[i, 0] = x
        pts[i, 1] = y
    return pts

# ---------------------------
# 4) Ejecutar y visualizar
# ---------------------------
if __name__ == "__main__":
    # Parámetros
    r = 4.0
    x0 = 0.327
    N = 5_000_000           # >= 5000 como pediste

    burnin = 2000
    alpha = 0.5


    # xs = logistic_series(r=r, x0=x0, N=N, burnin=burnin)
    # xs = np.random.rand(N)
    # xs, ys = henon_map(a=1.4, b= 0.3, n_points=10_000_000)

    # t,xs,ys,zs = lorenz_rk4(10_000_000)
    xs = np.loadtxt('series/temp_madison.txt')

    # Etiquetas por cuartiles (frecuencias iguales)
    labels = bins_equal_freq_4(xs)
    print(len(labels[np.where(labels == 0)]))
    print(len(labels[np.where(labels == 1)]))
    print(len(labels[np.where(labels == 2)]))

    # Juego del caos (4 vértices)
    points = chaos_game_4(labels, alpha=alpha)

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    ax.scatter(points[:, 0], points[:, 1], s=0.2, linewidths=0.2, alpha=1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Juego del caos (4 vértices) desde serie logística con bins por cuartiles")
    # Opcional: dibuja los vértices
    V = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    # V = np.array([[0,0],[2,0],[1,2]], dtype=float)
    ax.scatter(V[:,0], V[:,1], s=30, marker="s", edgecolor="k", facecolor="none")
    plt.show()
