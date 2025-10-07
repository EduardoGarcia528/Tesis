import numpy as np
import matplotlib.pyplot as plt

# Logistic
def logistic_series(r, x0, N, burnin):
    x = np.empty(N + burnin, dtype=np.float64)
    x[0] = x0
    for i in range(N + burnin - 1):
        x[i+1] = r * x[i] * (1.0 - x[i])
    return x[burnin:]  

# Binning 
def bins_equal_freq_4(arr):

    n = len(arr)
    order = np.argsort(arr, kind='mergesort')
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    print(rank[order])
    labels = (rank * 4) // n
    labels = np.clip(labels, 0, 3).astype(np.int8)
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
        vertices = np.array([[0.0, 0.0],
                             [1.0, 0.0],
                             [1.0, 1.0],
                             [0.0, 1.0]], dtype=np.float64)

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

    # Serie logística
    xs = logistic_series(r=r, x0=x0, N=N, burnin=burnin)
    # xs = np.random.rand(N)
    # Etiquetas por cuartiles (frecuencias iguales)
    labels = bins_equal_freq_4(xs)
    print(len(labels[np.where(labels == 0)]))
    print(len(labels[np.where(labels == 1)]))
    print(len(labels[np.where(labels == 2)]))

    # Juego del caos (4 vértices)
    points = chaos_game_4(labels, alpha=alpha)

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    ax.scatter(points[:, 0], points[:, 1], s=0.2, linewidths=0, alpha=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Juego del caos (4 vértices) desde serie logística con bins por cuartiles")
    # Opcional: dibuja los vértices
    V = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    # V = np.array([[0,0],[2,0],[1,2]], dtype=float)
    ax.scatter(V[:,0], V[:,1], s=30, marker="s", edgecolor="k", facecolor="none")
    plt.show()
