import numpy as np


def heaviside_clamp(z):
    return np.where(z > 0.0, z, 0.0)


def laplacian_periodic(field):
    up    = np.roll(field, -1, axis=0)
    down  = np.roll(field,  1, axis=0)
    right = np.roll(field, -1, axis=1)
    left  = np.roll(field,  1, axis=1)
    return up + down + right + left - 4.0 * field


def lv_cml_step(x, y, a=4.0, b=1.0, D1=0.10, D2=0.15):
    lap_x = laplacian_periodic(x)
    lap_y = laplacian_periodic(y)

    x_local = a * x * (1.0 - x - y)
    y_local = b * x*y  
    
    x_new = x_local + D1 * lap_x
    y_new = y_local + D2 * lap_y

    x_new = heaviside_clamp(x_new)
    y_new = heaviside_clamp(y_new)

    return x_new, y_new


def inicializar_lattice(N=100, seed=None):

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
    N = 200
    pasos = 25000

    x_final, y_final, snaps = simular_lv_cml(
        N=N,
        pasos=pasos,
        a=4.0,
        b=4.0,
        D1=0.099,
        D2=0.250,  
        seed=42,
        guardar_cada=1000,
    )

    import matplotlib.pyplot as plt
    plt.imshow(x_final, origin="lower")
    plt.colorbar()
    plt.show()
