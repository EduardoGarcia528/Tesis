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
    y_local = b * x * y 

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


def norm_state(dx, dy):
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

    rng = np.random.default_rng(seed)

    x, y = inicializar_lattice(N=N, seed=seed)

    for _ in range(pasos_transitorio):
        x, y = lv_cml_step(x, y, a=a, b=b, D1=D1, D2=D2)

    dx = rng.normal(0.0, 1.0, size=(N, N))
    dy = rng.normal(0.0, 1.0, size=(N, N))
    d_norm = norm_state(dx, dy)
    dx *= (delta0 / d_norm)
    dy *= (delta0 / d_norm)

    x_p = x + dx
    y_p = y + dy

    suma_logs = 0.0

    for _ in range(pasos_lyap):
        x,   y   = lv_cml_step(x,   y,   a=a, b=b, D1=D1, D2=D2)
        x_p, y_p = lv_cml_step(x_p, y_p, a=a, b=b, D1=D1, D2=D2)

        dx = x_p - x
        dy = y_p - y

        d_norm = norm_state(dx, dy)

        g = d_norm / delta0
        if g <= 0.0:
            continue

        suma_logs += np.log(g)

        dx *= (delta0 / d_norm)
        dy *= (delta0 / d_norm)

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