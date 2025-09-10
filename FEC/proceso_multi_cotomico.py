import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def simulate_brownian_FPT(x0=1.0, D=0.5, dt=1e-3, Tmax=100.0):
    """
    Simula trayectoria browniana paso a paso hasta el primer arribo a 0.
    Devuelve el tiempo de primer arribo.
    """
    x = x0
    t = 0.0
    sqrt_2Ddt = np.sqrt(2*D*dt)

    while t < Tmax:
        dx = sqrt_2Ddt * np.random.randn()   # numba soporta np.random.randn
        x += dx
        t += dt
        if x <= 0.0:   # primer arribo
            return t

    return Tmax  # si no llega en Tmax, truncamos

def simulate_CTRW_with_FPT(M=5000, T=200.0, dt=0.01, x0=1.0, D=0.5, 
                           step_sigma=1.0, seed=123, dt_brown=1e-3):
    """
    Simula CTRW donde los tiempos de espera provienen del FPT
    de trayectorias brownianas explícitas.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, T+1e-12, dt)
    msd = np.zeros_like(times)

    for m in range(M):
        print(m)
        t, x = 0.0, 0.0
        jump_times = [0.0]
        jump_positions = [0.0]

        while t < T:
            tau = simulate_brownian_FPT(x0=x0, D=D, dt=dt_brown)
            t_next = t + tau
            if t_next > T:
                break
            dx = rng.normal(0, step_sigma)
            x += dx
            t = t_next
            jump_times.append(t)
            jump_positions.append(x)

        # reconstrucción x(t) en malla uniforme
        k = 0
        x_curr = jump_positions[0]
        t_next_jump = jump_times[1] if len(jump_times) > 1 else np.inf
        for i, ti in enumerate(times):
            while ti > t_next_jump and k < len(jump_times)-1:
                k += 1
                x_curr = jump_positions[k]
                t_next_jump = jump_times[k+1] if (k+1)<len(jump_times) else np.inf
            msd[i] += x_curr**2

    msd /= M
    return times, msd

if __name__ == "__main__":
    times, msd = simulate_CTRW_with_FPT(M=2000, T=200.0, dt=0.1, 
                                        x0=1.0, D=0.5, dt_brown=1e-3)

    # Ajuste log-log
    mask = times > 10
    slope, intercept = np.polyfit(np.log(times[mask]), np.log(msd[mask]), 1)
    print(f"Pendiente estimada ≈ {slope:.3f} (esperado: 0.5)")

    # Graficar
    plt.loglog(times[1:], msd[1:], label="MSD con FPT simulado")
    t_ref, msd_ref = times[-1], msd[-1]
    guide = msd_ref*(times[1:]/t_ref)**0.5
    plt.loglog(times[1:], guide, '--', label=r'$\propto t^{1/2}$')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel(r"$\langle x^2(t)\rangle$")
    plt.show()
