import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def pasos_browniano(x, dt, D):
    # np.random.normal() soportado en numba, sin argumentos
    return x + np.sqrt(2*D*dt) * np.random.randn(len(x))

@njit
def segundo_momento(x_t):
    n = len(x_t)
    suma = 0.0
    for i in range(n):
        suma += x_t[i]*x_t[i]
    return suma / n

@njit
def promedio(array):
    suma = 0.0
    for i in range(len(array)):
        suma += array[i]
    return suma / len(array)

@njit
def main(M, N, dt, b,q, alpha=1.0):
    # Reloj browniano en y: parte siempre en 0 y busca cruce de umbral y=a
    y = np.zeros(M)
    x = np.zeros(M)
    history = [np.array([0.0]) for _ in range(M)]
    msd = np.zeros(N)

    for t in range(1,N):
        print(t)
        y = pasos_browniano(y, dt, D=1)
        # evolucionar y
        for i in range(M):
            # cruce de umbral (primer arribo) 
            if y[i] >= b:
                # print(i)
                if np.random.random() < q:
                    # memoria: elegir índice de evento 0..n_events (uniforme)
                    k = np.random.randint(0, len(history[i]) )
                    # print("k:", k)
                    # print("history[i]",history[i])
                    x[i] = history[i][k]
                else:
                    if np.random.random() < 0.5:
                        x[i] = x[i] + alpha
                    else:
                        x[i] = x[i] - alpha
                history[i] = np.concatenate((history[i], np.array([x[i]])))
                # renovar el reloj: reiniciar y para que los tiempos de espera sean i.i.d.
                y[i] = 0.0

        msd[t] = segundo_momento(x)

    return msd


if __name__ == '__main__':
    dt = 0.01
    T = 10000
    N = int(T/dt)
    M = 100000
    b = 1.0
    q = 0.5
    alpha = 1.0

    # msd = main(M, N, dt, b, q, alpha)
    t = np.arange(0,N) *dt

    # np.save("msd_boyer_espina_q_5.npy", msd)
    # msd = np.load("msd_boyer_q_8.npy")
    msd = np.load("msd_boyer_espina_q_5.npy")
    msd = np.load("msd_simulation_prueba.npy")
    m, b = np.polyfit(np.log(t[1:]), msd[1:], 1)
    # --- Graficar ---
    plt.figure(figsize=(8, 5))
    plt.plot(t[1:], msd[1:], lw=1.5, label="MSD(t) simulado")
    plt.plot(t[1:], ((1-q)/q)*(0.5*alpha*alpha*np.log(t[1:])) + b, 'k--', lw=1.5, label="MSD teórico")
    plt.plot(t[1:],m*np.log(t[1:]) + b)
    plt.xlabel("t (pasos)")
    plt.ylabel("MSD(t)")
    plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"MSD vs t para {M} caminantes (q={q}, α={alpha}) m = {m:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

