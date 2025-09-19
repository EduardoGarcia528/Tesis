import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def pasos_browniano(x0, dt, D):
    # np.random.normal() soportado en numba, sin argumentos
    return x0 + np.sqrt(2*D*dt) * np.random.normal()

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
def main(M, N, dt, a, D_y=1.0, step_std=1.0):
    # Reloj browniano en y: parte siempre en 0 y busca cruce de umbral y=a
    y = np.zeros(M)
    x = np.zeros(M)
    msd = np.zeros(N)

    for t in range(N):
        print(t)
        # evolucionar y
        for i in range(M):
            y_old = y[i]
            y_new = pasos_browniano(y_old, dt, D=D_y)

            # cruce de umbral (primer arribo) 
            if y_new >= a:
                # salto en x con varianza finita
                if np.random.rand() < 0.5:
                    x[i] += step_std
                else:
                    x[i] -= step_std
                # renovar el reloj: reiniciar y para que los tiempos de espera sean i.i.d.
                y[i] = 0.0
            else:
                # sin evento: continuar trayectoria
                y[i] = y_new

        msd[t] = segundo_momento(x)

    return msd


if __name__ == '__main__':
    dt = 0.001
    N = 100000
    M = 9000

    momento_2 = main(M, N, dt, a = 1.4)
    # np.save('CTRW5.npy',momento_2)
    # momento_2 = np.load('CTRW5.npy')

    time = np.linspace(0.0, N*dt, N)
    x2_theory = np.sqrt(time)  # comportamiento esperado
    coef = np.polyfit(np.log10(time[1:]), np.log10(momento_2[1:]),1)
    fitline = np.polyval(coef, np.log10(time[1:]))

    plt.figure(figsize=(8,5))
    plt.loglog(time[1:], momento_2[1:], label="Simulación")
    plt.loglog(time[1:], time[1:]**(1/2), 'r--', label=r'$\propto t^{1/2}$', linewidth=2, alpha=0.7)
    plt.xlabel("t")
    plt.ylabel(r"$\langle x^2(t) \rangle$")
    print(coef)
    # plt.title("CTRW con tiempos de espera de primer arribo browniano, pendiente: ", coef[0])
    plt.legend()
    plt.grid(True)
    plt.show()
