import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def pasos_dicotomico(x0, v0, dt,lam):
    x1 = x0 + v0*dt
    # con probabilidad lam*dt se invierte la velocidad
    if np.random.rand() < lam*dt:
        v0 = -v0
    return x1, v0

@njit
def segundo_momento(x_t):
    n = len(x_t)
    suma = 0.0
    for i in range(n):
        suma += x_t[i]*x_t[i]
    momento2 = suma / n
    return momento2

@njit
def main_estocastico(M,N,dt,v0,lam):
    x0 = [0.0]*M
    vstate = np.empty(M)

    # Estado inicial aleatorio: +v o -v
    for i in range(M):
        if np.random.rand() < 0.5:
            vstate[i] = v0
        else:
            vstate[i] = -v0
    momento_2 = []
    for t in range(N): # N total
        print(t)
        x_t = []
        
        for i in range(M): # particulas
            paso, vstate[i] = pasos_dicotomico(x0[i], vstate[i], dt, lam)
            x_t.append(paso)
        momento_2.append(segundo_momento(x_t))
        x0 = x_t
    return momento_2



if __name__ == '__main__':

    dt = 0.01
    N = 10000
    M = 10000
    v0 = 1.0       # velocidad fija
    lam = 0.001    # tasa de cambio
    momento_2 = main_estocastico(M,N,dt, v0, lam)

    time = np.linspace(0.0,N*dt,N)
    x2_theory = (v0**2 / (2*lam**2)) * (2*lam*time - 1 + np.exp(-2*lam*time)) 

    plt.figure(figsize=(8,5))
    plt.plot(time[1:], momento_2[:-1], label="Simulación", alpha=0.7)
    plt.plot(time, x2_theory, 'r--', label="Solución analítica", linewidth=2)
    plt.xlabel("t")
    plt.ylabel(r"$\langle x^2(t) \rangle$")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Segundo momento en el proceso dicotómico de difusión")
    plt.legend()
    plt.grid(True)
    plt.show()
