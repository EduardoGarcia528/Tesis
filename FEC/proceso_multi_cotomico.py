import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def pasos_browniano(x0,dt,D):
    x1 = x0 + np.sqrt(2*D*dt)*np.random.normal(loc=0.0, scale=1.0, size=None)

    return x1

@njit
def segundo_momento(x_t):
    n = len(x_t)
    suma = 0.0
    for i in range(n):
        suma += x_t[i]*x_t[i]
    momento2 = suma / n
    return momento2

@njit
def main(M,N,dt,franja,lam):
    y0 = [0.0]*M
    x0 = [0.0]*M
    momento_2 = []
    for t in range(N): # N total
        print(t)
        y_t = []
        x_t = []
        
        for i in range(M): # particulas
            y_t.append(pasos_browniano(y0[i],dt,1))
            if franja[0] <= y0[i] <= franja[1]:
                # if np.random.rand() < lam*dt:
                if np.random.rand() < 0.5:
                    x_t.append(x0[i] + 1)
                else:
                    x_t.append(x0[i] - 1)
                # else:
                    # x_t.append(x0[i])
            else:
                x_t.append(x0[i])
                    
        momento_2.append(segundo_momento(x_t))
        y0 = y_t
        x0 = x_t
    return momento_2

if __name__ == '__main__':

    dt = 0.001
    N = 1000
    M = 400000
    lam = 5    # tasa de cambio
    franja = [-0.5,0.5]
    momento_2 = main(M,N,dt,franja,lam)
    # np.save("momento2_proceso_multi_cotomico.npy", momento_2)

    time = np.linspace(0.0,N*dt,N)
    x2_theory = np.sqrt(time)

    plt.figure(figsize=(8,5))
    plt.plot(time[1:], momento_2[:-1], label="Simulación")
    plt.plot(time, x2_theory, 'r--', label="Solución analítica", linewidth=2,alpha=0.7)
    plt.xlabel("t")
    plt.ylabel(r"$\langle x^2(t) \rangle$")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Segundo momento en el proceso dicotómico de difusión")
    plt.legend()
    plt.grid(True)
    plt.show()
