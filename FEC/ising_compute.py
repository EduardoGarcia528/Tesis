import numpy as np
from numba import njit
import matplotlib.pyplot as plt 
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
from scipy.ndimage import convolve, generate_binary_structure

def red_inicial(L, p):
    U = np.random.random((L,L))
    red = np.zeros((L,L), dtype=np.int32)
    red[U >= p] = 1
    red[U < p] = -1
    return red

def get_energy(red, J=1):
    kernel = generate_binary_structure(2,1,)
    kernel[1,1] = False
    energia = J*np.sum(-red * convolve(red, kernel, mode='wrap'))
    return energia // 2  # cada par contado dos veces

#Metropolis

@njit
def metropolis(red, time_steps,BJ, energy, J =1):
    red = red.copy()
    magnetizacion = np.zeros(time_steps)
    energia = np.zeros(time_steps)
    for t in range(time_steps):
        print(t)
        x = np.random.randint(red.shape[0])
        y = np.random.randint(red.shape[1])
        spin_i = red[x,y]

        E_i = -J*spin_i*(red[x-1,y] + red[x,y-1] + red[x+1,y] + red[x,y+1])

        delta_E = 2*E_i

        if delta_E <= 0:
            red[x,y] *= -1
            energy += delta_E
        else:
            r = np.random.random()
            if r < np.exp(-BJ*delta_E):
                red[x,y] *= -1
                energy += delta_E

        magnetizacion[t] = np.sum(red)
        energia[t] = energy
    return magnetizacion, energia


def simulate_ising(L, p, time_steps, T, J=1):
    red = red_inicial(L, p)
    energy = get_energy(red, J=J)
    BJ = 1.0 / T # Kb = 1, T en unidades de J, entonces BJ = 1/T
    magnetizacion, energia = metropolis(red, time_steps, BJ, energy, J=J)
    return magnetizacion, energia

if __name__ == "__main__":
    L = 1000
    p = 0.75
    time_steps = 100_000
    T = 2.269
    J = 1

    magnetizaciones, energias = simulate_ising(L, p, time_steps, T, J=J)
    np.save("magnetizaciones.npy", magnetizaciones)
    np.save("energias.npy", energias)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(magnetizaciones)
    plt.title('Magnetización vs Tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Magnetización')

    plt.subplot(1, 2, 2)
    plt.plot(energias)
    plt.title('Energía vs Tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Energía')

    plt.tight_layout()
    plt.show()
