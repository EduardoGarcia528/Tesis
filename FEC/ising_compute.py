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

def get_energy(red):
    kernel = generate_binary_structure(2,1)
    kernel[1,1] = 0
    energia = np.sum(red * convolve(red, kernel, mode='wrap'))
    return -0.5*energia  # cada par contado dos veces
    # E/J

#Metropolis

@njit
def metropolis(red, time_steps,BJ, energy):
    red = red.copy()
    Lx, Ly = red.shape
    magnetizacion = np.zeros(time_steps)
    energia = np.zeros(time_steps)
    M = np.sum(red)
    for t in range(time_steps):
        # if t in [time_steps//4, time_steps//2, 3*time_steps//4]:
            # print(t)
        x = np.random.randint(0,Lx)
        y = np.random.randint(0,Ly)
        spin_i = red[x,y]
        xm = x - 1 if x > 0 else Lx - 1
        xp = x + 1 if x < Lx - 1 else 0
        ym = y - 1 if y > 0 else Ly - 1
        yp = y + 1 if y < Ly - 1 else 0
        E_i = spin_i*(red[xm, y] + red[xp, y] + red[x, ym] + red[x, yp])
        delta_E = 2*E_i

        if delta_E <= 0:
            red[x,y] *= -1
            energy += delta_E
            M += -2*spin_i
        else:
            r = np.random.random()
            if r < np.exp(-BJ*delta_E):
                red[x,y] *= -1
                energy += delta_E
                M += -2*spin_i

        magnetizacion[t] = M/(Lx*Ly)
        energia[t] = energy
    return magnetizacion, energia


def simulate_ising(L, p, time_steps, BJ):
    red = red_inicial(L, p)
    energy = get_energy(red)
    # BJ = 1.0 / T # Kb = 1, T en unidades de J, entonces BJ = 1/T
    magnetizacion, energia = metropolis(red, time_steps, BJ, energy)
    # print("ya")
    return magnetizacion, energia



def binder_cumulant(m_series):
    m2 = np.mean(m_series**2)
    m4 = np.mean(m_series**4)
    return 1.0 - m4 / (3.0 * m2 * m2)

if __name__ == "__main__":
    L = 100
    p = 0.75
    time_steps  = 10_200_000
    T = 2.269  #T critico ~ 2.269
    # T = 3.0
    BJ = 1.0 / T

    # magnetizaciones, energias = simulate_ising(L, p, time_steps, BJ)

    Tc = 2.269185314213
    T_coarse = np.linspace(1.5, 3.5, 20)
    T_fine = np.linspace(Tc - 0.05, Tc + 0.05, 25)
    T_list = np.unique(np.concatenate([T_coarse, T_fine]))


    for L, tamaño in zip([256, 128, 64, 32, 16],['256', '128', '64', '32', '16']):
        U_4_L = np.empty(len(T_list))
        print(L)
        for i,T in enumerate(T_list):
            BJ = 1.0 / T
            magnetizaciones, energias = simulate_ising(L, p, time_steps, BJ)
            U_4_L[i] = binder_cumulant(magnetizaciones)
        np.save('ising/U_4_'+tamaño+'.npy', U_4_L)

    
    # np.save("magnetizaciones.npy", magnetizaciones[200_000:])
    # np.save("energias.npy", energias[200_000:])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(magnetizaciones[200_000:])
    plt.title('Magnetización vs Tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Magnetización')

    plt.subplot(1, 2, 2)
    plt.plot(energias[200_000:])
    plt.title('Energía vs Tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Energía')

    plt.tight_layout()
    plt.show()
