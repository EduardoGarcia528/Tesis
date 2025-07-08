import numpy as np
import matplotlib.pyplot as plt

def list_vector(n1, n2):
    n = n1 * n2
    ne = np.zeros((n, 4), dtype=int)
    for i2 in range(n2):
        m2 = i2 * n1
        for i1 in range(n1):
            m1 = m2 + i1
            ne[m1, 0] = m2 + (i1 + 1) % n1
            ne[m1, 1] = ((i2 + 1) % n2) * n1 + i1
            ne[m1, 2] = m2 + (i1 - 1 + n1) % n1
            ne[m1, 3] = ((i2 - 1 + n2) % n2) * n1 + i1
    return ne

def inits(init, n):
    if init == 0:
        s = np.where(np.random.rand(n) > 0.5, 1.0, -1.0)
    else:
        s = np.ones(n)
    return s

def ising(beta, s, ne, h=0.0):
    n = len(s)
    nhit = 0
    for _ in range(n):
        m = np.random.randint(n)
        ds = 2 * s[m] * (np.sum(s[ne[m]]) + h)
        if np.random.rand() < np.exp(-beta * ds):
            s[m] *= -1
            nhit += 1
    hit_ratio = nhit / n
    return s, hit_ratio

def magnet(s):
    return np.mean(s)

def hamiltonian(s, ne, h=0.0):
    energy = 0.0
    for i in range(len(s)):
        energy += - s[i] * (np.sum(s[ne[i]]) + h)
    return energy / len(s)

def main():
    n1, n2 = 32, 32
    beta = 0.4407   # Puedes cambiar la temperatura: beta = 1 / T
    therm = 500
    niter = 1000000
    init = 0
    h = 0.0
    n = n1 * n2

    ne = list_vector(n1, n2)

    print(f"size: {n1} {n2} hot(0)/cold(1): {init} Therm: {therm} NConf: {niter} h: {h:.2f}")

    s = inits(init, n)

    # Termalización
    for _ in range(therm):
        s, _ = ising(beta, s, ne, h)

    magnetization_series = []
    energy_series = []

    for i in range(niter):
        s, _ = ising(beta, s, ne, h)
        mgn = magnet(s)
        energy = hamiltonian(s, ne, h)
        magnetization_series.append(mgn)
        energy_series.append(energy)
        if i % 100 == 0:
            print(f"Step {i}: M = {mgn:.4f}, E = {energy:.4f}")

    # Guardar en archivos
    np.save("magnetization_time_series.npy", magnetization_series)
    np.save("energy_time_series.npy", energy_series)

    # Graficar magnetización
    plt.figure(figsize=(10, 6))
    plt.plot(magnetization_series, label=f'β={beta:.4f} (T={1/beta:.4f})')
    plt.xlabel('Monte Carlo step')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs. Monte Carlo time')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("magnetization_time_series.png")
    plt.show()

if __name__ == "__main__":
    main()
