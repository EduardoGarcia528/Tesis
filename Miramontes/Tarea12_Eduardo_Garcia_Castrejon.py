import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def logistic_exponential_map(r, x0, n, l):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] * np.exp(r*(1 - x[i-1])) + l
    return x[3*n//4:]

@njit
def detect_period(x, max_period=4, tol=1e-3):
    for p in range(1, max_period + 1):
        if np.all(np.abs(x[-p:] - x[-2*p:-p]) < tol):
            if p == 3:
                return 0
            return p
    return 0

@njit
def senoidal_map(r, x0, n, l):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = np.sin(l * x[i-1]) + r
    return x[3*n//4:]




# Mapa Logístico-Exponencial (Periodo reverso) 
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for r in np.linspace(1.8, 4.0, 1000):
    x = logistic_exponential_map(r, 0.5, 1000, 0.0)
    axs[0].plot([r]*len(x), x, ',b', alpha=1)
axs[0].set_title("Mapa Logístico-Exponencial")
axs[0].set_ylabel("x")

# Segundo subplot
for r in np.linspace(1.8, 4.0, 1000):
    x = logistic_exponential_map(r, 0.5, 1000, 0.06)
    axs[1].plot([r]*len(x), x, ',b', alpha=1.0)
axs[1].set_title("Mapa Logístico-Exponencial (Periodo reverso)")
axs[1].set_xlabel("r")
axs[1].set_ylabel("x")

plt.tight_layout()
plt.show()


# Mapa senoidal
for r in np.linspace(0.0, 6.5, 1000):
    x = senoidal_map(r, 0.5, 1000, 2)
    plt.plot([r]*len(x), x, ',b', alpha=1)
plt.title("Mapa Logístico-senoidal")
plt.ylabel("x")
plt.xlabel("r")
plt.show()


# Atractor del mapeo logístico-exponencial ($\lambda=0.06$)
x = np.linspace(0, 3.5, 1000)
for r in [1.7,2.5, 2.7, 3.0, 3.5, 4.0]:
    y = x * np.exp(r*(1 - x)) + 0.06
    plt.plot(x, y, label =f"r={r}")
plt.plot(x, x, 'k--', label='y=x')
plt.title(r'Atractor del mapeo logístico-exponencial ($\lambda=0.06$)')
plt.xlabel('x_n')
plt.ylabel('x_{n+1}')
plt.legend()
plt.show()



# Regiones de periodicidad en el plano (lamda, r)
r_min, r_max, nr = 0.0, 5.0, 500
lam_min, lam_max, nlam = 0.00375, 1.20, 500
iters = 1000      # iteraciones totales por punto
x0 = 0.5          # condición inicial

r_vals = np.linspace(r_min, r_max, nr)
lam_vals = np.linspace(lam_min, lam_max, nlam)
matriz = np.zeros((nlam, nr, 3))
for fila in reversed(range(nlam)):
    matriz[fila, :, 0] = lam_vals[fila]
for columna in range(nr):
    matriz[:, columna, 1] = r_vals[columna]

for colum in range(nr):
    for fila in range(nlam):
        x = logistic_exponential_map(matriz[fila,colum,1], x0, iters, matriz[fila,colum,0])
        p = detect_period(x, max_period=4, tol=1e-3)
        matriz[fila, colum, 2] = p

P = matriz[:, :, 2]
changes = P[1:, :] != P[:-1, :] 


# GRAFICAR FRONTERAS

i_idx, j_idx = np.where(changes)
lam_front = 0.5 * (lam_vals[i_idx] + lam_vals[i_idx + 1])  # punto medio en lambda
r_front = r_vals[j_idx]

# --- Plot ---
plt.figure(figsize=(9,6))
plt.scatter(r_front, lam_front, s=6, marker='s',color='black', linewidths=0)

plt.xlabel(r'$r$')
plt.ylabel(r'$\lambda$')
plt.title(r'Fronteras de cambio de periodo en el plano $(\lambda, r)$ del mapeo logístico-exponencial')
plt.xlim(r_min, r_max)
plt.ylim(lam_min, lam_max)
plt.tight_layout()
plt.show()


