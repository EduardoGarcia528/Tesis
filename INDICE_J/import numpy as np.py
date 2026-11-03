import numpy as np
import mi_libreria as ml
# --------------------------------------------------
# Serie
# --------------------------------------------------
# a = ml.logistic_map(4.0, 10000)
a = np.random.rand(5_000_000)
# a = np.concatenate(([0], a, [0]))
print(a[0], a[-1])
a1 = a[1:]
a2 = a[:-1]

# --------------------------------------------------
# Transformadas
# --------------------------------------------------
A1 = np.fft.rfft(a1)
A2 = np.fft.rfft(a2)

L = len(a1)

w = 2 * np.pi * np.fft.rfftfreq(L, d=1)

# --------------------------------------------------
# Comprobación de la relación exacta
# --------------------------------------------------
A_teorica = np.exp(1j * w) * (
    A2
    - a[0]
    + a[-1] * np.exp(-1j * w * (len(a) - 1))
)

print("Relación Fourier correcta:",
      np.allclose(A1, A_teorica))

print("Error máximo:",
      np.max(np.abs(A1 - A_teorica)))


# ==================================================
# FASES Y PUNTOS P_k
# ==================================================

phi1 = np.angle(A1)
phi2 = np.angle(A2)

# P_k = (phi1(k), phi2(k))
P = np.column_stack((phi1, phi2))


# ==================================================
# VECTORES v_k = P_{k+1} - 2 P_k
# ==================================================

vectores, angulos = ml.angulos_alpha(seriex=a, seriey=None)

import matplotlib.pyplot as plt
plt.hist(angulos, bins=200, density=True, alpha=0.7, color='blue')
plt.xlabel('Ángulo (radianes)')
plt.ylabel('Frecuencia')
plt.title('Distribución de ángulos')
plt.show()