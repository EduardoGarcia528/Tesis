import numpy as np
from numba import njit


# tus funciones
def lyapunov_exponent_from_henon_orbit(xs, ys, a, b):
    v1, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])  
    sum_ln1, sum_ln2 = 0.0, 0.0
    n = len(xs)
    
    for i in range(n):
        dfdx = -2 * a * xs[i]
        dfdy = 1
        dgdx = b
        dgdy = 0
        J = np.array([[dfdx, dfdy],[dgdx, dgdy]])
        v1 = np.dot(J, v1)
        v2 = np.dot(J, v2)
        v1_norm = np.linalg.norm(v1)
        v1 = v1 / v1_norm
        v2_proj = np.dot(v1, v2) * v1
        v2 = v2 - v2_proj
        v2_norm = np.linalg.norm(v2)
        v2 = v2 / v2_norm
        sum_ln1 += np.log(v1_norm)
        sum_ln2 += np.log(v2_norm)
    le1 = sum_ln1 / n
    le2 = sum_ln2 / n
    return max(le1, le2)

@njit
def lyapunov_altern(a,b,N):
    x, y = 0.1, 0.1  # condición inicial
    v = np.array([1.0, 0.0])
    suma = 0

    for n in range(N):
        # Jacobiano
        J = np.array([[-2*a*x, 1],
                    [b,      0]])
        
        # Propagación del vector
        v = J @ v
        factor = np.sqrt(v[0]**2 + v[1]**2)
        v = v / factor
        
        if n >= 100_000:
            suma += np.log(factor)
        
        # Iterar mapeo de Hénon
        x_new = 1 - a*x**2 + y
        y_new = b*x
        x, y = x_new, y_new

    lambda1 = suma / N
    return lambda1


def henon_map(a, b, n, trans):
    total = n + trans
    xs = np.zeros(total)
    ys = np.zeros(total)
    xs[0] = 0.1
    ys[0] = 0.1
    for i in range(1, total):
        xs[i] = 1 - a * xs[i-1]**2 + ys[i-1]
        ys[i] = b * xs[i-1]
    return xs[trans:], ys[trans:]

# -----------------------------------------------
# Búsqueda de a crítica con Lyapunov
b = 0.3
a_min = 1.0576244479
tol = 1e-10   # tolerancia para a
max_iter = 16 # máximo de iteraciones de bisección
n_iter = 300_000
trans = 100_000

xs, ys = henon_map(a_min, b, n_iter, trans)
le = lyapunov_exponent_from_henon_orbit(xs, ys, a_min, b)
# le = lyapunov_altern(a_min,b,N=400_000)

for k in range(10,max_iter):
    print(a_min)
    # le_new = 0.0
    print(1*10**(-k))
    dx = np.arange(1,10,1)
    a_mid = np.round(a_min + dx*10**(-k), k)
    le_new = np.zeros((len(a_mid)))
    for i in range(len(a_mid)):
        xs, ys = henon_map(a_min, b, n_iter, trans)
        le_new[i] = lyapunov_exponent_from_henon_orbit(xs, ys, round(a_mid[i], k), b)
        # le_new[i] = lyapunov_altern(a_mid[i],b,N=400_000)
    print(le_new)
    a_min = a_mid[np.argmin(np.abs(le_new))]
    # while abs(le) >= abs(le_new):
    #     a_mid = round(a_min - 1*10**(-k),k )
    #     # xs, ys = henon_map(a_mid, b, n_iter, trans)
    #     # le_new = lyapunov_exponent_from_henon_orbit(xs, ys, a_mid, b)
    #     le_new = lyapunov_altern(a_mid,b,N=400_000)
    #     print(a_mid,abs(le) - abs(le_new))
    #     if abs(le) < abs(le_new):
    #         a_mid = round(a_min + 1*10**(-k), k)
    #         # xs, ys = henon_map(a_mid, b, n_iter, trans)
    #         # le_new = lyapunov_exponent_from_henon_orbit(xs, ys, a_mid, b)
    #         le_new = lyapunov_altern(a_mid,b,N=400_000)
    #         print(a_mid,abs(le) - abs(le_new))
    #     if abs(le) > abs(le_new):
    #         a_min = a_mid
    #         le = le_new

a_crit = a_mid
print(f"a_crítica (Lyapunov≈0) ≈ {a_crit}")
