import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math
from collections import Counter

def rossler_system(state, a, b, c):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

# Método de Runge-Kutta de cuarto orden
def runge_kutta_step_rossler(state, dt, a, b, c):
    k1 = rossler_system(state, a, b, c)
    k2 = rossler_system(state + 0.5 * dt * k1, a, b, c)
    k3 = rossler_system(state + 0.5 * dt * k2, a, b, c)
    k4 = rossler_system(state + dt * k3, a, b, c)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def Rossler_generator(a,b,c,dt,transitorio,total_puntos):
    # Condición inicial
    initial_state = np.array([1.0, 1.0, 1.0])

    # Inicialización de arrays para almacenar resultados
    states = np.zeros((total_puntos, 3))
    states[0] = initial_state

    # Integración numérica
    for i in range(1, total_puntos):
        states[i] = runge_kutta_step_rossler(states[i - 1], dt, a, b, c)

    # Eliminar el transitorio
    states = states[transitorio:]

    # Extraer coordenadas x, y, z
    x, y, z = states[:, 0], states[:, 1], states[:, 2]

    # Graficar el atractor
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, lw=0.5, color='blue')
    # ax.set_title("Atractor de Rössler", fontsize=14)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    # plt.plot(x,y)

    return x,y,z

# Función del sistema de Lorenz
def lorenz_system(state, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

# Método de Runge-Kutta de cuarto orden
def runge_kutta_step_lorenz(state, dt, sigma, beta, rho):
    k1 = lorenz_system(state, sigma, beta, rho)
    k2 = lorenz_system(state + 0.5 * dt * k1, sigma, beta, rho)
    k3 = lorenz_system(state + 0.5 * dt * k2, sigma, beta, rho)
    k4 = lorenz_system(state + dt * k3, sigma, beta, rho)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def lorenz_generator(rho, dt, transitorio, total_puntos, sigma = 10.0, beta=8.0/3.0):
    # Condición inicial
    initial_state = np.array([1.0, 1.0, 1.0])

    # Inicialización de arrays para almacenar resultados
    states = np.zeros((total_puntos, 3))
    states[0] = initial_state

    # Integración numérica
    for i in range(1, total_puntos):
        states[i] = runge_kutta_step_lorenz(states[i - 1], dt, sigma, beta, rho)

    # Eliminar el transitorio
    states = states[transitorio:]

    # Extraer coordenadas x, y, z
    x, y, z = states[:, 0], states[:, 1], states[:, 2]

    # Graficar el atractor
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, lw=0.5, color='red')
    # ax.set_title("Atractor de Lorenz", fontsize=14)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    # plt.plot(x,y)
    # plt.show()

    return x, y, z



@njit
def distancia(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

@njit
def mejor_vector(p1, p2):
    # Precomputar diferencias en los 9 cuadrantes
    diffs = [
        [p2[0], p2[1]],
        [p2[0], p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1] + 2 * np.pi],
        [p2[0] + 2 * np.pi, p2[1]],
        [p2[0] + 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0], p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1] - 2 * np.pi],
        [p2[0] - 2 * np.pi, p2[1]],
        [p2[0] - 2 * np.pi, p2[1] + 2 * np.pi],
    ]
    # Encontrar el índice con menor distancia
    d_og = distancia(p1,p2)
    min_idx = 0
    for i in range(9):
        d = distancia(p1, diffs[i])
        if d < d_og:
            min_idx = i
            d_og = d
    p2 = diffs[min_idx]
    return [p2[0] - 2*p1[0], p2[1] - 2*p1[1]]

@njit
def calcular_angulos(vectores):
    n = len(vectores) - 1
    angulos = np.empty(n)
    for i in range(n):
        v1 = vectores[i]
        v2 = vectores[i + 1]
        norm_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if norm_v1 == 0 or norm_v2 == 0:
            angulo = 0.0
        else:
            v1n0 = v1[0] / norm_v1
            v1n1 = v1[1] / norm_v1
            v2n0 = v2[0] / norm_v2
            v2n1 = v2[1] / norm_v2
            dot = v1n0 * v2n0 + v1n1 * v2n1
            if dot > 1.0: dot = 1.0
            if dot < -1.0: dot = -1.0
            angulo = np.arccos(dot)
            cruz = v1[0] * v2[1] - v1[1] * v2[0]
            if cruz > 0:
                angulo = np.pi - angulo
            elif cruz == 0 and angulo < 0:
                angulo = np.pi
            elif cruz < 0:
                angulo += np.pi
        angulos[i] = angulo
    return angulos

def caminata_univariante(X, tau):
    x1 = X[tau:]
    y1 = X[:-tau]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    n = len(ff1) - 1
    vectores = np.empty((n, 2))
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def indice_J(angulos):
    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    return J

def entropia_shannon(x, bins=100):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    p = hist / hist.sum()
    H = -np.sum(p * np.log2(p))
    return H / np.log2(bins)

def entropia_permutacion(x, m=4, tau=1):
    n = len(x)
    if n < (m - 1) * tau + 1:
        return np.nan

    patrones = []
    for i in range(n - (m - 1) * tau):
        ventana = x[i:i + tau * m:tau]
        orden = tuple(np.argsort(ventana))
        patrones.append(orden)

    cuenta = Counter(patrones)
    total = sum(cuenta.values())
    p = np.array(list(cuenta.values())) / total
    H = -np.sum(p * np.log2(p))
    H_norm = H / np.log2(math.factorial(m))  # normalización
    return H_norm

def diff_S(d, angulos):
    for _ in range(d):
        dif_angulos = np.diff(angulos)

    return entropia_shannon(dif_angulos)

def main(X):
    vectores = caminata_univariante(X,tau = 1)
    angulos = calcular_angulos(vectores)
    entropia = diff_S(d=1,angulos=angulos)
    J = indice_J(angulos)
    return J, entropia

def lyapunov_from_trajectory(x, y, z, dt, a, b, c):
    n = len(x)
    if not (len(y) == len(z) == n):
        raise ValueError("x, y, z deben tener la misma longitud")

    # Vector de perturbación inicial (pequeño, pero normalizado)
    delta = np.array([1e-8, 0.0, 0.0])
    delta /= np.linalg.norm(delta)

    lyap_sum = 0.0

    for i in range(n):
        # Jacobiana evaluada en (x_i, y_i, z_i)
        J = np.array([
            [0.0, -1.0, -1.0],
            [1.0, a,    0.0],
            [z[i], 0.0, x[i] - c]
        ])

        # Evolucionar perturbación linealmente: delta -> delta + dt * J * delta
        delta_new = delta + dt * J @ delta

        # Norm y renormalización
        norm = np.linalg.norm(delta_new)
        delta = delta_new / norm

        # Acumular logaritmo del crecimiento
        lyap_sum += np.log(norm)

    # Exponente de Lyapunov promedio
    return lyap_sum / (n * dt)


from scipy import integrate
from numpy import *


class LorenzMap:
    """ The Lorenz map corresponds to advancing one unit of time over the integral curves of the Lorenz System.
        This class holds the parameters of the Lorenz Map and evaluate it along its directional derivative,
        which is computed via variational equation.
        The default parameters are chosen as the canonical ones in the initialization.

        It instantiates a callable object f_df, in such a way that f_df(xyz, w)
        returns two values f(xyz) and df(xyz, w), where
        f(xyz) is the solution of the Lorenz system starting at xyz after one unit of time and
        df(xyz, w) is the solution of the Lorenz variational equations starting at (xyz, w).
    """

    def __init__(_, sigma=10, rho=28, beta=8 / 3, h0=0.01):
        _.sigma, _.rho, _.beta = sigma, rho, beta
        _.h0 = h0

    @staticmethod
    def pack_variables(xyz, w):
        return concatenate((xyz, reshape(w, 9)), axis=0)

    @staticmethod
    def unpack_variables(xyzw):
        return xyzw[0:3], reshape(xyzw[3::], (3, 3))

    def variational_equation(_, xyzw, t=None):
        xyz, w = _.unpack_variables(xyzw)
        x, y, z = xyz

        dot_xyz = array([_.sigma * (-x + y),
                         x * (_.rho - z) - y,
                         x * y - _.beta * z])

        dot_w = array([[ -_.sigma, _.sigma,       0],
                       [_.rho - z,      -1,      -x],
                       [        y,       x, -_.beta]]) @ w

        return _.pack_variables(dot_xyz, dot_w)

    def __call__(_, xyz, w):
        xyzw = _.pack_variables(xyz, w)
        next_xyzw = integrate.odeint(_.variational_equation, xyzw, array([0, 1]), h0=_.h0)
        return _.unpack_variables(next_xyzw[1])

def lyapunov_max_lorenz(sigma=10, rho=28, beta=8/3, 
                        h0=0.01, tmax=100, tskip=10):
    """
    Calcula el mayor exponente de Lyapunov del sistema de Lorenz
    usando el método de Benettin con una sola columna de la matriz W.
    """
    lorenz_map = LorenzMap(sigma=sigma, rho=rho, beta=beta, h0=h0)

    # Estado inicial aleatorio
    xyz = np.random.rand(3) * 10.0

    # Matriz W inicial (identidad 3×3)
    W = np.eye(3)

    lyap_sum = 0.0
    steps = 0

    n_skip = int(tskip)
    n_total = int(tmax)

    for i in range(n_total):
        xyz, W_new = lorenz_map(xyz, W)

        # Tomamos solo la primera columna para el mayor exponente
        v = W_new[:, 0]
        norm = np.linalg.norm(v)
        if norm == 0:
            # Evitar división por cero
            v = np.random.rand(3)
            norm = np.linalg.norm(v)

        # Reemplazar la primera columna normalizada, mantener las demás igual
        W = np.eye(3)
        W[:, 0] = v / norm

        if i >= n_skip:
            lyap_sum += np.log(norm)
            steps += 1

    return lyap_sum / steps




#Valores de r alrededor del crítico 
# c_values = [4, 6, 12, 8.5, 12.6, 8.7, 13, 9, 18] 
c_values = [350.0, 100.5, 160.0, 99.65, 28.0]  # Valores de rho correspondientes
a = 0.1
b = 0.1
# c = 9.0
dt = 0.01  # Paso de integración
transitorio = 100_000  # Tiempo en pasos para eliminar el transitorio
total_puntos = 200_000

#Cálculo de entropía de res para cada r 
entropias = []
Js = []
lyapunovs = []
for c in c_values:
    dt = 1/100
    print(c)
    # if c == 18:
        # dt = 0.1
    # # if c == 18:
    #     # dt = 1/14
    # if c in [8.5,12.6]:
    #     dt = 1/200
    # serie, serie_y, serie_z = Rossler_generator(a,b,c,dt,transitorio,total_puntos)
    serie,serie_y, serie_z = lorenz_generator(c, dt, transitorio, total_puntos)
    print("ya")
    J,S = main(serie)
    print("ya2")
    # lyapunov = lyapunov_from_trajectory(serie, serie_y, serie_z, dt, a, b, c)
    lyapunov = lyapunov_max_lorenz(sigma=10, rho=c, beta=8/3, h0=0.01, tmax=200, tskip=50)
    entropias.append(S)
    Js.append(J)
    lyapunovs.append(lyapunov)

# === Graficar resultado ===
fig,ax1 = plt.subplots(figsize=(5,3))
ax1.set_ylabel('J',rotation=360,fontsize=10)
ax1.plot(entropias, color='orange', marker='.', linestyle='-', label='S')
ax2 = ax1.twinx()
# ax2.set_ylim(-0.1,1.0)
ax2.plot(lyapunovs, color='black', marker='.', linestyle='-', label=r'$\lambda$')
ax2.legend(loc = 'center left')
ax1.plot(Js, color='red', marker='.', linestyle='-', label='J')
ax1.legend()
ax1.set_xlabel('Parámetro rho')
# plt.ylabel('S ó J')
# plt.ylim(0,1.0)
ax1.get_xaxis().set_visible(False)
plt.title('Sistema Lorenz', fontsize=8)

# Añadir etiquetas
for i, (x, c) in enumerate(zip(range(len(c_values)), c_values)):
    ax1.text(x+0.05, entropias[i], f'rho={c}', fontsize=8, ha='right', va='bottom')

plt.show()