import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

# ---------------- utilidades base ----------------

def red_inicial(L, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.rand(L, L)
    red = np.where(U < p, -1, 1).astype(np.int8)
    return red

def get_energy(spins):
    kernel = generate_binary_structure(2, 1).astype(np.int8)
    kernel[1,1] = 0
    nb = convolve(spins, kernel, mode='wrap')
    return -0.5 * np.sum(spins * nb)   # adimensional (J=1)

def binder_cumulant(m_series):
    m2 = np.mean(m_series**2)
    m4 = np.mean(m_series**4)
    return 1.0 - m4 / (3.0 * m2 * m2)

def block_average(series, block_size):
    series = np.asarray(series)
    n_blocks = len(series) // block_size
    if n_blocks == 0:
        return series.copy()
    series = series[:n_blocks*block_size]
    return series.reshape(n_blocks, block_size).mean(axis=1)

# ------------- Metropolis por SWEeps (numba) -------------

@njit
def metropolis_sweeps(spins, sweeps, beta):
    """
    Devuelve magnetización (por espín) y energía (adim) por sweep.
    Usa tabla de aceptación para ΔE in {+4,+8} (J=1).
    """
    Lx, Ly = spins.shape
    N = Lx * Ly

    # estado inicial (incremental)
    M = np.int64(0)
    for i in range(Lx):
        for j in range(Ly):
            M += spins[i, j]
    # energía inicial no dentro (se pasa desde Python si quieres ahorrar)
    # pero la calculamos aquí para mantener numba puro:
    # Nota: aquí un estimador rápido de energía local por links derecha/abajo
    E = 0.0
    for i in range(Lx):
        ip = i+1 if i < Lx-1 else 0
        for j in range(Ly):
            jp = j+1 if j < Ly-1 else 0
            s = spins[i, j]
            E -= s * (spins[ip, j] + spins[i, jp])
    # J=1 => E ya está a mitad (cada enlace contado una vez), correcto.

    m_out = np.empty(sweeps, dtype=np.float64)
    e_out = np.empty(sweeps, dtype=np.float64)

    # tabla de aceptación
    # ΔE posibles en 2D (J=1): -8,-4,0,+4,+8 ; solo +4 y +8 necesitan prob < 1
    p4 = np.exp(-beta * 4.0)
    p8 = np.exp(-beta * 8.0)

    for t in range(sweeps):
        for _ in range(N):
            x = np.random.randint(0, Lx)
            y = np.random.randint(0, Ly)
            s = spins[x, y]

            xm = x - 1 if x > 0 else Lx - 1
            xp = x + 1 if x < Lx - 1 else 0
            ym = y - 1 if y > 0 else Ly - 1
            yp = y + 1 if y < Ly - 1 else 0

            nb_sum = spins[xm, y] + spins[xp, y] + spins[x, ym] + spins[x, yp]
            dE = 2.0 * s * nb_sum  # adim

            accept = False
            if dE <= 0.0:
                accept = True
            else:
                # usa tabla
                if dE == 4.0:
                    if np.random.random() < p4:
                        accept = True
                elif dE == 8.0:
                    if np.random.random() < p8:
                        accept = True
                else:
                    # dE puede ser 0 (ya capturado) o valores raros por errores
                    if np.random.random() < np.exp(-beta * dE):
                        accept = True

            if accept:
                spins[x, y] = -s
                M += -2 * s      # magnetización total
                E += dE / 2.0    # OJO: nuestra E cuenta cada enlace una vez,
                                 # y dE estaba con vecinos dobles -> dividir 2

        m_out[t] = M / float(N)
        e_out[t] = E / float(N)   # por espín

    return m_out, e_out

# ------------- pipeline: eq -> medición -> binning -> U4 -------------

def run_one(L, T, sweeps_eq, sweeps_mc, p=0.5, seed=None,
            block_size_sweeps=500):
    beta = 1.0 / T
    spins = red_inicial(L, p, seed)
    # equilibrar
    _m_eq, _e_eq = metropolis_sweeps(spins, sweeps_eq, beta)
    # medir
    m, e = metropolis_sweeps(spins, sweeps_mc, beta)

    # binning en sweeps (reduce autocorrelación)
    if block_size_sweeps > 1:
        m_b = block_average(m, block_size_sweeps)
    else:
        m_b = m
    U4 = binder_cumulant(m_b)
    return U4, m_b, e  # devuelvo m_b por si quieres inspeccionar

# ------------------------ main demo ------------------------

if __name__ == "__main__":
    # rejilla de T: gruesa + fina cerca de Tc
    Tc = 2.26918531
    T_coarse = np.linspace(1.5, 3.5, 20)
    T_fine   = np.linspace(Tc-0.05, Tc+0.05, 41)
    T_list   = np.unique(np.concatenate([T_coarse, T_fine]))

    # parámetros por tamaño (más sweeps para L grande)
    sizes = [256]
    sweeps_eq_base = 2000     # puedes subir a 1e4 en Tc
    sweeps_mc_base = 20000    # mediciones por T (sweeps), sube si necesitas
    block_per_L    = {16:100, 32:200, 64:400, 128:800}  # tamaño de bloque (sweeps)

    # corridas independientes para promediar (reduce ruido)
    n_runs = 5

    all_U4 = {}
    for L in sizes:
        U4_vs_T = []
        print(f"L={L}")
        for T in T_list:
            U4_runs = []
            for r in range(n_runs):
                seed = 1000 + 17*L + 37*r  # semillas distintas
                sweeps_eq = sweeps_eq_base + 10*L   # más eq para L grande
                sweeps_mc = sweeps_mc_base
                U4, _, _ = run_one(
                    L, T, sweeps_eq, sweeps_mc, p=0.5, seed=seed,
                    block_size_sweeps=block_per_L[L]
                )
                U4_runs.append(U4)
            U4_vs_T.append(np.mean(U4_runs))
        all_U4[L] = np.array(U4_vs_T)

    # guardar si quieres
    for L, U4 in all_U4.items():
        np.save(f"U_4_{L}.npy", U4)
