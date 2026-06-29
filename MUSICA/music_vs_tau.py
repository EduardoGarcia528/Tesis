import numpy as np
import mi_libreria as ml
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
import automutualinformation as MI

def autocorrelacion_tau(x, tau=1):
    """
    Calcula la autocorrelación normalizada de una serie x para un retardo tau.

    Parámetros
    ----------
    x : array_like
        Serie de tiempo.
    tau : int
        Retardo temporal.

    Retorna
    -------
    rho : float
        Autocorrelación normalizada C(tau).
    """

    x = np.asarray(x, dtype=float)

    if tau < 0:
        raise ValueError("tau debe ser >= 0")

    if tau >= len(x):
        raise ValueError("tau debe ser menor que la longitud de la serie")

    x_centered = x - np.mean(x)

    numerador = np.sum(x_centered[:-tau] * x_centered[tau:]) if tau > 0 else np.sum(x_centered**2)
    denominador = np.sum(x_centered**2)

    if denominador == 0:
        return np.nan

    return numerador / denominador

import numpy as np

def autoinformacion_mutua_tau(x, tau=1, bins="sturges", base=2, normalized=False):
    """
    Calcula la autoinformación mutua I(x_t ; x_{t+tau}) de una serie.

    Parámetros
    ----------
    x : array_like
        Serie de tiempo.
    tau : int
        Retardo temporal.
    bins : int or str
        Número de bins para discretizar la serie.
        También puede ser 'fd', 'sturges', 'sqrt', etc.
    base : float
        Base del logaritmo. base=2 da bits, base=np.e da nats.
    normalized : bool
        Si True, devuelve I / min(Hx, Hy).

    Retorna
    -------
    I : float
        Autoinformación mutua para el retardo tau.
    """

    x = np.asarray(x, dtype=float)

    if tau < 0:
        raise ValueError("tau debe ser >= 0")

    if tau >= len(x):
        raise ValueError("tau debe ser menor que la longitud de la serie")

    if tau == 0:
        x1 = x
        x2 = x
    else:
        x1 = x[:-tau]
        x2 = x[tau:]

    # Eliminar NaN o infinitos
    mask = np.isfinite(x1) & np.isfinite(x2)
    x1 = x1[mask]
    x2 = x2[mask]

    if len(x1) == 0:
        return np.nan

    # Histograma conjunto
    pxy, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins, density=False)

    pxy = pxy / np.sum(pxy)

    # Marginales
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Evitar log(0)
    mask = pxy > 0

    px_py = px[:, None] * py[None, :]

    I = np.sum(
        pxy[mask] * np.log(pxy[mask] / px_py[mask])
    )

    # Cambio de base
    I = I / np.log(base)

    if normalized:
        Hx = -np.sum(px[px > 0] * np.log(px[px > 0])) / np.log(base)
        Hy = -np.sum(py[py > 0] * np.log(py[py > 0])) / np.log(base)

        Hmin = min(Hx, Hy)

        if Hmin == 0:
            return np.nan

        I = I / Hmin

    return I

for num in range(1,24):
    melody = np.load(f"melodies/{str(num)}.npy")

    # C,g = ml.gamma_index_rank_ties(melody,6,mu=2)
    S = []
    S_null = []
    for i in range(3,8):
        # S.append(ml.H_orbit(melody, m=i))
        # S_null.append(ml.H_orbit(np.random.permutation(melody), m=i))
        S.append(ml.modified_permutation_entropy(melody, m=i, tau=1,norm=False))
        S_null.append(ml.modified_permutation_entropy(np.random.permutation(melody), m=i, tau=1,norm=False))
        # S.append(ml.indice_S_eff_fast(melody, tau=i, delta=False))
        # S_null.append(ml.indice_S_eff_fast(np.random.permutation(melody), tau=i, delta=False))

    H = []
    H_null = []
    auto = []
    auto_null = []
    tau = range(3,8)
    for t in range(3,8):
    #     dmelody = np.diff(melody)
    #     # H.append(ml.indice_S_eff_fast(dmelody, tau=t,delta=False))
        melody_null = np.random.permutation(melody)
    #     dmelody_null = np.diff(melody_null)
    #     H.append(ml.indice_S_eff_fast(melody, tau=t,delta=False))
    #     H_null.append(ml.indice_S_eff_fast(melody_null,tau=t, null = "no",delta=False))
        auto.append(a)
    #     # auto.append(autocorrelacion_tau(melody,tau=t))
        a_null = autoinformacion_mutua_tau(melody_null, tau=t)
        auto_null.append(a_null)
    #     # auto_null.append(autocorrelacion_tau(np.random.permutation(melody), tau=t))
    # print(np.mean(H), np.mean(H_null))
    # print("spearman: ", spearmanr(auto,H), "pearson: ", pearsonr(auto,H))
    # plt.plot(tau,H, label='J_h',color='red')
    # plt.plot(tau, H_null, label='J shuffle')
    # plt.ylim(-0.5,0.5)
    # plt.plot(tau, 1-g, label='gamma')
    plt.plot(tau, auto_null,'.-', label = 'Autocorrelation null')
    plt.plot(tau, auto, '.-',label='Autocorrelation')
    # plt.plot(tau, S, '.-', label='S_eff')
    # plt.plot(tau, np.array(S_null), '.-', label='S_eff null')
    plt.legend()
    plt.title(str(num))
    plt.show()
