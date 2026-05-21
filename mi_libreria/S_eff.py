import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
from tqdm import tqdm
from mi_libreria import construir_puntos_toro, construir_vectores_geodesicos, calcular_angulos_entre_vectores, wrap_pi
from scipy.signal import hilbert

def shuffle_vectores(vectores, tau):
    N = len(vectores[:,0])
    indices = np.arange(N)
    np.random.shuffle(indices)
    dtheta_shuffled1 = vectores[indices, 0]  
    dtheta_shuffled2 = vectores[indices, 1]  
    vectores_shuffled = np.empty((N-np.abs(tau), 2), dtype=np.float64)
    if tau > 0:
        vectores_shuffled[:,0] = dtheta_shuffled1[tau:]
        vectores_shuffled[:,1] = dtheta_shuffled2[:-tau]
    elif tau < 0:
        vectores_shuffled[:,0] = dtheta_shuffled1[:tau]
        vectores_shuffled[:,1] = dtheta_shuffled2[-tau:]
    else:
        vectores_shuffled = np.column_stack((dtheta_shuffled1, dtheta_shuffled2))
    return vectores_shuffled


def obtener_fases_instantaneas(
    seriex,
    seriey=None,
    tau=1,
    delta = False,
    modo_univariante="global",
    null="no"
):
    """
    Construye dos secuencias de fases instantáneas f1 y f2 usando la transformada de Hilbert.

    Caso univariante:
        modo_univariante="global":
            Primero calcula theta = fase instantánea de X completa.
            Luego:
                f1 = theta[tau:]
                f2 = theta[:-tau]

        modo_univariante="segmentos":
            Calcula fases instantáneas por separado:
                f1 = fase instantánea de X[tau:]
                f2 = fase instantánea de X[:-tau]

    Caso bivariante:
        f1 = fase instantánea de X
        f2 = fase instantánea de Y

    Parámetros
    ----------
    seriex : array_like
        Serie temporal principal.

    seriey : array_like or None
        Segunda serie temporal. Si es None, se usa el caso univariante con retardo tau.

    tau : int
        Retardo temporal para el caso univariante y bivariante.

    delta : bool
        Si True, calcula la diferencia entre fases instantáneas.
        Si False, devuelve las fases instantáneas directamente.

    unwrap : bool
        Si False, devuelve fases envueltas en [-pi, pi].
        Si True, devuelve fases desenvueltas con np.unwrap.

        Para una caminata sobre el toro, usualmente conviene usar unwrap=False.

    modo_univariante : {"global", "segmentos"}
        Define cómo calcular la fase instantánea en el caso univariante.

        "global" es el modo recomendado para construir:
            (theta_i, theta_{i+tau})

        "segmentos" imita más literalmente la lógica de la función de Fourier,
        pero puede introducir diferencias por efectos de borde en Hilbert.

    Returns
    -------
    f1, f2 : np.ndarray
        Secuencias de fases instantáneas.
    """

    def fase_instantanea(x):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError("La serie debe ser un array unidimensional.")

        if len(x) < 3:
            raise ValueError("La serie debe tener al menos 3 puntos.")

        if not np.all(np.isfinite(x)):
            raise ValueError("La serie contiene NaN o infinitos.")

        if True:
            x = x - np.mean(x)

        z = hilbert(x)
        theta = np.angle(z)

        if False:
            theta = np.unwrap(theta)

        return theta
    
    seriex = np.asarray(seriex, dtype=np.float64)

    if seriey is None:

        if modo_univariante == "global":
            theta = fase_instantanea(seriex)
            if delta:
                theta = wrap_pi(np.diff(theta))
            if null == "shuffle":
                theta = np.random.permutation(theta)
            if tau > 0:
                f1 = theta[tau:]
                f2 = theta[:-tau]
            elif tau < 0:
                f1 = theta[:tau]
                f2 = theta[-tau:]
            else:
                f1, f2 = theta, theta
            return f1, f2

        elif modo_univariante == "segmentos":
            x1 = seriex[tau:]
            y1 = seriex[:-tau]

            f1 = fase_instantanea(x1)
            f2 = fase_instantanea(y1)
            if delta:
                f1 = wrap_pi(np.diff(f1))
                f2 = wrap_pi(np.diff(f2))

        else:
            raise ValueError("modo_univariante debe ser 'global' o 'segmentos'.")

    else:
        seriey = np.asarray(seriey, dtype=np.float64)

        if len(seriex) != len(seriey):
            raise ValueError(
                "En el caso bivariante, seriex y seriey deben tener la misma longitud."
            )
        if tau == 0:
            f1 = fase_instantanea(seriex)
            f2 = fase_instantanea(seriey)
            if delta:
                f1 = wrap_pi(np.diff(f1))
                f2 = wrap_pi(np.diff(f2))
        elif tau > 0:
            seriex = seriex[tau:]
            seriey = seriey[:-tau]
            f1 = fase_instantanea(seriex)
            f2 = fase_instantanea(seriey)
            if delta:
                f1 = wrap_pi(np.diff(f1))
                f2 = wrap_pi(np.diff(f2))
        elif tau < 0:
            seriex = seriex[:tau]
            seriey = seriey[-tau:]
            f1 = fase_instantanea(seriex)
            f2 = fase_instantanea(seriey)
            if delta:
                f1 = wrap_pi(np.diff(f1))
                f2 = wrap_pi(np.diff(f2))


    if null == "shuffle":
        indices = np.arange(len(f1))
        np.random.shuffle(indices)
        f1 = f1[indices]
        f2 = f2[indices]
        # f1 = np.random.permutation(f1)
        # f2 = np.random.permutation(f2)

    return f1, f2

def indice_H(seriex, seriey=None, tau=1,null="no", delta=False):
    if null == "shuffle2":
        tau_null = tau
        tau = 0
    # 1. Fases de Fourier
    f1, f2 = obtener_fases_instantaneas(seriex,
        seriey=seriey,
        tau=tau,
        delta=delta,
        modo_univariante="global",
        null=null)

    # 2. Puntos en el toro
    puntos = construir_puntos_toro(f1, f2)

    # 3. Vectores geodésicos
    vectores = construir_vectores_geodesicos(puntos)
    if null == "shuffle2":
        vectores = shuffle_vectores(vectores, tau_null)

    # 4. Ángulos entre vectores consecutivos
    angulos = calcular_angulos_entre_vectores(vectores)

    if len(angulos) == 0:
        H = np.nan 
    else:
        e = np.exp(1j * angulos)
        H = 1.0 - np.abs(np.mean(e))
    return H

def angulos_alpha_H(seriex, seriey, tau = 1,null="no",delta=False):
    if null == "shuffle2":
        tau_null = tau
        tau = 0
    f1, f2 = obtener_fases_instantaneas(seriex,
        seriey=seriey,
        tau=tau,
        delta=delta,
        modo_univariante="global",
        null=null)
    puntos = construir_puntos_toro(f1, f2)
    vectores = construir_vectores_geodesicos(puntos)
    if null == "shuffle2":
        vectores = shuffle_vectores(vectores, tau_null)
    corr_artifact = np.corrcoef(vectores[:-1, 0], vectores[1:, 1])[0, 1]
    print(f"Correlación entre componentes x de v_i y componentes y de v_(i+1): {corr_artifact:.3f}")

    angulos = calcular_angulos_entre_vectores(vectores)
    return angulos

@njit
def _S_eff_core_numba(theta, M, n_grid, sigma, usar_pesos):
    dospi = 2.0 * np.pi
    N = len(theta)

    c_re = np.zeros(M)
    c_im = np.zeros(M)

    # Coeficientes c_n = <exp(i n theta)>
    for i in range(N):
        th = theta[i]

        cos1 = np.cos(th)
        sin1 = np.sin(th)

        cosk = 1.0
        sink = 0.0

        for m in range(M):
            # Multiplicar por exp(i theta)
            nuevo_cosk = cosk * cos1 - sink * sin1
            nuevo_sink = sink * cos1 + cosk * sin1

            c_re[m] += nuevo_cosk
            c_im[m] += nuevo_sink

            cosk = nuevo_cosk
            sink = nuevo_sink

    for m in range(M):
        c_re[m] /= N
        c_im[m] /= N

        if usar_pesos:
            k = m + 1
            peso = np.exp(-(k * k) / (2.0 * sigma * sigma))
            c_re[m] *= peso
            c_im[m] *= peso

    # Reconstrucción de la densidad f(theta)
    f_theta = np.empty(n_grid)

    Z = 0.0
    dtheta = dospi / n_grid

    for j in range(n_grid):
        th = dospi * j / n_grid

        cos1 = np.cos(th)
        sin1 = np.sin(th)

        cosk = 1.0
        sink = 0.0

        suma = 0.0

        for m in range(M):
            nuevo_cosk = cosk * cos1 - sink * sin1
            nuevo_sink = sink * cos1 + cosk * sin1

            # Re[c_k exp(-i k theta)] = a_k cos(k theta) + b_k sin(k theta)
            suma += c_re[m] * nuevo_cosk + c_im[m] * nuevo_sink

            cosk = nuevo_cosk
            sink = nuevo_sink

        valor = 1.0 / dospi + suma / np.pi

        if valor < 0.0:
            valor = 0.0

        f_theta[j] = valor
        Z += valor

    Z *= dtheta

    if Z <= 0.0:
        return np.nan, np.nan, f_theta

    # Normalización
    for j in range(n_grid):
        f_theta[j] /= Z

    # Entropía diferencial
    h_alpha = 0.0

    for j in range(n_grid):
        if f_theta[j] > 0.0:
            h_alpha -= f_theta[j] * np.log(f_theta[j]) * dtheta

    S_eff = np.exp(h_alpha) / dospi

    if S_eff < 0.0:
        S_eff = 0.0

    if S_eff > 1.0:
        S_eff = 1.0

    return S_eff, h_alpha, f_theta

def S_eff_desde_angulos(
    angulos,
    M=None,
    n_grid=500,
    sigma=None,
    usar_pesos=True,
    return_details=False
):
    angulos = np.asarray(angulos, dtype=np.float64)

    if len(angulos) == 0:
        if return_details:
            return np.nan, np.nan, None, None, angulos
        return np.nan

    theta = np.mod(angulos, 2.0 * np.pi)
    theta = np.ascontiguousarray(theta)

    N = len(theta)

    if M is None:
        M = int(np.sqrt(N))

    M = max(1, int(M))

    if sigma is None:
        sigma = M / 3.0

    sigma = float(sigma)

    S_eff, h_alpha, f_theta = _S_eff_core_numba(
        theta,
        M,
        int(n_grid),
        sigma,
        bool(usar_pesos)
    )

    if return_details:
        theta_grid = np.linspace(0, 2.0 * np.pi, n_grid, endpoint=False)
        return S_eff, h_alpha, theta_grid, f_theta, angulos

    return S_eff

def indice_S_eff_fast(
    seriex,
    seriey=None,
    tau=1,
    null="no",
    delta=False,
    M=None,
    n_grid=500,
    sigma=None,
    usar_pesos=True,
    return_details=False,
    modo_univariante="global"
):
    if null == "shuffle2":
        tau_null = tau
        tau = 0
    f1, f2 = obtener_fases_instantaneas(
        seriex,
        seriey=seriey,
        tau=tau,
        delta=delta,
        modo_univariante=modo_univariante,
        null=null
    )

    puntos = construir_puntos_toro(f1, f2)
    vectores = construir_vectores_geodesicos(puntos)
    if null == "shuffle2":
        vectores = shuffle_vectores(vectores, tau_null)
    angulos = calcular_angulos_entre_vectores(vectores)

    return S_eff_desde_angulos(
        angulos,
        M=M,
        n_grid=n_grid,
        sigma=sigma,
        usar_pesos=usar_pesos,
        return_details=return_details
    )