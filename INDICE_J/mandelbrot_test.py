"""
Replica del panel J del conjunto de Mandelbrot
Aguilar-Hernandez et al., Chaos, Solitons & Fractals 198 (2025), 116575

Implementacion base:
    z_{n+1} = z_n**2 + c
    z_0 = 0
    1200 iteraciones
    serie analizada = |z_n|
    tau = 1
    retardo NO circular:
        x = s[tau:]
        y = s[:-tau]
    600 fases de Fourier:
        se elimina DC y se conserva Nyquist
    vectores con el factor 2, en la forma confirmada por los autores:
        v_i = P_{i+1}' - 2 P_i
    donde P_{i+1}' es la imagen de P_{i+1} que produce
    el paso corto desde P_i sobre el toro.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ============================================================
# 0. PARAMETROS
# ============================================================

# --- Especificados / inferidos directamente del articulo ---
N_ITER = 1200
TAU = 1

# --- NO aparecen inequívocamente especificados en el texto accesible ---
# Estos son un encuadre estándar que contiene el conjunto completo.
# Si obtenemos de los autores los límites exactos de Fig. 3,
# solamente hay que cambiarlos aquí.
RE_MIN = -2.0
RE_MAX = 0.5
IM_MIN = -1.25
IM_MAX = 1.25

# Tampoco tengo confirmada la resolución exacta de Fig. 3.
# 600x600 es suficiente para verificar la réplica.
# Después podemos aumentar a 1000x1000 o al tamaño original.
NX = 600
NY = 600

# Número de procesos.
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Para representación logarítmica.
J_FLOOR = 1e-14


# ============================================================
# 1. GEOMETRIA ANGULAR
# ============================================================

def wrap_pi(a):
    """
    Envuelve un ángulo al intervalo [-pi, pi).
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# 2. ORBITA DE MANDELBROT
# ============================================================

def orbita_mandelbrot_abs(c, n_iter=N_ITER):
    """
    Genera

        z_0, z_1, ..., z_n_iter

    con

        z_{n+1} = z_n^2 + c

    y devuelve

        |z_0|, |z_1|, ..., |z_n_iter|.

    Por tanto, para n_iter=1200 obtenemos 1201 valores.
    Esto permite construir, para tau=1,

        x = s[1:]   -> 1200 valores
        y = s[:-1]  -> 1200 valores.

    Para puntos fuera del conjunto, la iteración eventualmente
    rebasa la representación float64. En ese caso conservamos
    solamente el prefijo finito de la órbita.

    ESTE tratamiento del overflow es uno de los detalles que
    convendría confirmar directamente con los autores.
    """

    c = np.complex128(c)
    z = np.complex128(0.0 + 0.0j)

    s = np.empty(n_iter + 1, dtype=np.float64)
    s[0] = 0.0

    with np.errstate(over="ignore", invalid="ignore"):

        for n in range(1, n_iter + 1):

            z = z * z + c

            # Si ya hemos abandonado la aritmética finita,
            # nos quedamos con el prefijo que sí es representable.
            if not (np.isfinite(z.real) and np.isfinite(z.imag)):
                return s[:n]

            a = np.abs(z)

            if not np.isfinite(a):
                return s[:n]

            s[n] = a

    return s


# ============================================================
# 3. SERIES RETARDADAS: OPCION A
# ============================================================

def construir_series_retardadas(s, tau=TAU):
    """
    Caso confirmado por los autores:

        x = s[tau:]
        y = s[:-tau]

    NO se hace un desplazamiento circular.
    """

    if tau <= 0:
        raise ValueError("tau debe ser positivo.")

    if len(s) <= tau:
        raise ValueError("Serie demasiado corta para el retardo.")

    # x = s[tau:]
    # y = s[:-tau]
    x = s
    y = np.roll(s, -tau)  # Desplazamiento circular

    return x, y


# ============================================================
# 4. FASES DE FOURIER
# ============================================================

def fases_fourier(x):
    """
    Fases de Fourier de una serie real.

    Para N = 1200:

        rfft -> 601 coeficientes:
                k = 0, 1, ..., 600

        quitamos k = 0 (DC)

        quedan exactamente 600 fases:
                k = 1, ..., 600

    Por tanto SE CONSERVA la frecuencia de Nyquist.

    np.angle devuelve las fases en [-pi, pi].
    """

    F = np.fft.rfft(x)

    # Quitar solamente DC.
    # Importante: NO quitamos Nyquist.
    phi = np.angle(F[1:])

    return phi


# ============================================================
# 5. PUNTOS EN EL TORO
# ============================================================

def construir_puntos(phi1, phi2):
    """
    P_k = (phi1_k, phi2_k)
    """

    n = min(len(phi1), len(phi2))

    if n < 4:
        return np.empty((0, 2), dtype=np.float64)

    return np.column_stack((phi1[:n], phi2[:n]))


# ============================================================
# 6. VECTORES CON FACTOR 2
# ============================================================

def construir_vectores_factor2(P):
    """
    Implementacion del factor 2.

    Primero elegimos la imagen de P_{i+1} que produce
    el desplazamiento angular corto desde P_i:

        Delta_i = wrap_pi(P_{i+1} - P_i)

    Esa imagen en el covering space es

        P'_{i+1} = P_i + Delta_i.

    Luego aplicamos la definicion confirmada:

        v_i = P'_{i+1} - 2 P_i

    y por tanto

        v_i = Delta_i - P_i.

    Es exactamente la estructura:

        wrap_pi(P_{i+1} - P_i) - P_i
    """

    if len(P) < 2:
        return np.empty((0, 2), dtype=np.float64)

    delta = wrap_pi(P[1:] - P[:-1])

    V = delta - P[:-1]

    return delta


# ============================================================
# 7. ANGULOS ALFA
# ============================================================

def calcular_alfas(V):
    """
    Angulo orientado entre v_i y v_{i+1}:

        alpha_i = atan2(v_i x v_{i+1},
                        v_i . v_{i+1})

    devuelto en [0, 2*pi).
    """

    if len(V) < 2:
        return np.empty(0, dtype=np.float64)

    v1 = V[:-1]
    v2 = V[1:]

    norm1_sq = np.sum(v1 * v1, axis=1)
    norm2_sq = np.sum(v2 * v2, axis=1)

    validos = (norm1_sq > 0.0) & (norm2_sq > 0.0)

    if not np.any(validos):
        return np.empty(0, dtype=np.float64)

    v1 = v1[validos]
    v2 = v2[validos]

    dot = (
        v1[:, 0] * v2[:, 0]
        + v1[:, 1] * v2[:, 1]
    )

    cross = (
        v1[:, 0] * v2[:, 1]
        - v1[:, 1] * v2[:, 0]
    )

    alpha = np.arctan2(cross, dot)

    alpha = np.mod(alpha, 2.0 * np.pi)

    return alpha


# ============================================================
# 8. INDICE J
# ============================================================

def J_desde_serie(s, tau=TAU):
    """
    Calcula J para una serie real s.
    """

    if len(s) <= tau + 5:
        return np.nan

    # --------------------------------------------------------
    # Reescalamiento NUMERICO
    # --------------------------------------------------------
    #
    # Multiplicar una serie por una constante real positiva
    # no cambia sus fases de Fourier.
    #
    # Esto evita que valores enormes causen overflow dentro
    # de la propia FFT.
    # --------------------------------------------------------

    escala = np.max(np.abs(s))

    if escala == 0.0:
        # c = 0 produce la órbita idénticamente nula.
        return 0.0

    s = s / escala

    # Retardo truncado.
    x, y = construir_series_retardadas(s, tau=tau)

    # Fourier.
    phi1 = fases_fourier(x)
    phi2 = fases_fourier(y)

    # Toro.
    P = construir_puntos(phi1, phi2)

    if len(P) < 4:
        return np.nan

    # Factor 2.
    V = construir_vectores_factor2(P)

    # Angulos.
    alpha = calcular_alfas(V)

    if len(alpha) == 0:
        return np.nan

    # Índice J.
    J = 1.0 - np.abs(np.mean(np.exp(1j * alpha)))

    return float(J)


def J_mandelbrot(c):
    """
    Calcula J asociado a un punto c del plano complejo.
    """

    s = orbita_mandelbrot_abs(c, n_iter=N_ITER)

    return J_desde_serie(s, tau=TAU)


# ============================================================
# 9. FILA DEL PLANO COMPLEJO
# ============================================================

RE_VALUES = np.linspace(RE_MIN, RE_MAX, NX)
IM_VALUES = np.linspace(IM_MIN, IM_MAX, NY)


def calcular_fila(j):
    """
    Calcula una fila completa Im(c) = constante.

    Debe estar definida a nivel global para que funcione
    correctamente con multiprocessing en Windows.
    """

    im = IM_VALUES[j]

    fila = np.empty(NX, dtype=np.float64)

    for i, re in enumerate(RE_VALUES):

        c = re + 1j * im

        fila[i] = J_mandelbrot(c)

    return j, fila


# ============================================================
# 10. CALCULO DEL MAPA
# ============================================================

def calcular_mapa_J():

    Jmap = np.empty((NY, NX), dtype=np.float64)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(calcular_fila, j)
            for j in range(NY)
        ]

        for future in tqdm(
            as_completed(futures),
            total=NY,
            desc="Calculando Mandelbrot-J"
        ):

            j, fila = future.result()

            Jmap[j, :] = fila

    return Jmap


# ============================================================
# 11. GRAFICA
# ============================================================

def graficar(Jmap):

    # El artículo usa escala logarítmica para el panel J.
    Z = np.array(Jmap, copy=True)

    mask = np.isfinite(Z)

    Z[mask] = np.maximum(Z[mask], J_FLOOR)

    Z = np.log10(Z)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[
            RE_MIN,
            RE_MAX,
            IM_MIN,
            IM_MAX
        ],
        aspect="equal",
        interpolation="nearest",
        cmap="Reds"
    )

    ax.set_xlabel(r"$\mathrm{Re}(c)$")
    ax.set_ylabel(r"$\mathrm{Im}(c)$")

    ax.set_title(
        r"Mandelbrot mediante índice $J$"
        + "\n"
        + rf"$N={N_ITER}$, $\tau={TAU}$"
    )

    cbar = fig.colorbar(im, ax=ax)

    cbar.set_label(r"$\log_{10} J$")

    fig.tight_layout()

    return fig, ax


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Comprobación importante:
    # una serie de longitud 1200 debe producir 600 fases
    # al quitar DC y conservar Nyquist.
    # --------------------------------------------------------

    prueba = np.zeros(N_ITER)

    n_fases = len(fases_fourier(prueba))

    print(f"Número de datos por señal retardada: {N_ITER}")
    print(f"Número de fases de Fourier utilizadas: {n_fases}")

    assert n_fases == 600

    print()
    print(f"Malla: {NX} x {NY}")
    print(f"Procesos: {MAX_WORKERS}")
    print()

    # Calcular.
    Jmap = calcular_mapa_J()

    # Guardar datos crudos ANTES de cualquier transformación gráfica.
    np.savez_compressed(
        "mandelbrot_J_replica.npz",
        J=Jmap,
        re=RE_VALUES,
        im=IM_VALUES,
        N_ITER=N_ITER,
        TAU=TAU
    )

    # Graficar.
    fig, ax = graficar(Jmap)

    fig.savefig(
        "mandelbrot_J_replica.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()