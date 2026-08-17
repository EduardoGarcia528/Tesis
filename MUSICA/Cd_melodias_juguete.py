# ============================================================
# Gamma/C_d y mPE para una melodía:
#   1. Alturas MIDI
#   2. Intervalos melódicos
#
# Modelo nulo:
#   - Alturas: shuffle de las alturas.
#   - Intervalos: shuffle de la secuencia de intervalos.
#
# Salidas:
#   - Figuras de C_d y pendiente de log(C_d)
#   - Figuras de mPE, Z-score y diferencia null - observado
#   - CSV con todos los resultados
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

import mi_libreria as ml

import numpy as np


def generar_melodia_patron(
    n_repeticiones,
    patron=(1, 1, 2, 0, 4, 4, 3),
    midi_min=36,
    midi_max=96,
    intervalos_posibles=(1, 2, 3, 4, 5, 7, 12),
    cambiar_cada_repeticion=True,
    seed=None
):
    """
    Genera una secuencia de alturas MIDI que respeta exactamente
    un patrón ordinal con empates.

    Para el patrón [1, 1, 2, 0, 4, 4, 3]:

        nivel 0 < nivel 1 < nivel 2 < nivel 3 < nivel 4

    y cada bloque tiene la forma:

        [nivel_1, nivel_1, nivel_2,
         nivel_0,
         nivel_4, nivel_4, nivel_3]

    Los intervalos entre niveles pueden cambiar en cada repetición.

    Parámetros
    ----------
    n_repeticiones : int
        Número de bloques de siete notas.

    patron : secuencia de int
        Patrón ordinal. Los valores indican los niveles relativos.

    midi_min, midi_max : int
        Registro MIDI permitido.

    intervalos_posibles : secuencia de int
        Intervalos positivos, en semitonos, que pueden separar
        niveles consecutivos. Incluir 12 permite saltos de octava.

    cambiar_cada_repeticion : bool
        Si es True, cada bloque utiliza alturas e intervalos nuevos.
        Si es False, se genera un solo bloque y se repite literalmente.

    seed : int o None
        Semilla para reproducibilidad.

    Retorna
    -------
    melodia : np.ndarray
        Serie completa de alturas MIDI.

    bloques : np.ndarray
        Matriz de tamaño (n_repeticiones, len(patron)).
    """
    rng = np.random.default_rng(seed)

    patron = np.asarray(patron, dtype=int)
    intervalos_posibles = np.asarray(intervalos_posibles, dtype=int)

    if n_repeticiones < 1:
        raise ValueError("n_repeticiones debe ser al menos 1.")

    if midi_min < 0 or midi_max > 127 or midi_min >= midi_max:
        raise ValueError("El rango MIDI debe satisfacer 0 <= midi_min < midi_max <= 127.")

    if patron.min() != 0:
        raise ValueError("El nivel mínimo del patrón debe ser 0.")

    niveles_unicos = np.unique(patron)

    if not np.array_equal(
        niveles_unicos,
        np.arange(niveles_unicos.size)
    ):
        raise ValueError(
            "Los niveles del patrón deben ser enteros consecutivos desde 0."
        )

    if np.any(intervalos_posibles <= 0):
        raise ValueError("Todos los intervalos deben ser positivos.")

    n_niveles = niveles_unicos.size

    def generar_bloque():
        # Se intenta hasta encontrar una colección de niveles
        # que quepa dentro del registro MIDI permitido.
        for _ in range(10_000):
            gaps = rng.choice(
                intervalos_posibles,
                size=n_niveles - 1,
                replace=True
            )

            span = int(gaps.sum())

            if span > midi_max - midi_min:
                continue

            nota_inferior = rng.integers(
                midi_min,
                midi_max - span + 1
            )

            niveles = np.concatenate((
                [nota_inferior],
                nota_inferior + np.cumsum(gaps)
            ))

            return niveles[patron]

        raise RuntimeError(
            "No fue posible generar el bloque dentro del rango MIDI. "
            "Amplía midi_min/midi_max o reduce los intervalos posibles."
        )

    if cambiar_cada_repeticion:
        bloques = np.array([
            generar_bloque()
            for _ in range(n_repeticiones)
        ])
    else:
        bloque = generar_bloque()
        bloques = np.tile(bloque, (n_repeticiones, 1))

    melodia = bloques.ravel()

    return melodia, bloques
# ============================================================
# Configuración
# ============================================================

MAX_GAMMA = 9
MU = 2

# Con max_gamma=9 se espera:
# C = [C_0, C_1, ..., C_10], con C_0 = 1.
FIT_D_MIN = 1
FIT_D_MAX = None

# Para N ~ 300, m > 7 puede producir estimaciones muy dispersas
# porque el número de patrones posibles crece como m!.
M_VALUES = np.arange(2, 8)  # m = 2, 3, ..., 7
TAU = 1

N_SHUFFLES = 1000
SEED = 20260729

OUTPUT_DIR = Path("resultados_gamma_mpe_melodia")
SHOW_PLOTS = True


# ============================================================
# Melodía
# ============================================================

midi = np.array(
    [
        64, 67, 66, 62, 59, 66, 64, 67, 71, 69, 66, 62, 57, 59, 64,
        67, 66, 62, 59, 66, 64, 67, 71, 69, 66, 62, 57, 63, 66, 66,
        66, 64, 71, 71, 71, 69, 69, 67, 67, 66, 66, 62, 62, 64, 52,
        54, 55, 60, 64, 67, 54, 55, 59, 64, 67, 71, 71, 69, 67, 69,
        67, 66, 67, 62, 60, 62, 52, 53, 58, 62, 64, 65, 70, 74, 54,
        55, 60, 64, 67, 72, 77, 75, 74, 77, 75, 74, 75, 74, 72, 62,
        63, 68, 72, 64, 67, 72, 76, 77, 80, 79, 75, 72, 79, 77, 80,
        84, 82, 79, 75, 70, 72, 84, 84, 84, 82, 82, 80, 80, 79, 79,
        77, 77, 76, 76, 72, 74, 62, 65, 69, 70, 74, 77, 82, 67, 69,
        70, 74, 79, 82, 86, 91, 86, 84, 82, 84, 82, 81, 82, 81, 79,
        81, 79, 77, 79, 74, 75, 79, 77, 75, 74, 72, 70, 72, 67, 65,
        63, 65, 63, 62, 63, 62, 60, 79, 77, 75, 77, 75, 74, 75, 67,
        72, 74, 72, 70, 72, 68, 67, 65, 63, 62, 60, 58, 56, 55, 65,
        72, 77, 84, 75, 65, 68, 67, 63, 60, 67, 65, 56, 60, 58, 55,
        51, 46, 75, 70, 72, 55, 53, 58, 73, 71, 70, 68, 67, 66, 67,
        58, 58, 57, 62, 67, 70, 69, 74, 74, 74, 73, 82, 82, 82, 81,
        86, 85, 79, 82, 81, 77, 72, 75, 74, 70, 68, 71, 70, 66, 68,
        68, 71, 70, 66, 68, 71, 70, 66, 68, 71, 70, 66, 68, 71, 70,
        66, 63, 70, 68
    ],
    dtype=int
)

midi1 = np.array(
    [69,69,69,65,67,67,67,64,76,76,76,72,70,70,70,69,84,84,84,81,76,76,76,71,70,70,70,69,74,74,74,71,76,76,74,72,76,72,72,74,76,
     76,72,72,74,76,76,76,74,72,69,76,77,77,77,74,82,82,82,79,71,71,71,68,70,70,70,69,64,64,64,69,82,82,82,79,71,71,71,68,70,70,70,69,64,64,64,69,72,72,72,73,73,73,76,76,62,62,62,65,
     65,64,64,67,67,66,66,69,69,67,67,70,70,69,69,72,72,70,70,72,74,73,73,76,74,77,77,77,74,69,69,69,65,62,62,62,65,62,62,62,65,62,62,62,61,76,76,76,73,69,67,67,
     76,73,69,67,64,61,62,62,62,77,77,77,74,83,83,83,81,77,77,77,74,71,71,71,71,72,79,79,79,72,74,67,72,77,76,77,79,74,74,72,72,77,76,77,79,74,74,72,72,77,76,77,79,74,74,72,
     67,69,70,69,67,69,67,65,70,72,74,72,70,72,70,69,72,74,76,74,72,74,75,74,72,74,75,74,72,74,75,74,72,74,75,74,75,78,79,86,84,82,82,81,79,77,77,76,74,76,79,77,72,69,76,74,70,67,
     74,72,69,65,60,83,84,83,84,83,84,83,84,86,84,82,82,81,79,77,77,76,74,76,79,77,72,69,76,74,72,65,74,72,69,65,60,72,84,77,72,79,79,79,76,72,72,72,79,79,79,81,72,79,79,79,76,72,72,72,67,72,72,72,72,72,72,72,72,76,76,76,77,],
    dtype=int
)

midi1, bloques = generar_melodia_patron(
    n_repeticiones=350,
    # patron = (3,0,3,1,4,5,2,1,0,3),
    patron = (0),
    midi_min=36,       # C2
    midi_max=96,       # C7
    intervalos_posibles=(1, 2, 3, 4, 5, 7,8,9, 12),
    seed=42
)
plt.plot(midi,'.-')
plt.show()
intervalos = np.diff(midi)


# ============================================================
# Funciones auxiliares
# ============================================================

def calcular_C(arr):
    """
    Calcula C_d mediante gamma_index_rank_ties.

    Devuelve
    --------
    d : ndarray
        Valores d = 0, 1, ..., len(C)-1.
    C : ndarray
        Integrales de correlación.
    g : ndarray
        Índices gamma devueltos por la biblioteca.
    """
    C, g = ml.gamma_index_rank_ties(
        arr,
        max_gamma=MAX_GAMMA,
        mu=MU
    )

    C = np.asarray(C, dtype=float)
    g = np.asarray(g, dtype=float)
    d = np.arange(len(C), dtype=int)

    return d, C, g


def ajustar_log_C(d, C, d_min=0, d_max=None):
    """
    Ajusta:

        log(C_d) = intercepto + pendiente * d

    Solo usa valores finitos y estrictamente positivos de C_d.
    """

    d = np.asarray(d, dtype=float)
    C = np.asarray(C, dtype=float)

    mascara = np.isfinite(d) & np.isfinite(C) & (C > 0)

    if d_min is not None:
        mascara &= d >= d_min

    if d_max is not None:
        mascara &= d <= d_max

    d_fit = d[mascara]
    C_fit = C[mascara]

    if len(d_fit) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "stderr": np.nan,
            "n_points": len(d_fit),
            "d_fit": d_fit,
            "logC_fit": np.log(C_fit)
        }

    resultado = linregress(d_fit, np.log(C_fit))

    return {
        "slope": resultado.slope,
        "intercept": resultado.intercept,
        "r2": resultado.rvalue**2,
        "stderr": resultado.stderr,
        "n_points": len(d_fit),
        "d_fit": d_fit,
        "logC_fit": np.log(C_fit)
    }


def calcular_mpe_para_m(arr, m_values):
    """
    Calcula mPE no normalizada para todos los valores de m.
    """
    resultados = np.full(len(m_values), np.nan, dtype=float)

    for i, m in enumerate(m_values):
        try:
            resultados[i] = ml.modified_permutation_entropy(
                arr,
                int(m),
                tau=TAU,
                norm=False
            )
        except Exception as error:
            print(
                f"Advertencia: no se pudo calcular mPE "
                f"para m={m}: {error}"
            )

    return resultados


def media_std_cuantiles(x, axis=0):
    """
    Estadísticos ignorando NaN.
    """
    return {
        "mean": np.nanmean(x, axis=axis),
        "std": np.nanstd(x, axis=axis, ddof=1),
        "q025": np.nanquantile(x, 0.025, axis=axis),
        "median": np.nanquantile(x, 0.50, axis=axis),
        "q975": np.nanquantile(x, 0.975, axis=axis)
    }


def z_score(observado, media_null, std_null):
    """
    Z = (observado - media_null) / std_null.
    """
    observado = np.asarray(observado, dtype=float)
    media_null = np.asarray(media_null, dtype=float)
    std_null = np.asarray(std_null, dtype=float)

    z = np.full_like(observado, np.nan, dtype=float)

    mascara = (
        np.isfinite(observado)
        & np.isfinite(media_null)
        & np.isfinite(std_null)
        & (std_null > 0)
    )

    z[mascara] = (
        observado[mascara] - media_null[mascara]
    ) / std_null[mascara]

    return z


def p_empirico_dos_colas(observado, null):
    """
    Contraste empírico bilateral respecto al centro de la distribución nula.

    Se compara la distancia del observado a la media nula con las distancias
    obtenidas dentro de la propia distribución nula.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]

    if len(null) == 0 or not np.isfinite(observado):
        return np.nan

    centro = np.mean(null)
    distancia_obs = abs(observado - centro)
    distancias_null = np.abs(null - centro)

    return (
        1 + np.sum(distancias_null >= distancia_obs)
    ) / (len(null) + 1)


# ============================================================
# Análisis principal de una representación
# ============================================================

def analizar_representacion(nombre, arr, rng):
    """
    Calcula C_d, pendiente de decaimiento y mPE para:
        - serie observada
        - N_SHUFFLES permutaciones
    """
    midi = np.asarray(arr)

    if nombre == "alturas_MIDI":
        arr = np.asarray(arr)
    elif nombre == "intervalos":
        arr = np.diff(np.asarray(arr))
    else:
        raise ValueError(
            f"Nombre de representación desconocido: {nombre}"
        )

    print("\n" + "=" * 70)
    print(f"Analizando: {nombre}")
    print(f"Longitud: {len(arr)}")
    print("=" * 70)

    # --------------------------------------------------------
    # Valores observados
    # --------------------------------------------------------

    d, C_obs, gamma_obs = calcular_C(arr)

    ajuste_obs = ajustar_log_C(
        d,
        C_obs,
        d_min=FIT_D_MIN,
        d_max=FIT_D_MAX
    )

    mpe_obs = calcular_mpe_para_m(arr, M_VALUES)

    # --------------------------------------------------------
    # Reservar matrices para el modelo nulo
    # --------------------------------------------------------

    C_null = np.full(
        (N_SHUFFLES, len(C_obs)),
        np.nan,
        dtype=float
    )

    slope_null = np.full(N_SHUFFLES, np.nan, dtype=float)
    intercept_null = np.full(N_SHUFFLES, np.nan, dtype=float)
    r2_null = np.full(N_SHUFFLES, np.nan, dtype=float)

    mpe_null = np.full(
        (N_SHUFFLES, len(M_VALUES)),
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # Permutaciones
    # --------------------------------------------------------
    iaaft = ml.iaaft(midi,N_SHUFFLES)
    for i in range(N_SHUFFLES):
        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"\rPermutaciones: {i + 1}/{N_SHUFFLES}",
                end="",
                flush=True
            )

        if nombre == "alturas_MIDI":
            arr_shuffle = iaaft[i]
            # arr_shuffle = rng.permutation(midi)
        elif nombre == "intervalos":
            arr_shuffle = np.diff(iaaft[i])
            # arr_shuffle = np.diff(rng.permutation(midi))
            # arr_shuffle = rng.permutation(np.diff(midi))
        else:
            raise ValueError(
                f"Nombre de representación desconocido: {nombre}"
            )

        # C_d
        d_shuffle, C_shuffle, _ = calcular_C(arr_shuffle)

        if len(C_shuffle) != len(C_obs):
            raise ValueError(
                "La longitud de C cambió entre el observado y el shuffle."
            )

        C_null[i, :] = C_shuffle

        ajuste_shuffle = ajustar_log_C(
            d_shuffle,
            C_shuffle,
            d_min=FIT_D_MIN,
            d_max=FIT_D_MAX
        )

        slope_null[i] = ajuste_shuffle["slope"]
        intercept_null[i] = ajuste_shuffle["intercept"]
        r2_null[i] = ajuste_shuffle["r2"]

        # mPE
        mpe_null[i, :] = calcular_mpe_para_m(
            arr_shuffle,
            M_VALUES
        )
    print(np.sort(np.unique(midi)))
    print(np.sort(np.unique(intervalos)))
    print()

    # --------------------------------------------------------
    # Estadísticos nulos
    # --------------------------------------------------------

    C_stats = media_std_cuantiles(C_null, axis=0)
    mpe_stats = media_std_cuantiles(mpe_null, axis=0)

    mpe_z = z_score(
        mpe_obs,
        mpe_stats["mean"],
        mpe_stats["std"]
    )

    # La diferencia solicitada:
    # diferencia = media del nulo - observado
    mpe_null_minus_obs = mpe_stats["mean"] - mpe_obs

    slope_finitas = slope_null[np.isfinite(slope_null)]

    slope_media = np.mean(slope_finitas)
    slope_std = np.std(slope_finitas, ddof=1)
    slope_q025 = np.quantile(slope_finitas, 0.025)
    slope_mediana = np.quantile(slope_finitas, 0.50)
    slope_q975 = np.quantile(slope_finitas, 0.975)

    slope_z = (
        ajuste_obs["slope"] - slope_media
    ) / slope_std

    slope_p = p_empirico_dos_colas(
        ajuste_obs["slope"],
        slope_finitas
    )

    # p unilateral:
    # pendiente más negativa = decaimiento más rápido
    p_decaimiento_mas_rapido = (
        1 + np.sum(slope_finitas <= ajuste_obs["slope"])
    ) / (len(slope_finitas) + 1)

    # pendiente menos negativa = decaimiento más lento
    p_decaimiento_mas_lento = (
        1 + np.sum(slope_finitas >= ajuste_obs["slope"])
    ) / (len(slope_finitas) + 1)

    # --------------------------------------------------------
    # Mostrar resumen
    # --------------------------------------------------------

    print(f"Representación: {nombre}")
    print(
        f"Pendiente observada       = "
        f"{ajuste_obs['slope']:.8f}"
    )
    print(
        f"Constante lambda = -slope = "
        f"{-ajuste_obs['slope']:.8f}"
    )
    print(
        f"R² observado              = "
        f"{ajuste_obs['r2']:.6f}"
    )
    print(
        f"Pendiente nula media      = "
        f"{slope_media:.8f}"
    )
    print(
        f"Desviación pendiente nula = "
        f"{slope_std:.8f}"
    )
    print(
        f"Z-score de la pendiente   = "
        f"{slope_z:.6f}"
    )
    print(
        f"p empírico bilateral      = "
        f"{slope_p:.6f}"
    )
    print(
        f"p decaimiento más rápido  = "
        f"{p_decaimiento_mas_rapido:.6f}"
    )
    print(
        f"p decaimiento más lento   = "
        f"{p_decaimiento_mas_lento:.6f}"
    )

    return {
        "nombre": nombre,
        "arr": arr,
        "d": d,
        "C_obs": C_obs,
        "gamma_obs": gamma_obs,
        "ajuste_obs": ajuste_obs,
        "C_null": C_null,
        "C_stats": C_stats,
        "slope_null": slope_null,
        "intercept_null": intercept_null,
        "r2_null": r2_null,
        "slope_media": slope_media,
        "slope_std": slope_std,
        "slope_q025": slope_q025,
        "slope_mediana": slope_mediana,
        "slope_q975": slope_q975,
        "slope_z": slope_z,
        "slope_p": slope_p,
        "p_decaimiento_mas_rapido": p_decaimiento_mas_rapido,
        "p_decaimiento_mas_lento": p_decaimiento_mas_lento,
        "mpe_obs": mpe_obs,
        "mpe_null": mpe_null,
        "mpe_stats": mpe_stats,
        "mpe_z": mpe_z,
        "mpe_null_minus_obs": mpe_null_minus_obs
    }


# ============================================================
# Gráficas de C_d
# ============================================================

def graficar_C_y_pendiente(resultado, output_dir):
    nombre = resultado["nombre"]
    d = resultado["d"]
    C_obs = resultado["C_obs"]
    C_stats = resultado["C_stats"]
    ajuste_obs = resultado["ajuste_obs"]
    slope_null = resultado["slope_null"]

    # Ajuste de la curva media nula.
    ajuste_C_null_media = ajustar_log_C(
        d,
        C_stats["mean"],
        d_min=FIT_D_MIN,
        d_max=FIT_D_MAX
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5)
    )

    # --------------------------------------------------------
    # Panel 1: C_d vs d
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        d,
        C_obs,
        marker="o",
        linewidth=2,
        label="Observado"
    )

    ax.plot(
        d,
        C_stats["mean"],
        marker="s",
        linewidth=1.5,
        label="Media shuffle"
    )

    ax.fill_between(
        d,
        C_stats["q025"],
        C_stats["q975"],
        alpha=0.25,
        label="IC nulo 95%"
    )

    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$C_d$")
    ax.set_title(f"{nombre}: $C_d$ vs $d$")
    ax.grid(alpha=0.3)
    ax.legend()

    # --------------------------------------------------------
    # Panel 2: log(C_d) y ajustes lineales
    # --------------------------------------------------------

    ax = axes[1]

    mascara_obs = (
        np.isfinite(C_obs)
        & (C_obs > 0)
    )

    mascara_null = (
        np.isfinite(C_stats["mean"])
        & (C_stats["mean"] > 0)
    )

    ax.scatter(
        d[mascara_obs],
        np.log(C_obs[mascara_obs]),
        label="Observado"
    )

    ax.scatter(
        d[mascara_null],
        np.log(C_stats["mean"][mascara_null]),
        marker="s",
        label="Media shuffle"
    )

    if np.isfinite(ajuste_obs["slope"]):
        d_linea = np.linspace(
            np.min(ajuste_obs["d_fit"]),
            np.max(ajuste_obs["d_fit"]),
            200
        )

        logC_linea = (
            ajuste_obs["intercept"]
            + ajuste_obs["slope"] * d_linea
        )

        ax.plot(
            d_linea,
            logC_linea,
            linewidth=2,
            label=(
                "Ajuste observado\n"
                f"pendiente={ajuste_obs['slope']:.4f}, "
                f"$R^2$={ajuste_obs['r2']:.3f}"
            )
        )

    if np.isfinite(ajuste_C_null_media["slope"]):
        d_linea_null = np.linspace(
            np.min(ajuste_C_null_media["d_fit"]),
            np.max(ajuste_C_null_media["d_fit"]),
            200
        )

        logC_linea_null = (
            ajuste_C_null_media["intercept"]
            + ajuste_C_null_media["slope"] * d_linea_null
        )

        ax.plot(
            d_linea_null,
            logC_linea_null,
            linestyle="--",
            linewidth=2,
            label=(
                "Ajuste media shuffle\n"
                f"pendiente={ajuste_C_null_media['slope']:.4f}"
            )
        )

    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$\log(C_d)$")
    ax.set_title("Ajuste lineal en escala logarítmica")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # Panel 3: distribución nula de pendientes
    # --------------------------------------------------------

    ax = axes[2]

    slope_finitas = slope_null[np.isfinite(slope_null)]

    ax.hist(
        slope_finitas,
        bins=35,
        alpha=0.75,
        label="Pendientes shuffle"
    )

    ax.axvline(
        resultado["slope_media"],
        linestyle="--",
        linewidth=2,
        label=(
            f"Media nula = "
            f"{resultado['slope_media']:.4f}"
        )
    )

    ax.axvline(
        ajuste_obs["slope"],
        linewidth=2.5,
        label=(
            f"Observado = "
            f"{ajuste_obs['slope']:.4f}"
        )
    )

    ax.set_xlabel(r"Pendiente de $\log(C_d)$")
    ax.set_ylabel("Frecuencia")
    ax.set_title(
        "Contraste de la pendiente\n"
        f"Z={resultado['slope_z']:.2f}, "
        f"p bilateral={resultado['slope_p']:.4f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Decaimiento de las integrales de correlación: {nombre}",
        fontsize=14
    )

    fig.tight_layout()

    ruta = output_dir / f"Cd_y_pendiente_{nombre}.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Gráficas de mPE
# ============================================================

def graficar_mpe(resultado, output_dir):
    nombre = resultado["nombre"]

    mpe_obs = resultado["mpe_obs"]
    mpe_stats = resultado["mpe_stats"]
    mpe_z = resultado["mpe_z"]
    diferencia = resultado["mpe_null_minus_obs"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5)
    )

    # --------------------------------------------------------
    # Panel 1: mPE observada y nula
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        M_VALUES,
        mpe_obs,
        marker="o",
        linewidth=2,
        label="Observado"
    )

    ax.plot(
        M_VALUES,
        mpe_stats["mean"],
        marker="s",
        linewidth=1.5,
        label="Media shuffle"
    )

    ax.fill_between(
        M_VALUES,
        mpe_stats["q025"],
        mpe_stats["q975"],
        alpha=0.25,
        label="IC nulo 95%"
    )

    ax.set_xlabel(r"$m$")
    ax.set_ylabel("mPE no normalizada")
    ax.set_title(f"{nombre}: mPE vs $m$")
    ax.set_xticks(M_VALUES)
    ax.grid(alpha=0.3)
    ax.legend()

    # --------------------------------------------------------
    # Panel 2: Z-score
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        M_VALUES,
        mpe_z,
        marker="o",
        linewidth=2
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    # Referencias visuales aproximadas para ±1.96
    ax.axhline(
        1.96,
        linestyle=":",
        linewidth=1,
        label=r"$Z=1.96$"
    )

    ax.axhline(
        -1.96,
        linestyle=":",
        linewidth=1,
        label=r"$Z=-1.96$"
    )

    ax.set_xlabel(r"$m$")
    ax.set_ylabel(
        r"$Z=(\mathrm{mPE}_{obs}-"
        r"\langle\mathrm{mPE}_{null}\rangle)"
        r"/\sigma_{null}$"
    )
    ax.set_title("Z-score de mPE")
    ax.set_xticks(M_VALUES)
    ax.grid(alpha=0.3)
    ax.legend()

    # --------------------------------------------------------
    # Panel 3: null - observado
    # --------------------------------------------------------

    ax = axes[2]

    ax.plot(
        M_VALUES,
        diferencia,
        marker="o",
        linewidth=2
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(r"$m$")
    ax.set_ylabel(
        r"$\langle\mathrm{mPE}_{null}\rangle"
        r"-\mathrm{mPE}_{obs}$"
    )
    ax.set_title("Diferencia nulo − observado")
    ax.set_xticks(M_VALUES)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Entropía de permutación modificada: {nombre}",
        fontsize=14
    )

    fig.tight_layout()

    ruta = output_dir / f"mPE_{nombre}.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Exportación de resultados
# ============================================================

def guardar_resultados(resultados, output_dir):
    tablas_C = []
    tablas_mpe = []
    tablas_pendientes = []
    pendientes_individuales = []
    mpe_null_individual = []

    for resultado in resultados:
        nombre = resultado["nombre"]

        # ----------------------------------------------------
        # C_d observado y estadísticos nulos
        # ----------------------------------------------------

        tabla_C = pd.DataFrame(
            {
                "representacion": nombre,
                "d": resultado["d"],
                "C_observado": resultado["C_obs"],
                "C_null_media": resultado["C_stats"]["mean"],
                "C_null_std": resultado["C_stats"]["std"],
                "C_null_q025": resultado["C_stats"]["q025"],
                "C_null_mediana": resultado["C_stats"]["median"],
                "C_null_q975": resultado["C_stats"]["q975"]
            }
        )

        tablas_C.append(tabla_C)

        # ----------------------------------------------------
        # mPE
        # ----------------------------------------------------

        tabla_mpe = pd.DataFrame(
            {
                "representacion": nombre,
                "m": M_VALUES,
                "tau": TAU,
                "mPE_observada": resultado["mpe_obs"],
                "mPE_null_media": resultado["mpe_stats"]["mean"],
                "mPE_null_std": resultado["mpe_stats"]["std"],
                "mPE_null_q025": resultado["mpe_stats"]["q025"],
                "mPE_null_mediana": resultado["mpe_stats"]["median"],
                "mPE_null_q975": resultado["mpe_stats"]["q975"],
                "mPE_zscore": resultado["mpe_z"],
                "mPE_null_menos_observada":
                    resultado["mpe_null_minus_obs"]
            }
        )

        tablas_mpe.append(tabla_mpe)

        # ----------------------------------------------------
        # Resumen de pendientes
        # ----------------------------------------------------

        tabla_pendiente = pd.DataFrame(
            {
                "representacion": [nombre],
                "longitud_serie": [len(resultado["arr"])],
                "fit_d_min": [FIT_D_MIN],
                "fit_d_max": [
                    FIT_D_MAX
                    if FIT_D_MAX is not None
                    else int(np.max(resultado["d"]))
                ],
                "pendiente_observada": [
                    resultado["ajuste_obs"]["slope"]
                ],
                "constante_decaimiento_lambda": [
                    -resultado["ajuste_obs"]["slope"]
                ],
                "intercepto_observado": [
                    resultado["ajuste_obs"]["intercept"]
                ],
                "R2_observado": [
                    resultado["ajuste_obs"]["r2"]
                ],
                "error_estandar_pendiente_observada": [
                    resultado["ajuste_obs"]["stderr"]
                ],
                "pendiente_null_media": [
                    resultado["slope_media"]
                ],
                "pendiente_null_std": [
                    resultado["slope_std"]
                ],
                "pendiente_null_q025": [
                    resultado["slope_q025"]
                ],
                "pendiente_null_mediana": [
                    resultado["slope_mediana"]
                ],
                "pendiente_null_q975": [
                    resultado["slope_q975"]
                ],
                "zscore_pendiente": [
                    resultado["slope_z"]
                ],
                "p_empirico_bilateral": [
                    resultado["slope_p"]
                ],
                "p_decaimiento_mas_rapido": [
                    resultado["p_decaimiento_mas_rapido"]
                ],
                "p_decaimiento_mas_lento": [
                    resultado["p_decaimiento_mas_lento"]
                ],
                "n_shuffles": [N_SHUFFLES]
            }
        )

        tablas_pendientes.append(tabla_pendiente)

        # ----------------------------------------------------
        # Pendientes de cada shuffle
        # ----------------------------------------------------

        tabla_pendientes_ind = pd.DataFrame(
            {
                "representacion": nombre,
                "shuffle": np.arange(1, N_SHUFFLES + 1),
                "pendiente": resultado["slope_null"],
                "intercepto": resultado["intercept_null"],
                "R2": resultado["r2_null"]
            }
        )

        pendientes_individuales.append(tabla_pendientes_ind)

        # ----------------------------------------------------
        # mPE individual de cada shuffle, formato largo
        # ----------------------------------------------------

        for i_shuffle in range(N_SHUFFLES):
            for j, m in enumerate(M_VALUES):
                mpe_null_individual.append(
                    {
                        "representacion": nombre,
                        "shuffle": i_shuffle + 1,
                        "m": int(m),
                        "mPE_null":
                            resultado["mpe_null"][i_shuffle, j]
                    }
                )

    pd.concat(
        tablas_C,
        ignore_index=True
    ).to_csv(
        output_dir / "Cd_observado_y_null.csv",
        index=False
    )

    pd.concat(
        tablas_mpe,
        ignore_index=True
    ).to_csv(
        output_dir / "mPE_observada_y_null.csv",
        index=False
    )

    pd.concat(
        tablas_pendientes,
        ignore_index=True
    ).to_csv(
        output_dir / "resumen_pendientes_Cd.csv",
        index=False
    )

    pd.concat(
        pendientes_individuales,
        ignore_index=True
    ).to_csv(
        output_dir / "pendientes_shuffles.csv",
        index=False
    )

    pd.DataFrame(
        mpe_null_individual
    ).to_csv(
        output_dir / "mPE_shuffles_formato_largo.csv",
        index=False
    )


# ============================================================
# Ejecución
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generadores independientes para alturas e intervalos.
    semilla_maestra = np.random.SeedSequence(SEED)
    semillas_hijas = semilla_maestra.spawn(2)

    rng_alturas = np.random.default_rng(semillas_hijas[0])
    rng_intervalos = np.random.default_rng(semillas_hijas[1])

    resultado_alturas = analizar_representacion(
        nombre="alturas_MIDI",
        arr=midi,
        rng=rng_alturas
    )

    resultado_intervalos = analizar_representacion(
        nombre="intervalos",
        arr=midi,
        rng=rng_alturas
    )

    resultados = [
        resultado_alturas,
        resultado_intervalos
    ]

    for resultado in resultados:
        graficar_C_y_pendiente(
            resultado,
            OUTPUT_DIR
        )

        graficar_mpe(
            resultado,
            OUTPUT_DIR
        )

    guardar_resultados(
        resultados,
        OUTPUT_DIR
    )

    print("\n" + "=" * 70)
    print("Análisis terminado.")
    print(f"Resultados guardados en: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()