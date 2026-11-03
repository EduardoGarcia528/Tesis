"""
indice_no_linealidad_mdfa.py

Implementación compacta del índice de no linealidad xi basado en MDFA
y surrogados IAAFT, siguiendo la construcción descrita en:

A. González-Espinoza, G. Martínez-Mekler, L. Lacasa,
"Arrow of time across five centuries of classical music",
Physical Review Research 2, 033166 (2020), Appendix B.

La función principal es:

    xi, alpha = indice_no_linealidad_mdfa(x, plot=False)

y, si se desea visualizar la curva MDFA y la banda de surrogados:

    xi, alpha = indice_no_linealidad_mdfa(x, plot=True)

Notas de fidelidad:
- Se generan 20 surrogados IAAFT por defecto, como en el artículo.
- El MDFA usa la serie de magnitudes de los incrementos.
- Esa serie de magnitudes se integra una vez ANTES del DFA.
- El DFA hace su integración estándar mediante fathonUtils.toAggregated.
- Se usa F(s)/s para compensar la integración adicional del MDFA.
- xi se calcula con las derivadas de UN ajuste polinomial GLOBAL para
  la curva original y para cada surrogate.
- El artículo no especifica numéricamente:
    (i) el grado del polinomio global,
    (ii) el orden del detrending del DFA,
    (iii) la malla exacta de tamaños s.
  Por ello se conservan como defaults las decisiones del código previo:
    grado global = 4, detrending DFA = 2,
    s generado por fathonUtils.linRangeByStep(20, min(500, N/4)).

Dependencias:
    numpy
    matplotlib
    fathon
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu


def _validate_1d(x):
    """Valida y limpia una serie 1D finita."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("La entrada debe ser un array 1D.")

    if x.size < 8:
        raise ValueError("La serie es demasiado corta.")

    if not np.all(np.isfinite(x)):
        raise ValueError("La serie contiene NaN o infinitos.")

    if np.allclose(np.std(x), 0.0):
        raise ValueError("La serie es constante.")

    return x


def iaaft_surrogate(
    x,
    rng,
    max_iter=10_000,
    tol=1e-8,
):
    """
    Genera UN surrogate IAAFT.

    Preserva exactamente la distribución marginal (por reordenamiento)
    y aproxima el espectro de amplitudes de la serie original.

    Parameters
    ----------
    x : array_like, shape (N,)
        Serie original.
    rng : numpy.random.Generator
        Generador aleatorio.
    max_iter : int
        Máximo de iteraciones.
    tol : float
        Tolerancia sobre el cambio relativo del error espectral.

    Returns
    -------
    y : ndarray, shape (N,)
        Surrogate IAAFT.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    x_sorted = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))

    # Inicialización mediante permutación aleatoria
    y = rng.permutation(x).astype(float)

    prev_error = np.inf
    target_norm = np.linalg.norm(target_amp)
    if target_norm == 0:
        return y

    for _ in range(max_iter):
        # Paso de ajuste espectral
        Y = np.fft.rfft(y)
        phases = np.angle(Y)
        y_spectral = np.fft.irfft(
            target_amp * np.exp(1j * phases),
            n=n,
        )

        # Paso de ajuste de amplitudes:
        # se impone exactamente la misma distribución marginal que x.
        order = np.argsort(y_spectral, kind="mergesort")
        y_new = np.empty_like(y_spectral)
        y_new[order] = x_sorted

        # Error espectral relativo
        current_amp = np.abs(np.fft.rfft(y_new))
        error = np.linalg.norm(current_amp - target_amp) / target_norm

        # Convergencia: el error espectral deja de cambiar
        if np.isfinite(prev_error):
            if abs(prev_error - error) <= tol * max(1.0, prev_error):
                y = y_new
                break

        y = y_new
        prev_error = error

    return np.asarray(y, dtype=float)


def _dfa_fluctuation(series, win_sizes, dfa_order=2):
    """
    DFA estándar usando fathon.

    IMPORTANTE:
    fu.toAggregated(series) implementa el perfil integrado estándar del DFA.
    """
    series = np.asarray(series, dtype=float)
    aggregated = fu.toAggregated(series)

    pydfa = fathon.DFA(aggregated)
    n, F = pydfa.computeFlucVec(
        win_sizes,
        revSeg=False,
        polOrd=dfa_order,
    )

    n = np.asarray(n, dtype=float)
    F = np.asarray(F, dtype=float)

    valid = np.isfinite(n) & np.isfinite(F) & (F > 0)
    return n[valid], F[valid]


def _mdfa_curve(x, win_sizes, dfa_order=2):
    """
    Calcula la curva MDFA de magnitudes.

    Procedimiento:
    1. Incrementos Delta x.
    2. Magnitudes |Delta x|.
    3. Restar la media.
    4. Integración adicional de la serie de magnitudes.
    5. DFA estándar sobre la serie ya integrada.
    6. Dividir F(s) entre s para compensar la integración adicional.

    Returns
    -------
    s : ndarray
        Tamaños de ventana.
    Fcorr : ndarray
        Curva corregida F(s)/s.
    """
    dx = np.diff(x)

    magnitude = np.abs(dx)
    magnitude = magnitude - np.mean(magnitude)

    # Integración adicional prescrita por el MDFA.
    magnitude_integrated = np.cumsum(magnitude)

    # El DFA hará después su integración estándar.
    s, F = _dfa_fluctuation(
        magnitude_integrated,
        win_sizes=win_sizes,
        dfa_order=dfa_order,
    )

    Fcorr = F / s

    valid = np.isfinite(Fcorr) & (Fcorr > 0)
    return s[valid], Fcorr[valid]


def _polyfit_slopes(log_s, log_F, degree):
    """
    Ajusta UN polinomio global y devuelve su derivada en cada log(s).
    """
    log_s = np.asarray(log_s, dtype=float)
    log_F = np.asarray(log_F, dtype=float)

    if len(log_s) <= degree:
        raise ValueError(
            f"Hay {len(log_s)} puntos, insuficientes para un "
            f"polinomio de grado {degree}."
        )

    coeffs = np.polyfit(log_s, log_F, degree)
    polynomial = np.poly1d(coeffs)
    derivative = np.polyder(polynomial)

    fitted = polynomial(log_s)
    slopes = derivative(log_s)

    return fitted, slopes, coeffs


def indice_no_linealidad_mdfa(
    time_series,
    *,
    n_surrogates=20,
    s_min=20,
    s_max=None,
    dfa_order=2,
    poly_degree=4,
    seed=None,
    iaaft_max_iter=10_000,
    iaaft_tol=1e-8,
    plot=False,
    show_surrogate_mean=True,
):
    """
    Calcula el índice de no linealidad xi mediante MDFA + IAAFT.

    Parameters
    ----------
    time_series : array_like, shape (N,)
        Serie temporal 1D.

    n_surrogates : int, default=20
        Número de surrogados IAAFT. El artículo usa 20.

    s_min : int, default=20
        Tamaño mínimo de ventana.

    s_max : int or None, default=None
        Tamaño máximo de ventana. Si es None:
            min(500, (N-1)//4)
        donde N-1 es la longitud de la serie de incrementos.

    dfa_order : int, default=2
        Orden del polinomio de detrending dentro del DFA.
        El artículo describe el DFA con orden genérico m; no fija aquí
        un valor numérico. Se conserva 2 del código original del usuario.

    poly_degree : int, default=4
        Grado del polinomio GLOBAL usado para suavizar log[F(s)/s]
        antes de derivar. El artículo no especifica el grado; se conserva
        4 del código original del usuario.

    seed : int or None
        Semilla reproducible para los surrogados.

    iaaft_max_iter : int, default=10000
        Máximo de iteraciones del IAAFT.

    iaaft_tol : float, default=1e-8
        Tolerancia de convergencia del IAAFT.

    plot : bool, default=False
        Si True, muestra la curva MDFA original junto con la banda
        min-max de los surrogados, como representación cercana a la
        mostrada en la Fig. 15 del artículo.

    show_surrogate_mean : bool, default=True
        Si True, añade también la media del ensamble surrogate.

    Returns
    -------
    xi : float
        Índice de no linealidad:
            xi = mean_s(
                |slope_real(s) - mean[slope_sur(s)]|
                / std[slope_sur(s)]
            )

    alpha : float
        Exponente de escalamiento global EFECTIVO de la curva MDFA
        corregida, obtenido mediante una regresión lineal de

            log10(F(s)/s) vs log10(s).

        Dado que el artículo reconoce múltiples regiones de escalamiento,
        este alpha debe interpretarse como un resumen global, no como prueba
        de una única ley de potencia.

    Notes
    -----
    - xi reproduce la estructura de la Eq. (B2).
    - Para xi se usa el ajuste polinomial global, no una regresión segmentada.
    - La banda gráfica surrogate es min-max del ensamble de 20 surrogados.
    """
    x = _validate_1d(time_series)

    if n_surrogates < 2:
        raise ValueError("n_surrogates debe ser al menos 2.")

    if poly_degree < 1:
        raise ValueError("poly_degree debe ser >= 1.")

    if dfa_order < 1:
        raise ValueError("dfa_order debe ser >= 1.")

    # La serie de magnitudes tiene longitud N-1.
    n_magnitude = len(x) - 1

    if s_max is None:
        s_max = min(500, n_magnitude // 4)

    s_min = int(s_min)
    s_max = int(s_max)

    if s_max <= s_min:
        raise ValueError(
            f"Rango de escalas inválido: s_min={s_min}, s_max={s_max}. "
            "Se necesita una serie más larga o reducir s_min."
        )

    # Se conserva exactamente el generador de escalas del código previo.
    # El artículo no especifica la malla exacta de s.
    win_sizes = fu.linRangeByStep(s_min, s_max)

    # ---------- Curva original ----------
    s, Fcorr = _mdfa_curve(
        x,
        win_sizes=win_sizes,
        dfa_order=dfa_order,
    )

    if len(s) <= poly_degree + 1:
        raise ValueError(
            "No quedaron suficientes escalas válidas para el ajuste."
        )

    log_s = np.log10(s)
    log_Fcorr = np.log10(Fcorr)

    fitted_real, slopes_real, _ = _polyfit_slopes(
        log_s,
        log_Fcorr,
        degree=poly_degree,
    )

    # Exponente global efectivo de la curva corregida
    alpha = float(np.polyfit(log_s, log_Fcorr, 1)[0])

    # ---------- Surrogados ----------
    rng = np.random.default_rng(seed)

    surrogate_logF = np.empty((n_surrogates, len(s)), dtype=float)
    surrogate_slopes = np.empty((n_surrogates, len(s)), dtype=float)

    for j in range(n_surrogates):
        x_surr = iaaft_surrogate(
            x,
            rng=rng,
            max_iter=iaaft_max_iter,
            tol=iaaft_tol,
        )

        s_surr, Fcorr_surr = _mdfa_curve(
            x_surr,
            win_sizes=win_sizes,
            dfa_order=dfa_order,
        )

        # Con series de igual longitud y misma malla, normalmente coincide.
        # Se verifica explícitamente para no comparar escalas distintas.
        if len(s_surr) != len(s) or not np.allclose(s_surr, s):
            raise RuntimeError(
                "Las escalas válidas del surrogate no coinciden con las "
                "de la serie original."
            )

        log_F_surr = np.log10(Fcorr_surr)
        surrogate_logF[j] = log_F_surr

        _, slopes_surr, _ = _polyfit_slopes(
            log_s,
            log_F_surr,
            degree=poly_degree,
        )
        surrogate_slopes[j] = slopes_surr

    # ---------- Índice xi ----------
    mean_slope_sur = np.mean(surrogate_slopes, axis=0)
    std_slope_sur = np.std(surrogate_slopes, axis=0, ddof=0)

    valid = (
        np.isfinite(slopes_real)
        & np.isfinite(mean_slope_sur)
        & np.isfinite(std_slope_sur)
        & (std_slope_sur > np.finfo(float).eps)
    )

    if not np.all(valid):
        warnings.warn(
            "Algunas escalas tienen desviación estándar surrogate nula "
            "o no finita y se excluyeron del promedio de xi.",
            RuntimeWarning,
        )

    if not np.any(valid):
        raise RuntimeError(
            "No hay escalas válidas para calcular xi."
        )

    z_abs = (
        np.abs(slopes_real[valid] - mean_slope_sur[valid])
        / std_slope_sur[valid]
    )

    xi = float(np.mean(z_abs))

    # ---------- Gráfica ----------
    if plot:
        mean_logF_sur = np.mean(surrogate_logF, axis=0)
        lower = np.min(surrogate_logF, axis=0)
        upper = np.max(surrogate_logF, axis=0)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.fill_between(
            log_s,
            lower,
            upper,
            alpha=0.25,
            label="IAAFT surrogates: min-max",
        )

        if show_surrogate_mean:
            ax.plot(
                log_s,
                mean_logF_sur,
                linestyle="--",
                linewidth=1.5,
                label="IAAFT surrogates: mean",
            )

        ax.plot(
            log_s,
            log_Fcorr,
            ".",
            markersize=4,
            label="Original MDFA",
        )

        ax.plot(
            log_s,
            fitted_real,
            linewidth=1.6,
            label=f"Global polynomial fit (degree {poly_degree})",
        )

        ax.set_xlabel(r"$\log_{10}(s)$")
        ax.set_ylabel(r"$\log_{10}[F(s)/s]$")
        ax.set_title(
            rf"MDFA: $\xi={xi:.3f}$, $\alpha_{{eff}}={alpha:.3f}$"
        )
        ax.legend()
        fig.tight_layout()
        plt.show()

    return xi, alpha


if __name__ == "__main__":
    # Ejemplo mínimo de uso.
    rng = np.random.default_rng(1234)
    x = rng.normal(size=2000)

    xi, alpha = indice_no_linealidad_mdfa(
        x,
        seed=1234,
        plot=True,
    )

    print("xi =", xi)
    print("alpha_eff =", alpha)
