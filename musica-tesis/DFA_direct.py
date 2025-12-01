import pandas as pd
import numpy as np
import piecewise_regression
from tqdm import tqdm
from scipy.io import wavfile
import matplotlib.pyplot as plt
import fathon
from scipy.signal import welch, get_window
from scipy.stats import linregress
from fathon import fathonUtils as fu
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly, hilbert
from fractions import Fraction

def plot_psd_beta(time_series, fs, 
                  fmin=None, fmax=None, 
                  nperseg=2**14, noverlap=None, window='hann',
                  bins_per_decade=None,  # e.g. 12 para promediar en bins log
                  detrend='constant', show=True):
    """
    Grafica PSD (Welch) en log–log y ajusta una recta para estimar Beta en S(f) ~ f^{-Beta}.
    
    Parámetros
    ----------
    time_series : array-like
        Señal temporal (1D).
    fs : float
        Frecuencia de muestreo (Hz).
    fmin, fmax : float or None
        Rango de frecuencias (Hz) a usar en el ajuste. Si None, usa todo el rango válido.
    nperseg : int
        Tamaño de segmento para Welch (potencia de 2 recomendable).
    noverlap : int or None
        Overlap para Welch (por defecto nperseg//2).
    window : str
        Ventana para Welch.
    bins_per_decade : int or None
        Si se especifica, promedia la PSD en bins logarítmicos (robusto ante picos).
    detrend : {'constant','linear', None}
        Detrend interno de Welch.
    show : bool
        Si True, muestra la figura.

    Returns
    -------
    beta : float
        Exponente tal que PSD ~ f^{-beta}.
    r2 : float
        Coeficiente de determinación del ajuste en log–log.
    slope, intercept : floats
        Parámetros de la recta en log10: log10(PSD) = slope * log10(f) + intercept.
    (f, Pxx) : ndarray
        Frecuencias y PSD originales de Welch.
    """

    x = np.asarray(time_series, dtype=np.float64)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)

    if noverlap is None:
        noverlap = nperseg // 2

    # PSD con Welch
    f, Pxx = welch(x, fs=fs, window=get_window(window, nperseg),
                   nperseg=nperseg, noverlap=noverlap,
                   detrend=detrend, return_onesided=True,
                   scaling='density', average='mean')

    # Mantener valores válidos
    valid = (f > 0) & np.isfinite(f) & np.isfinite(Pxx) & (Pxx > 0)
    if fmin is not None:
        valid &= (f >= fmin)
    if fmax is not None:
        valid &= (f <= fmax)

    f_fit_raw = f[valid]
    P_fit_raw = Pxx[valid]
    if f_fit_raw.size < 3:
        raise ValueError("Muy pocos puntos en el rango seleccionado para ajustar.")

    # (Opcional) promediado logarítmico para robustez
    if bins_per_decade is not None and bins_per_decade > 0:
        logf = np.log10(f_fit_raw)
        lo, hi = logf.min(), logf.max()
        nb = max(5, int((hi - lo) * bins_per_decade))
        edges = np.linspace(lo, hi, nb + 1)
        f_fit, P_fit = [], []
        for i in range(nb):
            m = (logf >= edges[i]) & (logf < edges[i+1])
            if m.sum() >= 3:
                f_fit.append(10**(logf[m].mean()))
                P_fit.append(10**(np.log10(P_fit_raw:=P_fit_raw if False else P_fit_raw) if False else np.log10(P_fit_raw)[:1]))  # placeholder to keep lints calm
                # promedio geométrico de potencia
                P_fit.append(10**(np.log10(P_fit_raw[m]).mean()))
        # La línea anterior agregó de más; rehagamos correctamente:
        f_fit, P_fit = [], []
        for i in range(nb):
            m = (logf >= edges[i]) & (logf < edges[i+1])
            if m.sum() >= 3:
                f_fit.append(10**(logf[m].mean()))
                P_fit.append(10**(np.log10(P_fit_raw[m]).mean()))
        f_fit = np.array(f_fit)
        P_fit = np.array(P_fit)
    else:
        f_fit = f_fit_raw
        P_fit = P_fit_raw

    # Ajuste en log10
    X = np.log10(f_fit)
    Y = np.log10(P_fit)
    slope, intercept, r, _, _ = linregress(X, Y)
    r2 = r**2
    beta = -slope  # porque log10(PSD) = slope*log10(f) + c  => PSD ~ f^{slope} = f^{-beta}

    # Curva ajustada (sobre el rango usado)
    y_fit = 10**(intercept + slope * np.log10(f_fit))

    # Gráfica
    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.loglog(f, Pxx, lw=1.2, label='PSD (Welch)')
    ax.loglog(f_fit, y_fit, '--', lw=2.0,
              label=f"Ajuste: β = {beta:.3f}  (R² = {r2:.3f})")
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('PSD [unidad²/Hz]')
    rtxt = f"Rango ajuste: [{f_fit.min():.3g}, {f_fit.max():.3g}] Hz"
    ax.set_title(f"Espectro de Potencias y Ajuste log–log\n{rtxt}")
    ax.grid(True, which='both', ls=':')
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()

    return beta, r2, slope, intercept, (f, Pxx)


def DFA(time_series, binlog=True):
    # Convertir la serie a un formato compatible con Fathon
    my_data = fu.toAggregated(time_series)
    data_length = len(my_data)

    # Definir tamaños de ventana
    winSizes = fu.linRangeByStep(20, min(500, data_length // 4))
    revSeg = False
    polOrd = 2

    # Crear el objeto DFA y calcular las fluctuaciones
    pydfa = fathon.DFA(my_data)
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    if not binlog:
        return n, F
    else:
        # Aplicar log-binning
        log_bins = np.logspace(np.log10(min(n)), np.log10(max(n)), num=50)
        binned_F = []
        binned_n = []
        for i in range(len(log_bins) - 1):
            mask = (n >= log_bins[i]) & (n < log_bins[i + 1])
            if np.any(mask):
                binned_n.append(np.mean(n[mask]))
                binned_F.append(np.mean(F[mask]))
        return np.array(binned_n), np.array(binned_F)
    
def decompose_series(increment_series):
    """Descompone la serie de incrementos en magnitudes y signos, y resta las medias."""
    magnitude_series = np.abs(increment_series)
    sign_series = np.sign(increment_series)
    
    # Restar las medias
    magnitude_series -= np.mean(magnitude_series)
    sign_series -= np.mean(sign_series)
    
    return magnitude_series, sign_series


def mdfa(time_series, binlog):
    """
    Implementación del MDFA siguiendo los pasos especificados.
    :param time_series: Serie temporal (numpy array).
    :param scale_range: Rango de tamaños de ventana para el análisis de escala.
    :return: Exponentes de escala para la serie de magnitudes y signos.
    """
    # Paso 1: Calcular la serie de incrementos
    increment_series = np.diff(time_series)
    
    # Paso 2: Descomponer la serie de incrementos en magnitud y signo, restar medias
    magnitude_series, sign_series = decompose_series(increment_series)
    
    # Paso 3: Aplicar DFA a ambas series 
    n_mag, F_mag = DFA(magnitude_series,binlog)
    # alpha_sign = DFA(sign_series)

    
    return n_mag, F_mag

"""iaaft - Iterative amplitude adjusted Fourier transform surrogates
        #! /usr/bin/env python3
        This module implements the IAAFT method [1] to generate time series
        surrogates (i.e. randomized copies of the original time series) which
        ensures that each randomised copy preserves the power spectrum of the
        original time series.

[1] Venema, V., Ament, F. & Simmer, C. A stochastic iterative amplitude
    adjusted Fourier Transform algorithm with improved accuracy (2006), Nonlin.
    Proc. Geophys. 13, pp. 321--328
    https://doi.org/10.5194/npg-13-321-2006

"""
# Created: Tue Jun 22, 2021  09:44am
# Last modified: Tue Jun 22, 2021  12:39pm
#
# Copyright (C) 2021  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de> This
# program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


import numpy as np
from tqdm import tqdm


def iaaft(x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"):
    """
    Returns iAAFT surrogates of given time series.

    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.

    See Also
    --------
    numpy.argsort

    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs

def ajuste_polinomial(x, y, grado):
    """
    Ajusta un polinomio de grado especificado a los datos y grafica el ajuste.
    
    :param x: Array de valores x de los datos.
    :param y: Array de valores y de los datos.
    :param grado: Grado del polinomio a ajustar.
    :return: Coeficientes del polinomio ajustado.
    """
    # Ajustar el polinomio usando np.polyfit
    coeficientes = np.polyfit(x, y, grado)
    
    # Crear un polinomio a partir de los coeficientes ajustados
    polinomio = np.poly1d(coeficientes)
    
    # Evaluar el polinomio ajustado en los puntos x
    y_ajustado = polinomio(x)
    
    # Retornar los coeficientes del polinomio
    return y_ajustado, coeficientes

def evaluar_derivada(coeficientes, x_valor):
    """
    Calcula la primera derivada de un polinomio y la evalúa en un valor arbitrario.
    
    :param coeficientes: Coeficientes del polinomio (array).
    :param x_valor: Valor en el que se desea evaluar la derivada.
    :return: Valor de la primera derivada del polinomio evaluada en x_valor.
    """
    # Crear el polinomio a partir de los coeficientes
    polinomio = np.poly1d(coeficientes)
    
    # Calcular la primera derivada del polinomio
    derivada = np.polyder(polinomio)
    
    # Evaluar la derivada en el valor especificado
    derivada_evaluada = derivada(x_valor)
    return np.array(derivada_evaluada)

def ajuste_polinomial_auto(x, y, grado_max=10, criterio="mse"):
    """
    Determina el grado óptimo del polinomio que mejor se ajusta a los datos
    y devuelve el polinomio ajustado junto con sus coeficientes.
    
    :param x: Array de valores x de los datos.
    :param y: Array de valores y de los datos.
    :param grado_max: Máximo grado de polinomio a evaluar.
    :param criterio: Criterio para seleccionar el grado óptimo ("mse", "aic" o "bic").
    :return: y_ajustado (valores del polinomio en los puntos x) y coeficientes del polinomio ajustado.
    """
    mejor_grado = 0
    mejor_metrica = np.inf
    mejor_coeficientes = None
    
    n = len(x)  # Número de datos
    
    for grado in range(1, grado_max + 1):
        # Ajustar el polinomio
        coeficientes = np.polyfit(x, y, grado)
        polinomio = np.poly1d(coeficientes)
        y_pred = polinomio(x)
        
        # Calcular el error cuadrático medio
        mse = np.mean((y - y_pred) ** 2)
        
        if criterio == "mse":
            metrica = mse
        elif criterio == "aic":
            # AIC = n * log(mse) + 2 * (p + 1), donde p es el grado
            metrica = n * np.log(mse) + 2 * (grado + 1)
        elif criterio == "bic":
            # BIC = n * log(mse) + log(n) * (p + 1)
            metrica = n * np.log(mse) + np.log(n) * (grado + 1)
        else:
            raise ValueError("Criterio no válido. Usa 'mse', 'aic' o 'bic'.")
        
        # Verificar si este grado es mejor
        if metrica < mejor_metrica:
            mejor_metrica = metrica
            mejor_grado = grado
            mejor_coeficientes = coeficientes
    
    # Crear el polinomio con los coeficientes del mejor ajuste
    mejor_polinomio = np.poly1d(mejor_coeficientes)
    y_ajustado = mejor_polinomio(x)
    
    return y_ajustado, mejor_coeficientes, mejor_grado

def main(time_series, method, binlog=False, grado_polinomio=4, N_surrogates=100, graficar=True, xi=True):
    H = []

    # CALCULAR MDFA O DFA
    if method == 'MDFA':
        n, F = mdfa(time_series,binlog=binlog)
    elif method == 'DFA':
        n, F = DFA(time_series, binlog=binlog)
    else:
        raise ValueError("El método debe ser 'MDFA' o 'DFA'")

    # GENERAR SURROGADOS
    if xi == True:
        flucts_surrogates = []
        surrogates = iaaft(time_series, N_surrogates)
        for i in range(N_surrogates):
            if method == 'MDFA':
                n_surr, flucts_surr = mdfa(surrogates[i, :],binlog=binlog)
            elif method == 'DFA':
                n_surr, flucts_surr = DFA(surrogates[i, :], binlog=binlog)
            flucts_surrogates.append(np.log10(flucts_surr))
        flucts_surrogates = np.vstack(flucts_surrogates)

    # REGRESIÓN SEGMENTADA
    n_breakpoints = 1
    while True:
        pw_fit = piecewise_regression.Fit(np.log10(n), np.log10(F), n_breakpoints=n_breakpoints)
        pw_results = pw_fit.get_results()
        pw_estimates = pw_results["estimates"]

        if pw_results['converged']:
            print('converged')
            break
        elif n_breakpoints != 1:
            n_breakpoints -= 1
        else:
            break
    
    if not pw_results['converged']:
        print('Not converged')
        return np.nan

    for value in pw_estimates:
        if 'alpha' in value:
            H.append(pw_estimates[value]['estimate'])

    if graficar:
        pw_fit.plot_data(color="red", s=1, label='log(F(s))')
        # pw_fit.plot_breakpoints()

    if xi == True:
        lower_bound = np.min(flucts_surrogates, axis=0)
        upper_bound = np.max(flucts_surrogates, axis=0)
    if graficar:
        if xi == True:
            plt.fill_between(np.log10(n_surr), lower_bound, upper_bound, color='blue', alpha=0.2, label='Surrogates')

    # AJUSTE POLINOMIAL Y DERIVADAS
    if not binlog:
        # === Ajuste por segmentos ===
        past_idx = 0
        derivadas = np.array([])
        y_ajustados = np.array([])

        for i in range(n_breakpoints):
            pts = pw_estimates[f'breakpoint{i + 1}']['estimate']
            idx_closest = int(np.argmin(np.abs(np.log10(n) - pts)))
            y_ajustado, coeficientes = ajuste_polinomial(np.log10(n)[past_idx:idx_closest + 1], np.log10(F)[past_idx:idx_closest + 1], grado=grado_polinomio)
            derivadas = np.concatenate((derivadas, evaluar_derivada(coeficientes, np.log10(n)[past_idx:idx_closest + 1])))
            y_ajustados = np.concatenate((y_ajustados, y_ajustado))
            past_idx = idx_closest + 1

        y_ajustado, coeficientes = ajuste_polinomial(np.log10(n)[past_idx:], np.log10(F)[past_idx:], grado=grado_polinomio)
        derivadas = np.concatenate((derivadas, evaluar_derivada(coeficientes, np.log10(n)[past_idx:])))
        y_ajustados = np.concatenate((y_ajustados, y_ajustado))

        if graficar:
            plt.plot(np.log10(n), y_ajustados, color='green', label=f'Ajuste polinomial (grado {grado_polinomio})')

        if xi == True:
            derivadas_surr = np.zeros([N_surrogates, len(np.log10(n_surr))])
            for surr_index in range(N_surrogates):
                past_idx_surr = 0
                for i in range(n_breakpoints):
                    pts_surr = pw_estimates[f'breakpoint{i + 1}']['estimate']
                    idx_closest_surr = np.argmin(np.abs(np.log10(n_surr) - pts_surr))
                    _, coef_surr = ajuste_polinomial(
                        np.log10(n_surr)[past_idx_surr:idx_closest_surr + 1],
                        flucts_surrogates[surr_index, past_idx_surr:idx_closest_surr + 1],
                        grado=grado_polinomio
                    )
                    derivadas_surr[surr_index, past_idx_surr:idx_closest_surr + 1] = evaluar_derivada(
                        coef_surr, np.log10(n_surr)[past_idx_surr:idx_closest_surr + 1]
                    )
                    past_idx_surr = idx_closest_surr + 1

                _, coef_surr = ajuste_polinomial(
                    np.log10(n_surr)[past_idx_surr:], 
                    flucts_surrogates[surr_index, past_idx_surr:], 
                    grado=grado_polinomio
                )
                derivadas_surr[surr_index, past_idx_surr:] = evaluar_derivada(
                    coef_surr, np.log10(n_surr)[past_idx_surr:]
                )

    else:
        # === Ajuste global ===
        y_ajustado, coeficientes = ajuste_polinomial(np.log10(n), np.log10(F), grado=grado_polinomio)
        derivadas = evaluar_derivada(coeficientes, np.log10(n))

        if graficar:
            plt.plot(np.log10(n), y_ajustado, color='green', label=f'Ajuste polinomial (grado {grado_polinomio})')

        if xi == True:
            derivadas_surr = np.zeros([N_surrogates, len(np.log10(n_surr))])
            for surr_index in range(N_surrogates):
                _, coef_surr = ajuste_polinomial(np.log10(n_surr), flucts_surrogates[surr_index], grado=grado_polinomio)
                derivadas_surr[surr_index, :] = evaluar_derivada(coef_surr, np.log10(n_surr))

    if xi == True:
        derivadas_surr_mean = np.mean(derivadas_surr, axis=0)
        derivadas_surr_std = np.std(derivadas_surr, axis=0)

    if graficar:
        plt.xlabel('log(s)', fontsize=8)
        plt.ylabel('log(F(s))', fontsize=8)
        plt.title(r'DFA ($\alpha$'+ f'={coeficientes[0]:.4f} )', fontsize=8)
        plt.legend(loc=0, fontsize=7)
        plt.show()
        plt.close()

        if xi == True:  
            plt.figure()
            
            plt.plot(np.log10(n), derivadas, color='green', ls='', marker='.', label='Derivadas de la serie original')
            plt.fill_between(
                np.log10(n_surr),
                derivadas_surr_mean - derivadas_surr_std,
                derivadas_surr_mean + derivadas_surr_std,
                color='blue', alpha=0.2, label='Surrogates: Mean ± Std'
            )
            plt.xlabel('log(s)', fontsize=14)
            plt.ylabel("Derivada del polinomio", fontsize=14)
            plt.title("Derivadas del ajuste polinomial vs log(s)", fontsize=14)
            plt.legend(loc=0, fontsize=10)
            plt.show()

    # ÍNDICE DE NO LINEALIDAD
    if xi == True:
        xi = np.sum(np.abs(derivadas - derivadas_surr_mean) / derivadas_surr_std)
        xi_index = xi / len(derivadas)
        return xi_index
    else:
        return H


def _to_float_mono(x):
    # a) a float64
    if np.issubdtype(x.dtype, np.integer):
        max_int = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / max_int
    else:
        x = x.astype(np.float64)
    # b) a mono
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x

def _sos_hp(fc, fs, order=4):
    return butter(order, fc/(fs*0.5), btype='highpass', output='sos')

def _sos_lp(fc, fs, order=4):
    return butter(order, fc/(fs*0.5), btype='lowpass', output='sos')

def _resample_poly_to_fs(x, fs_in, fs_target, max_den=1000):
    if fs_in == fs_target:
        return x, fs_in
    frac = Fraction(fs_target, fs_in).limit_denominator(max_den)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(x, up, down)
    return y, fs_target

def preprocess_wav_for_dfa(path, mode='wave', fs_target=2000,  # para onda cruda
                           env_lp_hz=10.0, fs_env=100):       # para envolvente
    """
    mode='wave' -> usa la onda cruda (alto muestreo, p.ej. 2 kHz), HP 20 Hz.
    mode='envelope' -> usa envolvente de amplitud (LP ~10 Hz), resample a 100 Hz.
    Devuelve: serie_preprocesada, fs_out
    """
    fs, x = wavfile.read(path)
    x = _to_float_mono(x)

    # Quitar NaN/Inf por si acaso
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == 'wave':
        # 1) eliminar DC/deriva (HP ~20 Hz suele bastar para la onda cruda)
        sos_hp = _sos_hp(20.0, fs, order=4)
        x = sosfiltfilt(sos_hp, x)

        # 2) resample a una Fs más manejable para DFA (p.ej. 2 kHz)
        x, fs_out = _resample_poly_to_fs(x, fs, fs_target)

        return x, fs_out

    elif mode == 'envelope':
        # 1) envolvente de Hilbert
        env = np.abs(hilbert(x))

        # 2) quitar DC y suavizar dinámica lenta con LP (p.ej. 10 Hz)
        env = env - np.mean(env)
        sos_lp = _sos_lp(env_lp_hz, fs, order=4)
        env = sosfiltfilt(sos_lp, env)

        # 3) resample a ~100 Hz para DFA de dinámica lenta
        env, fs_out = _resample_poly_to_fs(env, fs, fs_env)

        return env, fs

    else:
        return x, fs

def run_dfa_from_file(array,binlog=False, method='DFA', Result='H', graph=True):
    time_series = array
    if Result == 'H':
        grado_polinomio = 1
        xi = False
    elif Result == 'xi':
        grado_polinomio = 4
        xi = True
    else:
        raise ValueError("Opción no válida. Use 'xi' o 'H'.")
    xi_index = main(time_series, method=method, binlog=binlog, grado_polinomio=grado_polinomio, N_surrogates=100, graficar=graph, xi = xi)
    if Result == 'H':
        print(f'Exponente(s) de Hurst: {xi_index}')
        return xi_index
        # beta, r2, slope, intercept, (f, Pxx) = plot_psd_beta(time_series, fs_used, fmin=None, fmax=None, bins_per_decade=12)
        # print(f'Exponente Beta del PSD: {beta}, R²: {r2}')
    if Result == 'xi':
        print(f'Índice de no linealidad: {xi_index}')
        return xi_index


if __name__ == "__main__":
    # Ejemplo de uso
    archivo = input("Ingrese el tipo de archivo (.csv, .npy, .wav): ")
    if archivo.endswith('.csv'):
        data = pd.read_csv(r"new_data/DFA_FILES/"+ archivo)
        time_series = data.iloc[:, 0].values  # Asumiendo que la serie está en la primera columna
    elif archivo.endswith('.npy'):
        time_series = np.load(archivo)
    elif archivo.endswith('.wav'):
        print("Elige modo según lo que quieras analizar:")
        print("'wave': correlaciones en la onda (timbre/armónicos; requiere más Fs)")
        print("'envelope': correlaciones en la dinámica lenta (intensidad/expresividad)")
        mode = input("modo: ")          # o 'envelope'
        time_series, fs_used = preprocess_wav_for_dfa(r"new_data/DFA_FILES/"+archivo, mode=mode)
    else:
        raise ValueError("Formato de archivo no soportado. Use .csv, .npy o .wav")

    binlog = input("¿Usar binlog? (True/False): ")
    method = input("Método (DFA/MDFA): ")
    Result = input("Deseas calcular xi o Hurst? (xi/H): ")
    if Result == 'H':
        grado_polinomio = 1
        xi = False
    elif Result == 'xi':
        grado_polinomio = 4
        xi = True
    else:
        raise ValueError("Opción no válida. Use 'xi' o 'H'.")
    graph = input("¿Graficar? (True/False): ")
    xi_index = main(time_series, method=method, binlog=binlog, grado_polinomio=grado_polinomio, N_surrogates=100, graficar=graph, xi = xi)
    if Result == 'H':
        print(f'Exponente(s) de Hurst: {xi_index}')
        # beta, r2, slope, intercept, (f, Pxx) = plot_psd_beta(time_series, fs_used, fmin=None, fmax=None, bins_per_decade=12)
        # print(f'Exponente Beta del PSD: {beta}, R²: {r2}')
    if Result == 'xi':
        print(f'Índice de no linealidad: {xi_index}')
