import pandas as pd
import os
import time
from multiprocessing import Pool, freeze_support, cpu_count
from scipy.interpolate import PchipInterpolator
import numpy as np
import copy
from scipy.stats import spearmanr
import piecewise_regression
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu

#dataframe datos de compositores 
def extraer_dataset_musica():

    datos_composers = {}
    # carpeta = r'D:\La formula secreta de la cangreburger\Documentos\uaem\octavo semestre\Tesis\Sequences\labels'
    carpeta = r'D:\La formula secreta de la cangreburger\Documentos\uaem\octavo semestre\Tesis\musica-tesis\data\Sequences\labels'
    archivos_en_carpeta = os.listdir(carpeta)
    index0 = 0
    indice = 0

    for archivo in archivos_en_carpeta:
        ruta_completa = os.path.join(carpeta, archivo)
        serie = pd.read_csv(ruta_completa, header = None)
        composer = archivo.split('-')[1].capitalize() # nombre compositor
        datos_composers[composer] = {} #genero bibio para composer
        datos_composers[composer]['Birth_year'] = archivo.split('-')[0] #año de nacimiento
        index1 = serie.iloc[0, 0].split('\t')[0] #el # del primer serie del composer
        index2 = int(serie.iloc[len(serie)-3, 0].split('\t')[0]) - index0 # # Piezas
        index0 = index2 + index0 # numero total de piezas anteriores
        datos_composers[composer]['# Piezas'] = index2 # Piezas
        datos_composers[composer]['Indice'] = indice
        indice += 1

    composers = {}
    M = 0
    # carpeta = r'D:\La formula secreta de la cangreburger\Documentos\uaem\octavo semestre\Tesis\Sequences\Series'
    carpeta = r'D:\La formula secreta de la cangreburger\Documentos\uaem\octavo semestre\Tesis\musica-tesis\data\Sequences\Series'
    archivos_en_carpeta = os.listdir(carpeta)

    for archivo in archivos_en_carpeta:
        ruta_completa = os.path.join(carpeta, archivo)
        serie = pd.read_csv(ruta_completa)
        # escoge una serie
        composer = archivo.split('-')[1].capitalize() # nombre compositor
        composers[composer] = {}

        for pieza in range( datos_composers[composer]['# Piezas'] ):
            N = serie.iloc[0, 0].split('\t')[1] # # de elementos por pieza
            M = int(N) + M
            index_n1 = 0 
            index_n2 = int(N)+2 
            serie_n = serie[index_n1 + 2:index_n2].reset_index(drop=True) # resetear index
            serie = serie[index_n2 +1:] # recortar serie Original
            serie_n.index += 1 # que index empiece desde 1
            num_serie_T = serie.columns[0]  # numero de serie de todo el dataset
            num_serie = pieza + 1
            composers[composer]['Serie_'+str(num_serie)] = serie_n.squeeze().to_numpy().astype(float) # agregamos pieza al dicc composer con key como # serie

    ###
    ###

    composers_depurado = copy.deepcopy(composers)
    datos_composers_depurado = copy.deepcopy(datos_composers)

    for i,composer in enumerate(composers.keys()):
        d = 0
        for pieza in composers[composer].keys():
            if len(composers[composer][pieza])//2 < 400:
                del composers_depurado[composer][pieza]
                d = d + 1
        datos_composers_depurado[composer]['# Piezas'] = datos_composers[composer]['# Piezas'] - d


    # 40 promedio de numero de piezas por compositor
    composers_depurado_v2 = copy.deepcopy(composers_depurado)
    composers_depurado_v2_keychange = copy.deepcopy(composers_depurado_v2)
    datos_composers_depurado_v2 = copy.deepcopy(datos_composers_depurado)

    for composer in composers.keys():
        if datos_composers_depurado[composer]['# Piezas'] < 30:
            del composers_depurado_v2[composer]
            del datos_composers_depurado_v2[composer]
        
    for i,composer in enumerate(composers_depurado_v2.keys()):
        datos_composers_depurado_v2[composer]['Indice'] = i 

    for composer in composers_depurado_v2.keys():
        for i,serie in enumerate(composers_depurado_v2[composer].keys()):
            composers_depurado_v2_keychange[composer]['Serie_' + str(i+1)] = composers_depurado_v2_keychange[composer].pop(serie)

    print(" # de compositores restantes: ", len(composers_depurado_v2))

    return composers_depurado_v2, datos_composers_depurado_v2


def DFA(time_series, binlog):

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
        log_bins = np.logspace(np.log10(min(n)), np.log10(max(n)), num=15)
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


def main(time_series, method, binlog=False, grado_polinomio=4, N_surrogates=100, graficar=False):
    H = []

    # CALCULAR MDFA O DFA
    if method == 'MDFA':
        n, F = mdfa(time_series,binlog=binlog)
    elif method == 'DFA':
        n, F = DFA(time_series, binlog=binlog)
    else:
        raise ValueError("El método debe ser 'MDFA' o 'DFA'")

    # GENERAR SURROGADOS
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
        return np.nan

    for value in pw_estimates:
        if 'alpha' in value:
            H.append(pw_estimates[value]['estimate'])

    if graficar:
        pw_fit.plot_data(color="red", s=1, label='log(F(s))')
        pw_fit.plot_breakpoints()

    lower_bound = np.min(flucts_surrogates, axis=0)
    upper_bound = np.max(flucts_surrogates, axis=0)
    if graficar:
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

        derivadas_surr = np.zeros([N_surrogates, len(np.log10(n_surr))])
        for surr_index in range(N_surrogates):
            _, coef_surr = ajuste_polinomial(np.log10(n_surr), flucts_surrogates[surr_index], grado=grado_polinomio)
            derivadas_surr[surr_index, :] = evaluar_derivada(coef_surr, np.log10(n_surr))

    derivadas_surr_mean = np.mean(derivadas_surr, axis=0)
    derivadas_surr_std = np.std(derivadas_surr, axis=0)

    if graficar:
        plt.xlabel('log(s)', fontsize=14)
        plt.ylabel('log(F(s))', fontsize=14)
        plt.title('DFA', fontsize=14)
        plt.legend(loc=0, fontsize=7)
        plt.show()
        plt.close()

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
    xi = np.sum(np.abs(derivadas - derivadas_surr_mean) / derivadas_surr_std)
    xi_index = xi / len(derivadas)
    return xi_index



def procesar_serie(args):
    composer, serie, subject, birth_year = args
    index = int(serie.split('_')[1])
    valor = main(subject, 'MDFA', binlog=False)
    return (index, valor)

if __name__ == '__main__':
    start = time.time()
    freeze_support()  # Necesario en Windows o si haces ejecutable
    composers, datos_composers = extraer_dataset_musica()

    for composer in composers.keys():
    # composer = 'Debussy'
        birth_year = datos_composers[composer]['Birth_year']
        piezas = list(composers[composer].keys())
        subjects = [composers[composer][serie] for serie in piezas]

        # Prepara lista de argumentos
        args_list = [(composer, serie, subject, birth_year) 
                        for serie, subject in zip(piezas, subjects)]

        # Paraleliza el procesamiento de las piezas
        with Pool(processes=cpu_count()) as pool:
            resultados = pool.map(procesar_serie, args_list)

        # Guarda los resultados como matriz
        xi_index = np.array(resultados)
        path = 'D:/La formula secreta de la cangreburger/Documentos/uaem/octavo semestre/Tesis/DFA/xi_index_mdfa'
        os.makedirs('xi_index_mdfa', exist_ok=True)
        np.save(f'{path}/{birth_year}_{composer}_xi.npy', xi_index)
        end = time.time()
        print(f"Tiempo transcurrido: {(end - start)/60} min")
