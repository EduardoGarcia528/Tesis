from multiprocessing import Pool, freeze_support
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import pandas as pd
import os
import itertools
from scipy.stats import pearsonr

def J_univariante(X, tau, corte):
    def distancia(p1, p2):
        return np.linalg.norm(np.array(p2) - np.array(p1))
    X = np.array(X)
    x1 = X[tau:]
    y1 = X[:-tau]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    vectores = []
    for i in range(len(ff1) - 1):
        p1 = [ff1[i], ff2[i]]
        p2 = [ff1[i + 1], ff2[i + 1]]
        cuadrante = [
            [p2[0] - p1[0], p2[1] - p1[1]],
            [p2[0] - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
        ]
        distancias = np.array([distancia(p1, c) for c in cuadrante])
        p2 = cuadrante[np.argmin(distancias)]
        vectores.append([p2[0] - p1[0], p2[1] - p1[1]])

    vectores = np.array(vectores)
    norms = np.linalg.norm(vectores, axis=1, keepdims=True)
    v_norm = np.where(norms == 0, vectores, vectores / norms)
    
    angulos = np.arccos(np.clip(np.einsum('ij,ij->i', v_norm[:-1], v_norm[1:]), -1.0, 1.0))
    cruces = np.cross(v_norm[:-1], v_norm[1:])
    angulos = np.where(cruces > 0, np.pi - angulos, angulos)
    angulos = np.where((cruces == 0) & (angulos < 0), np.pi, angulos)
    angulos = np.where(cruces < 0, angulos + np.pi, angulos)

    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    
    return J

def J_bivariante(X, Y, corte):
    def distancia(p1, p2):
        return np.linalg.norm(np.array(p2) - np.array(p1))
    X = np.array(X)
    x1 = X[:]
    y1 = Y[:]
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(y1))

    vectores = []
    for i in range(len(ff1) - 1):
        p1 = [ff1[i], ff2[i]]
        p2 = [ff1[i + 1], ff2[i + 1]]
        cuadrante = [
            [p2[0] - p1[0], p2[1] - p1[1]],
            [p2[0] - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] + 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - 2 * np.pi - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] - p1[1]],
            [p2[0] - 2 * np.pi - p1[0], p2[1] + 2 * np.pi - p1[1]],
        ]
        distancias = np.array([distancia(p1, c) for c in cuadrante])
        p2 = cuadrante[np.argmin(distancias)]
        vectores.append([p2[0] - p1[0], p2[1] - p1[1]])

    vectores = np.array(vectores)
    norms = np.linalg.norm(vectores, axis=1, keepdims=True)
    v_norm = np.where(norms == 0, vectores, vectores / norms)
    
    angulos = np.arccos(np.clip(np.einsum('ij,ij->i', v_norm[:-1], v_norm[1:]), -1.0, 1.0))
    cruces = np.cross(v_norm[:-1], v_norm[1:])
    angulos = np.where(cruces > 0, np.pi - angulos, angulos)
    angulos = np.where((cruces == 0) & (angulos < 0), np.pi, angulos)
    angulos = np.where(cruces < 0, angulos + np.pi, angulos)

    e = np.exp(angulos * 1j)
    e1 = np.sum(e) / len(angulos)
    J = 1.0 - np.abs(e1.real)
    
    return J

def interpolador(subject, method, size):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    data = subject
    x = np.arange(len(data))
    
    # Crear 'size' puntos equidistantes
    x_new = np.linspace(0, len(data) - 1, size*(len(data)-1) + len(data))
    
    if method == 'lineal':
        data_interp = np.interp(x_new, x, data)
    elif method == 'herm':
        interpolator = PchipInterpolator(x, data)
        data_interp = interpolator(x_new)
    
    return x_new, data_interp

def calcular_gamma_opt(data, gamma_index, MS):
    N = len(data)
    sd = np.std(data, ddof=1)
    eps = sd / MS
    maxdat = np.max(data)
    data = np.concatenate(([0.0], data, [maxdat + 100 * eps]))  # data[0], data[N+1]

    Ci = [0] * (gamma_index + 2)  # Necesitamos Ci[i-1], Ci[i], Ci[i+1]

    for j in range(1, N + 1):
        for i in range(1, j):
            k = 0
            while k <= gamma_index + 1 and abs(data[i + k] - data[j + k]) <= eps:
                if k in (gamma_index - 1, gamma_index, gamma_index + 1):
                    Ci[k] += 1
                k += 1

    norm = 2.0 / (N * (N - 1))
    C = [1.0] + [0.0] * (gamma_index + 1)
    for k in (gamma_index - 1, gamma_index, gamma_index + 1):
        if k >= 1:
            C[k] = Ci[k] * norm

    denominator = C[gamma_index - 1] * C[gamma_index + 1]
    gamma = 0.0
    if denominator != 0:
        gamma = 1.0 - (C[gamma_index] ** 2) / denominator

    return gamma

# ==== PARÁMETROS CONFIGURABLES ====
usar_archivos_guardados = False
method = 'lineal'
size = 2
MS = 1.0
indice_gamma = 1

nombres = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J']
columnas_validas = ["PPn", "RRn", "TTn", "PR", "RT", "PT", "TPn"]
ruta_datos = "datos"
os.makedirs(ruta_datos, exist_ok=True)

# ==== FUNCIONES AUXILIARES ====
# Las funciones J_univariante, J_bivariante, interpolador y calcular_gamma_opt se asumen cargadas


def procesar_hoja(letra):
    archivo_J = f"{ruta_datos}/J_{letra}_method-{method}_size-{size}.npy"
    archivo_gamma = f"{ruta_datos}/gamma_{letra}_gamma-{indice_gamma}_MS-{MS}.npy"
    J_min = np.load('J_minus_continuo.npy')
    df = pd.read_csv(f"s{letra}.csv")

    if usar_archivos_guardados and os.path.exists(archivo_J) and os.path.exists(archivo_gamma):
        J_uni_bi = np.load(archivo_J, allow_pickle=True).item()
        gamma_vals = np.load(archivo_gamma, allow_pickle=True).item()
    else:
        columnas = [col for col in df.columns if col in columnas_validas]
        J_uni_bi = {}
        for col in columnas:
            serie = df[col].dropna().values
            if size > 0:
                _, serie = interpolador(serie, method, size)
            J_uni_bi[col] = J_univariante(serie, tau=1, corte=False)

        for col1, col2 in itertools.combinations(columnas, 2):
            serie1 = df[col1].dropna().values
            serie2 = df[col2].dropna().values
            if size > 0:
                _, serie1 = interpolador(serie1, method, size)
                _, serie2 = interpolador(serie2, method, size)
            J_uni_bi[f"{col1}-{col2}"] = J_bivariante(serie1, serie2, corte=False)

        gamma_vals = {}
        for col in columnas:
            serie = df[col].dropna().values
            gamma_vals[col] = calcular_gamma_opt(serie, indice_gamma, MS)

        np.save(archivo_J, J_uni_bi)
        np.save(archivo_gamma, gamma_vals)

    etiquetas = list(J_uni_bi.keys())
    valores = list(J_uni_bi.values())
    colores = ['skyblue' if '-' not in e else 'salmon' for e in etiquetas]

    numeros = []
    for etiqueta in etiquetas:
        if '-' in etiqueta:
            col1, col2 = etiqueta.split('-')
            N = min(len(df[col1].dropna()), len(df[col2].dropna()))
        else:
            N = len(df[etiqueta].dropna())
        mitad = N // 2
        idx = np.where(J_min[0, :] == mitad)[0]
        numeros.append(J_min[1, idx[0]] if idx.size > 0 else np.nan)

    x = np.arange(len(etiquetas))
    ymin = min(np.nanmin(valores), np.nanmin(numeros))
    plt.figure(figsize=(14, 6))
    plt.bar(x, valores, color=colores)
    plt.plot(x, numeros, 'ko', label='J mínimo continuo')
    plt.xticks(x, etiquetas, rotation=90)
    plt.ylim([ymin, 1.0])
    plt.title(f"Índice J - Hoja {letra}")
    plt.ylabel("Valor de J")
    plt.legend(handles=[
        plt.Line2D([0], [0], color='skyblue', lw=4, label='J_univariante'),
        plt.Line2D([0], [0], color='salmon', lw=4, label='J_bivariante'),
        plt.Line2D([0], [0], marker='o', color='k', lw=0, label='J mínimo continuo')
    ])
    plt.tight_layout()
    plt.savefig(f"J_indices_{letra}.png")
    plt.close()

    etiquetas_gamma = list(gamma_vals.keys())
    valores_gamma = list(gamma_vals.values())
    plt.figure(figsize=(10, 5))
    plt.bar(etiquetas_gamma, valores_gamma, color='steelblue')
    plt.title(f"Índice gamma[{indice_gamma}] - Hoja {letra}")
    plt.ylabel(f"gamma[{indice_gamma}]")
    plt.tight_layout()
    plt.savefig(f"gamma_{indice_gamma}_{letra}.png")
    plt.close()

    return letra, J_uni_bi, gamma_vals


if __name__ == "__main__":
    freeze_support()
    with Pool(processes=len(nombres)) as pool:
        resultados = pool.map(procesar_hoja, nombres)

    resultados_J = {letra: J for letra, J, _ in resultados}
    resultados_gamma = {letra: G for letra, _, G in resultados}

    corrs_col = {}
    for col in columnas_validas:
        Js = []
        gammas = []
        for letra in nombres:
            j_dict = resultados_J[letra]
            g_dict = resultados_gamma[letra]
            if col in j_dict and col in g_dict:
                Js.append(j_dict[col])
                gammas.append(g_dict[col])
        if len(Js) >= 2:
            corrs_col[col] = pearsonr(Js, gammas)[0]

    plt.figure(figsize=(10, 5))
    plt.bar(corrs_col.keys(), corrs_col.values(), color='mediumseagreen')
    plt.title(f"Correlación Pearson por columna\nmethod={method}, size={size}, gamma={indice_gamma}, MS={MS}")
    plt.ylabel("Coeficiente de correlación")
    plt.tight_layout()
    plt.savefig("correlacion_por_columna.png")
    plt.close()

    corrs_paciente = {}
    for letra in nombres:
        j_dict = resultados_J[letra]
        g_dict = resultados_gamma[letra]
        comunes = [col for col in columnas_validas if col in j_dict and col in g_dict]
        if len(comunes) >= 2:
            Js = [j_dict[col] for col in comunes]
            gammas = [g_dict[col] for col in comunes]
            corrs_paciente[letra] = pearsonr(Js, gammas)[0]

    plt.figure(figsize=(10, 5))
    plt.bar(corrs_paciente.keys(), corrs_paciente.values(), color='tomato')
    plt.title(f"Correlación Pearson por paciente\nmethod={method}, size={size}, gamma={indice_gamma}, MS={MS}")
    plt.ylabel("Coeficiente de correlación")
    plt.tight_layout()
    plt.savefig("correlacion_por_paciente.png")
    plt.close()

    for letra in nombres:
        print(f"[✓] Procesado: Hoja {letra}")
