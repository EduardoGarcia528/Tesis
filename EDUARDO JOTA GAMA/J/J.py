import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def J_univariante(X,tau,fases):
    def distancia(p1, p2):
        return np.linalg.norm(np.array(p2)-np.array(p1))
    x1 = X[tau:]
    y1 = X[:-tau]
    print(x1)
    ff1 = np.angle(np.fft.rfft(x1))[:fases]
    ff2 = np.angle(np.fft.rfft(y1))[:fases]
    vectores = []
    for i in range(len(ff1)-1):
        p1 = [ff1[i], ff2[i]]
        p2 = [ff1[i+1], ff2[i+1]]
        cuadrante = [[p2[0]-p1[0], p2[1]-p1[1]], [p2[0]-p1[0], p2[1]+2*np.pi-p1[1]],
            [p2[0]+2*np.pi-p1[0],p2[1]+2*np.pi-p1[1]],[p2[0]+2*np.pi-p1[0],p2[1]-p1[1]],
            [p2[0]+2*np.pi-p1[0],p2[1]-2*np.pi-p1[1]],[p2[0]-p1[0],p2[1]-2*np.pi-p1[1]],
            [p2[0]-2*np.pi-p1[0],p2[1]-2*np.pi-p1[1]],[p2[0]-2*np.pi-p1[0],p2[1]-p1[1]],
            [p2[0]-2*np.pi-p1[0],p2[1]+2*np.pi-p1[1]]]
        distancia1 = [distancia(p1,c) for c in cuadrante]
        p2 = cuadrante[np.argmin(distancia1)]
        vectores.append([p2[0]-p1[0],p2[1]-p1[1]])
    angulos = []
    for i in range(len(vectores)-1):
        v1=vectores[i]
        v2=vectores[i+1]
        v1_norm=v1/np.linalg.norm(v1)
        v2_norm=v2/np.linalg.norm(v2)
        angulo=np.arccos(np.clip(np.dot(v1_norm,v2_norm),-1.0,1.0))
        cruz=v1[0]*v2[1]-v1[1]*v2[0]
        if cruz>0:
            angulo=np.pi-angulo
        if cruz==0 and angulo==0:
            angulo=angulo
        if cruz==0 and angulo<0:
            angulo=np.pi
        if cruz<0:
            angulo=angulo+np.pi
        angulos.append(angulo)
    e=[]
    for k in range(len(angulos)):
        e.append(np.exp(angulos[k]*1j))
    e1=np.sum(e)/len(angulos)
    J=1.-np.abs(e1.real)
    return J


nombres = ['A','B','C','D','E','F','H','I','J']

for i in range(len(nombres)):
    # Nombre del archivo CSV
    archivo_csv = 's' + nombres[i] +'.csv'

    # Leer el archivo
    df = pd.read_csv(archivo_csv)

    # Número de columnas en el DataFrame (menos la primera)
    num_graficas = len(df.columns) - 1

    # Configurar filas y columnas para los subplots
    filas = (num_graficas + 1) // 2  # Asegura suficientes filas para dos columnas
    fig, axes = plt.subplots(filas, 2, figsize=(12, 4 * filas))  # Ajusta el tamaño de la figura

    # Aplanar los ejes para iterar fácilmente
    axes = axes.flatten()

    # Iterar sobre las columnas (omitiendo la primera)
    for i, columna_a_graficar in enumerate(df.columns[1:]):
        axes[i].scatter(range(len(df[columna_a_graficar])), df[columna_a_graficar], marker='.', alpha=0.7)
        axes[i].set_title(f'Gráfica de {columna_a_graficar}')
        axes[i].set_xlabel('Índice')
        axes[i].set_ylabel('Valores')

    # Ocultar los subplots vacíos (si hay más subplots que columnas)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el diseño para que no se superpongan las etiquetas
    plt.tight_layout()
    plt.show()

    J_por_columna = [J_univariante(df[columna],1,len(df[columna])) for columna in df.columns[1:]]

    for i in range(len(J_por_columna)):
        print('Columna: ', df.columns[i+1],', J index: ', J_por_columna[i])


    # Número de columnas
    num_graficas = len(df.columns[1:])
    filas = (num_graficas + 1) // 2  # Número de filas (ajustado para 2 columnas)

    # Crear subplots
    fig, axes = plt.subplots(filas, 2, figsize=(11, 4 * filas))  # Ajusta el tamaño de la figura

    # Aplanar los ejes para iterar fácilmente
    axes = axes.flatten()

    # Iterar sobre las columnas y graficar
    for i, columna in enumerate(df.columns[1:]):
        fases = np.angle(np.fft.rfft(df[columna]))

        # Graficar en el subplot correspondiente
        axes[i].scatter(range(len(fases)), fases, marker='.')
        axes[i].set_title(f'Fases de {columna}')
        axes[i].set_xlabel('Frecuencia')
        axes[i].set_ylabel('Ángulo (radianes)')

    # Ocultar los ejes vacíos (si los hay)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el diseño
    plt.tight_layout()
    plt.show()

