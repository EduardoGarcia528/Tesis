import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import mi_libreria as ml

def caminata_aleatoria(N):
    x1, y1 = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)
    # ff1 = np.random.uniform(-np.pi, np.pi, N)
    # ff2 = np.random.uniform(-np.pi, np.pi, N)
    ff1 = np.fft.rfft(x1)[1:]
    ff2 = ff1[:-1]
    n = len(ff1) - 1
    vectores = np.empty((n,2)) #(n,2)
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = ml.mejor_vector(p1, p2)
    
    return vectores

def main(X):
    x1 = np.random.uniform(0, 1, X)
    # y1 = np.random.uniform(0, 1, X)
    # angulos = ml.angulos_alpha(x1,y1)
    J = ml.entropia_J(x1, None)
    # J, _ = ml.gamma_index_circular(angulos,5,3)
    return J    

def calcular_J_N(N):
    for i in range(1):
        subjects = np.zeros(2000)
        for j in range(2000):
            subjects[j] = main(X=N)
            # entropia_shannon(np.diff(angulos))

        J_N_min = np.min(subjects)
        J_N_mean = np.mean(subjects)
        J_N_std = np.std(subjects)
    print(N)
    return J_N_min, J_N_std, J_N_mean

def derivada_index(array):
    array = np.asarray(array)
    derivada = np.abs(np.diff(array))  # Calcula la diferencia entre puntos subsecuentes
    return derivada

if __name__ == '__main__':
    mp.freeze_support()

    N0 = np.arange(10, 20, 1)
    N1 = np.arange(20, 100, 5)
    N2 = np.arange(100, 2000, 100)
    N3 = np.arange(2000, 20000, 2000)
    N4 = np.arange(20000, 100000, 10000)
    Ns = np.concatenate((N0, N1,N2,N3,N4)) 
  
    # Usa multiprocessing para calcular J_min en paralelo
    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados = pool.map(calcular_J_N, Ns)

    J_min, J_std, J_mean = map(np.array, zip(*resultados))

    # Ya puedes proceder como antes:
    derivadas = derivada_index(Ns)

    Js_min_interp = np.array([J_min[0]])
    Js_std_interp = np.array([J_std[0]])
    Js_mean_interp = np.array([J_mean[0]])
    for i, der in enumerate(derivadas):
        Js_min_interp = np.concatenate((
            Js_min_interp,
            ml.interpolador(J_min[i:i+2], 'lineal', der - 1)[1:]
        ))
        Js_mean_interp = np.concatenate((
            Js_mean_interp,
            ml.interpolador(J_mean[i:i+2], 'lineal', der - 1)[1:]
        ))
        Js_std_interp = np.concatenate((
            Js_std_interp,
            ml.interpolador(J_std[i:i+2], 'lineal', der - 1)[1:]
        ))

    print(len(ml.interpolador_constante(Ns)), len(Js_min_interp))

    J_null_continuo = np.vstack((ml.interpolador_constante(Ns), Js_min_interp,Js_mean_interp,Js_std_interp))

    np.save('new_data/J_null.npy' ,J_null_continuo)