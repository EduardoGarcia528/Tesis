import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from funciones import mejor_vector, calcular_angulos, permutation_entropy, entropia_shannon, interpolador, interpolador_constante, indice_J
from gamma_4 import correlation_integrals, gamma_index_jacobs, gamma

def caminata_aleatoria(N):
    ff1 = np.random.uniform(-np.pi, np.pi, N)
    ff2 = np.random.uniform(-np.pi, np.pi, N)
    n = len(ff1) - 1
    vectores = np.empty((n,2)) #(n,2)
    for i in range(n):
        p1 = (ff1[i], ff2[i])
        p2 = (ff1[i+1], ff2[i+1])
        vectores[i] = mejor_vector(p1, p2)

    return vectores

def main(X):
    # vectores = caminata_aleatoria(X)
    # angulos = calcular_angulos(vectores)
    # e = np.exp(angulos * 1j)
    # e1 = np.sum(e) / len(angulos)
    # J = 1.0 - np.abs(e1.real)
    # C, g = gamma_index_jacobs(angulos,3)
    # H = entropia_shannon(angulos,False)
    serie = np.random.uniform(0,1,X)
    serie = interpolador(serie,'lineal',5)
    J = indice_J(serie,False)
    # PE = permutation_entropy(serie,m=8,tau=1)
    return J

def calcular_J_N(N):
    J_N_min = np.ones(20)
    J_N_mean = np.ones(20)
    J_N_std = np.ones(20)
    for i in range(20):
        subjects = np.zeros(100)
        for j in range(100):
            subjects[j] = main(X=N)
            # entropia_shannon(np.diff(angulos))

        J_N_min[i] = np.min(subjects)
        J_N_mean[i] = np.mean(subjects)
        J_N_std[i] = np.std(subjects)
    J_N_min = np.mean(J_N_min)
    J_N_mean = np.mean(J_N_mean)
    J_N_std = np.mean(J_N_std)
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
    N3 = np.arange(2000, 10000, 1000)
    N4 = np.arange(10000, 100000, 10000)
    N5 = np.arange(100000, 200000, 20000)
    N6 = np.array([500_000, 1000000])
    N7 = np.array([100_000])
    # Ns = np.concatenate((N0, N1, N2, N3, N4, N5))
    Ns = np.concatenate((N0, N1,N2)) 
 
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
            interpolador(J_min[i:i+2], 'lineal', der - 1)[1:]
        ))
        Js_mean_interp = np.concatenate((
            Js_mean_interp,
            interpolador(J_mean[i:i+2], 'lineal', der - 1)[1:]
        ))
        Js_std_interp = np.concatenate((
            Js_std_interp,
            interpolador(J_std[i:i+2], 'lineal', der - 1)[1:]
        ))

    print(len(interpolador_constante(Ns)), len(Js_min_interp))

    J_null_continuo = np.vstack((interpolador_constante(Ns), Js_min_interp,Js_mean_interp,Js_std_interp))

    np.save('new_data/J_interp_null.npy' ,J_null_continuo)