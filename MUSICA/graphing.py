import numpy as np
import matplotlib.pyplot as plt
import os 
import mi_libreria as ml

arch = os.listdir('melodies')
A = []
B= []
E = []

betas= []
# arr = np.load(f'melodies/2.npy')
for i in arch:
    arr = np.load(f'melodies/{i}')
    A.append(ml.modified_permutation_entropy(arr,5,1))
    surr = ml.iaaft(arr,100,0.01,maxiter=10_000_000)
    C = []
    for s in surr:
        C.append(ml.modified_permutation_entropy(s,5,1))
    D = []
    for _ in range(100):
        s = np.random.permutation(arr)
        D.append(ml.modified_permutation_entropy(s,5,1))
    E.append(np.mean(D))
    B.append(np.mean(C))
    b = ml.graficar_espectro_beta(arr, plot_fit=False)
    betas.append(b)
plt.plot(range(1,len(A)+1), A, label='mPE')
plt.plot(range(1,len(B)+1), B, label='mPE iaaft')
plt.plot(range(1,len(E)+1), E, label='mPE random')
plt.plot(range(1,len(betas)+1), betas, label='beta')
plt.legend()
plt.show()
    # plt.plot(arr, marker='o', linestyle='-', linewidth=1.0, markersize=2.5)
    # plt.title(f'Melodía {i}')
    # plt.show()