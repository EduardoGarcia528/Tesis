import numpy as np
import matplotlib.pyplot as plt
from funciones import random_array, vocabulario_midi_centrado, permutation_entropy

k_values = []
PE_values = []

for k in range(5, 30):
    arr = random_array(vocabulario_midi_centrado(k=k), 1000, 0.0, 3)
    PE = permutation_entropy(arr, m=6, tau=1)
    print(f'k={k}, PE={PE}')
    k_values.append(k)
    PE_values.append(PE)
plt.plot(k_values, PE_values, marker='o')
plt.xlabel('k')
plt.ylabel('PE')
plt.title('Permutation Entropy vs k')
plt.grid()
plt.show()