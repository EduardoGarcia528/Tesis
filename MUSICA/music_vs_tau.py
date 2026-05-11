import numpy as np
import mi_libreria as ml
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

for num in range(1,24):
    melody = np.load(f"melodies/{str(num)}.npy")

    C,g = ml.gamma_index_rank_ties(melody,6,mu=2)

    H = []
    H_null = []
    tau = range(1,7)
    for t in range(1,7):
        H.append(ml.indice_H(melody, tau=t))
        H_null.append(ml.indice_H(melody,tau=t, null = "shuffle"))

    print("spearman: ", spearmanr(1-g,H), "pearson: ", pearsonr(1-g,H))
    plt.plot(tau,H, label='J_h',color='red')
    plt.plot(tau, H_null, label='J shuffle')
    plt.plot(tau, 1-g, label='gamma')
    plt.legend()
    plt.title(str(num))
    plt.show()
