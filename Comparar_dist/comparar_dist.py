import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jensenshannon#, wasserstein_distance

def compare_indices(X, Y, X_c, Y_c, n_perms=1000):
    results = {}

    # 1. Correlaciones
    results["pearson"] = pearsonr(X, Y)[0]
    results["spearman"] = spearmanr(X, Y)[0]

    # 2. Información mutua
    # discretización opcional
    bins = 20
    c_xy = np.histogram2d(X, Y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    results["mutual_info"] = mi

    # MI permutada (control)
    mi_perm = []
    for _ in range(n_perms):
        Y_perm = np.random.permutation(Y)
        c_xy_perm = np.histogram2d(X, Y_perm, bins)[0]
        mi_perm.append(mutual_info_score(None, None, contingency=c_xy_perm))
    results["mi_vs_perm"] = (mi, np.mean(mi_perm))

    # 3. Umbrales
    X_bin = X >= X_c
    Y_bin = Y <= Y_c
    table = np.zeros((2,2))
    for xb, yb in zip(X_bin, Y_bin):
        table[int(xb), int(yb)] += 1
    results["agreement_percent"] = (table[0,0] + table[1,1]) / len(X)

    # 4. Distribuciones
    # results["wasserstein"] = wasserstein_distance(X, Y)

    return results

import numpy as np
import os

composer = 'Handel'  
agreement_porcentage = np.zeros(19)
for i,composer in enumerate(['Byrd', 'Buxtehude', 'Handel', 'Scarlatti', 'Bach', 'Haydn', 'Mozart', 'Beethoven', 'Schubert', 'Chopin', 'Schumann', 'Liszt', 'Alkan', 'Brahms', 'Saint', 'Tchaikovsky', 'Dvorak', 'Faure', 'Debussy']):
    # Array X: índice de no linealidad (ejemplo con distribución normal)
    X = []
    folder_path = 'new_data/xi_index1'
    for filename in os.listdir(folder_path):
        if composer not in filename:
            continue
        arr = np.load(os.path.join(folder_path, filename))[:,1]
        X.extend(arr)


    Y = []
    folder_path = 'new_data/PEs'
    for filename in os.listdir(folder_path):
        if composer not in filename:
            continue
        arr = np.load(os.path.join(folder_path, filename))
        Y.extend(arr)
    X = np.array(X)
    Y = np.array(Y)
    X_c = 1.0
    Y_c = 1.776

    # print("X:", X[:10])  # primeros 10 valores
    print('')
    print('')
    # print("Y:", Y[:10])
    print(composer)
    results = compare_indices(X, Y, X_c, Y_c)
    agreement_porcentage[i] = results['agreement_percent']

    print("\nResultados de la comparación: ", results)
print(agreement_porcentage)
np.save(r'new_data\agreement_percent', agreement_porcentage)