import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jensenshannon#, wasserstein_distance

def compare_indices(X, Y, X_c, Y_c, n_perms=1000):
    results = {}

    # 1. Correlaciones
    results["pearson"] = pearsonr(X, Y)[0]
    results["spearman"] = spearmanr(X, Y)[0]
    results["kendall"] = kendalltau(X, Y)[0]

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
    Y_bin = Y >= Y_c
    table = np.zeros((2,2))
    for xb, yb in zip(X_bin, Y_bin):
        table[int(xb), int(yb)] += 1
    chi2, p, _, _ = chi2_contingency(table)
    results["contingency_table"] = table
    results["chi2_pval"] = p
    results["agreement_percent"] = (table[0,0] + table[1,1]) / len(X)

    # 4. Distribuciones
    # results["wasserstein"] = wasserstein_distance(X, Y)
    results["js"] = jensenshannon(np.histogram(X, bins)[0], np.histogram(Y, bins)[0])

    return results

import numpy as np

# Semilla para reproducibilidad
np.random.seed(42)

# Array X: índice de no linealidad (ejemplo con distribución normal)
X = np.random.normal(loc=0.6, scale=0.15, size=100)

# Array Y: índice J (ejemplo con correlación parcial con X + ruido)
X = np.random.normal(loc=0.5, scale=0.15, size=100)

# Y: muy parecido a X, con un poco de ruido
Y = X + np.random.normal(loc=0.0, scale=0.05, size=100)
# Umbrales arbitrarios
X_c = 0.5
Y_c = 0.5

print("X:", X[:10])  # primeros 10 valores
print("Y:", Y[:10])
results = compare_indices(X, Y, X_c, Y_c)

print("\nResultados de la comparación: ", results)