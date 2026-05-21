import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

x,y = ml.henon_map(a=1.4, b=0.3)
# x,y = np.random.normal(size=10000), np.random.normal(size=10000)
# x,y = np.random.permutation(x), np.random.permutation(y)
print(len(x))

f1, f2 = ml.obtener_fases_instantaneas(x[1:],None,tau=0)
f11,f22 = ml.obtener_fases_instantaneas(x,None,tau=1) 
angulos = ml.angulos_alpha(x,y,tau=10)

print(ml.indice_J(x,y,tau=10))
H = ml.indice_S_eff_fast(x,y,tau=0,null="no")
x = np.random.permutation(x)
y = np.random.permutation(y)
H_null = ml.indice_S_eff_fast(x,y,tau=0,null="no")

# plt.hist(f1, bins=30, density=True)
# plt.plot(f1, f2, 'o', markersize=2)
plt.plot(f1,f11, 'o', markersize=2)
plt.title(f"Histograma de ángulos (H={H:.3f}, H_null={H_null:.3f})")
plt.xlabel("Ángulo (radianes)")
plt.show()

print(spearmanr(f1, f11[:]))
print(spearmanr(f2, f22[:]))