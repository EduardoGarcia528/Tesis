import numpy as np
import matplotlib.pyplot as plt
import mi_libreria as ml
x1 = np.random.uniform(0,1,100_000)
x2 = np.random.uniform(0,1,100_000)
x3 = np.random.uniform(0,1,100_000)


t,x,y,z = ml.rossler_system(c=12)
ruido=[]
J=[]
print(np.std(x))
for var in range(1,40):
    r = np.random.normal(0,var,100_001)
    r2 = np.random.normal(0,var,100_001)
    X= x+r
    Y=y+r2
    SNR= np.std(x)/np.std(r)
    J.append(ml.indice_J(X,y))
    ruido.append(SNR)

plt.plot(ruido,J)
plt.ylim(0,1)
plt.show()
