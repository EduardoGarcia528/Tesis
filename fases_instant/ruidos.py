import mi_libreria as ml
import numpy as np
import matplotlib.pyplot as plt

colors = ["white","brown","pink","blue", "violet"]
for c in colors:
    A  = ml.colored_noise(3000,c)
    S = []
    for t in range(1,10):
        S.append(ml.entropia_J(A,None,tau=t))
    plt.plot(range(1,10),S, label = c)
plt.legend()
plt.show()