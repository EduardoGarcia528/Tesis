import numpy as np
import matplotlib.pyplot as plt

# array  = np.load('new_data/J_null_continuo.npy')
array  = np.load('new_data/J_null.npy')
# Suponiendo que array tiene forma (4, N)
x = array[0]
# x = (x + 5*(x-1))//2
y_min = array[1]
y_mean = array[2]
y_std = array[3]

print(y_min[np.where(x==1750)[0][0]])

plt.figure(figsize=(8, 5))

# Curva mínima
plt.plot(x, y_min, color='green', label='Mínimo')
# Curva promedio
plt.plot(x, y_mean, color='red', label='Promedio')

# Barras de error (std)
plt.errorbar(x, y_mean, yerr=y_std,fmt='none', ecolor='black',elinewidth=1,capsize=2,alpha=0.1,label='Std')

plt.xlabel('longitud N')
plt.ylabel(r'$PE$',rotation=360)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
