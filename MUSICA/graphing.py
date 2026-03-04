import numpy as np
import matplotlib.pyplot as plt
import os 

arch = os.listdir('melodies')
for i in arch:
    arr = np.load(f'melodies/{i}')
    plt.plot(arr, marker='o', linestyle='-', linewidth=1.0, markersize=2.5)
    plt.title(f'Melodía {i}')
    plt.show()