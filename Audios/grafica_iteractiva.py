from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

def interpolador(subject, method, size):
    # data = np.array([int(line.strip()) for line in subject.to_numpy()])  # Si lo obtienes de un DataFrame
    data = subject
    x = np.arange(len(data))
    
    # Crear 'size' puntos equidistantes
    x_new = np.linspace(0, len(data) - 1, size*(len(data)-1) + len(data))
    
    if method == 'lineal':
        data_interp = np.interp(x_new, x, data)
    elif method == 'herm':
        interpolator = PchipInterpolator(x, data)
        data_interp = interpolator(x_new)
    
    return x_new, data_interp

J_null_matrix = np.load('J_null_continuo.npy')
x = J_null_matrix[0,:]
J_min = J_null_matrix[1,:]
J_mean = J_null_matrix[2,:]
J_std = J_null_matrix[3,:]


# Tu código de datos aquí
# Graficar barras de error
step = 100
steps = np.arange(0, len(x), step)
plt.errorbar(x=x[steps], y=J_mean[steps], yerr=J_std[steps], fmt='none', ecolor='black', capsize=2, label='std. dev.')

# Graficar curva promedio
plt.plot(x, J_mean, color='red', label='Promedio')

# Graficar curva J mínima
plt.plot(x, J_min, 'green', label='J mínima')

# Definir puntos personalizados con color y label
puntos_x = [260, 778, 506, 1516,5395, 16183,69,205,165,493,197,589]  # Ejemplo de coordenadas x
puntos_y = [0.94785, 0.86114, 0.97517, 0.94704, 0.9784986342991956,0.9522797435374722,0.8472476329920281,
            0.8552133766900794,0.9240593229803462, 0.8943252815067188,0.9951701389408945,0.9961072339787621]  #y
print(len(puntos_x),len(puntos_y))
puntos_labels = ['Detroit become human', 'Detroit become human', 'Webern_-_Variationen_Op._27',
                 'Webern_-_Variationen_Op._27','Concerto No. 2 in E flat major','Concerto No. 2 in E flat major',
                 'Elf_Kurze_Stcke_No._4','Elf_Kurze_Stcke_No._4','Op.4, No.3','Op.4, No.3',
                 'Entflieht_auf_leichten_Kaehnen__Op_2','Entflieht_auf_leichten_Kaehnen__Op_2']
puntos_colors = ['blue', 'blue', 'orange','orange','red','red','black','black','green','green','gray','gray']

# Agregar los puntos a la gráfica
sc = plt.scatter(puntos_x, puntos_y, c=puntos_colors, label='Puntos etiquetados', zorder=3)

# Crear cursores interactivos con labels
cursor = mplcursors.cursor(sc, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(puntos_labels[sel.index]))

# Configurar ejes y leyenda
plt.xlabel('N/2 tamaño de la serie')
plt.ylabel('J')
plt.legend()
plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)

# Mostrar la gráfica
plt.show()