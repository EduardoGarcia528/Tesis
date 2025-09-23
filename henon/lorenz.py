import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz_rhs(x, y, z, sigma=10.0, rho=30.0, beta=8/3):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def lorenz_euler(N=5000, dt=0.011, x0=1.0, y0=1.0, z0=1.0,
                 sigma=10.0, rho=30.0, beta=8/3):
    t = np.arange(N+1) * dt
    x = np.empty(N+1); y = np.empty(N+1); z = np.empty(N+1)
    x[0], y[0], z[0] = x0, y0, z0

    for i in range(N):
        dx, dy, dz = lorenz_rhs(x[i], y[i], z[i], sigma, rho, beta)
        x[i+1] = x[i] + dt * dx
        y[i+1] = y[i] + dt * dy
        z[i+1] = z[i] + dt * dz


    return t, x, y, z


def lorenz_rk4(N=5000, dt=0.011, x0=1.0, y0=1.0, z0=1.0,
               sigma=10.0, rho=30.0, beta=8/3):
    """
    Integración del sistema de Lorenz con Runge-Kutta 4 (paso fijo).
    Devuelve: t, x, y, z (arrays de longitud N+1).
    """
    t = np.arange(N+1) * dt
    x = np.empty(N+1); y = np.empty(N+1); z = np.empty(N+1)
    x[0], y[0], z[0] = x0, y0, z0

    for i in range(N):
        k1x, k1y, k1z = lorenz_rhs(x[i], y[i], z[i], sigma, rho, beta)

        k2x, k2y, k2z = lorenz_rhs(
            x[i] + 0.5*dt*k1x,
            y[i] + 0.5*dt*k1y,
            z[i] + 0.5*dt*k1z,
            sigma, rho, beta
        )

        k3x, k3y, k3z = lorenz_rhs(
            x[i] + 0.5*dt*k2x,
            y[i] + 0.5*dt*k2y,
            z[i] + 0.5*dt*k2z,
            sigma, rho, beta
        )

        k4x, k4y, k4z = lorenz_rhs(
            x[i] + dt*k3x,
            y[i] + dt*k3y,
            z[i] + dt*k3z,
            sigma, rho, beta
        )

        x[i+1] = x[i] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        y[i+1] = y[i] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        z[i+1] = z[i] + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)

    return t, x, y, z

def plot_3d(x, y, z, title, xlabel="X", ylabel="Y", zlabel="Z",
            line=True, scatter=False, color="b", lw=1.0, s=5, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if line:
        ax.plot(x, y, z, color=color, lw=lw)
    if scatter:
        ax.scatter(x, y, z, color=color, s=s)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.tight_layout()
    plt.show()


def crossing_indices(arr, value):
    """
    Regresa los índices i donde arr cruza el nivel 'value' entre i y i+1.
    """
    arr = np.asarray(arr)
    y = arr - value
    mask = y[:-1] * y[1:] < 0  # cambio de signo
    return np.nonzero(mask)[0]



def plot_orbit_diagram(graficar, rho_min, rho_max,num_points_per_rho):
    
    rhos = np.linspace(rho_min, rho_max, num_points_per_rho)
    orbit_ejex= []
    x_array = []

    for rho in rhos:
        print(rho)
        t_eu, x_eu, y_eu, z_eu = lorenz_rk4(N=100_000, dt=0.0001, x0=1, y0=1, z0=1,
                    sigma=10, rho=rho, beta=8/3)
                
        x_eu = x_eu[10_000:]  
        y_eu = y_eu[10_000:]
        z_eu = z_eu[10_000:]

        mask = crossing_indices(z_eu, rho - 1)
        x = x_eu[mask]
        x_array.extend(x)
        orbit_ejex.extend([rho] * len(x))
        
    #A partir de aqui, orbita continua de logistica completada
        
    lyapunov_values = []


    if graficar == True:
        fig, ax1 = plt.subplots(figsize=(10,6))#figsize=(10,6)
        ax1.plot(orbit_ejex, x_array, '.', label='Bifurcación de la órbita', alpha=1)
                            
        ax1.set_xlabel('a') #.9873
        ax1.set_ylabel('x', color='blue', rotation = 360)
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='center left', bbox_to_anchor=(0.1, 0.25), framealpha=0.5)
        ax1.set_xlim(rho_min, rho_max)

        
        major_ticks_x = np.linspace(rho_min, rho_max, 6)  # 0.0, 0.2, ..., 1.0
        minor_ticks_x = np.linspace(rho_min, rho_max, 21)  # Ticks secundarios

        major_ticks_y = np.linspace(np.min(x_array), np.max(x_array), 6)  # -1, -0.5, 0, 0.5, 1.0
        minor_ticks_y = np.linspace(np.min(x_array),np.max(x_array) , 21)  # Ticks secundarios

        # Configurar los ticks del eje X
        plt.xticks(major_ticks_x)  # Solo etiquetar los ticks principales
        plt.gca().set_xticks(minor_ticks_x, minor=True)  # Agregar ticks menores sin etiquetas

        # Configurar los ticks del eje Y
        plt.yticks(major_ticks_y)  # Solo etiquetar los ticks principales
        plt.gca().set_yticks(minor_ticks_y, minor=True)  # Agregar ticks menores sin etiquetas

        # Activar la cuadrícula
        plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.5)  # Para ticks principales
        plt.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.5)         

        if False:
            ax2 = ax1.twinx()
            ax2.axhline(y=0, color='black', linestyle='--', alpha =0.55)
            ax2.plot(rhos, lyapunov_values, 'black', label = 'λ')
            ax2.set_ylabel('λ', color='black', rotation = 360)
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='center right',framealpha=0.5)
            # ax2.set_xlim(3.571906, 4.0)
        
        fig.tight_layout()  
        plt.show()

    return rhos, lyapunov_values    


# === Ejemplo de uso ===
if __name__ == "__main__":
    t_eu, x_eu, y_eu, z_eu = lorenz_euler(N=5000, dt=0.01, x0=1, y0=1, z0=1,
                                           sigma=10, rho=30, beta=8/3)
    plot_3d(x_eu, y_eu, z_eu, title="Atractor de Lorenz (Euler)")
    plot_3d(x_eu[:-10], x_eu[5:-5], x_eu[10:], title="Atractor Reconstruido de Lorenz (Euler)", xlabel="X(n)", ylabel="X(n+1)", zlabel="X(n+2)")


    rhos, lyapunov_values = plot_orbit_diagram(graficar=True, rho_min = 0.0, rho_max = 250.0,num_points_per_rho=200)
