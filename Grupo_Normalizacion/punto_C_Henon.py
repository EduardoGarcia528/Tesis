import numpy as np
from scipy.optimize import brentq
import math

# ----------------- parámetros que puedes ajustar -----------------
b = 0.3
delta_feigenbaum = 4.669201609
max_abs = 1e6          # límite para considerar que la órbita diverge
n_trans = 2000         # transitorio al buscar ciclo
n_sample = 4096        # iteraciones para tomar la "órbita final"
tol_root = 1e-10       # tolerancia de la raíz brentq
# -----------------------------------------------------------------

def henon_map(x, y, a, b):
    return 1 - a*x*x + y, b*x

def jacobian_at(x, y, a, b):
    # Jacobiano J = [[-2 a x, 1],
    #                [    b, 0]]
    return np.array([[-2.0*a*x, 1.0],
                     [       b, 0.0]])

def iterate_until(a, b, x0=0.1, y0=0.1, n_iter=5000, max_abs=1e6):
    x, y = x0, y0
    for _ in range(n_iter):
        x, y = henon_map(x, y, a, b)
        if abs(x) > max_abs or abs(y) > max_abs:
            return None  # diverge
    return x, y

def get_cycle_points(a, b, m, x0=0.1, y0=0.1, n_trans=2000, n_sample=4096, max_abs=1e6):
    """
    Intenta obtener los m puntos del ciclo estable de periodo m.
    Devuelve array shape (m,2) con el ciclo (en orden),
    o None si diverge o no converge.
    """
    # iterar para quitar transitorio
    x, y = x0, y0
    for _ in range(n_trans):
        x, y = henon_map(x, y, a, b)
        if abs(x) > max_abs or abs(y) > max_abs:
            return None

    # recoger una larga muestra final
    pts = []
    for _ in range(n_sample):
        x, y = henon_map(x, y, a, b)
        if abs(x) > max_abs or abs(y) > max_abs:
            return None
        pts.append((x, y))
    pts = np.array(pts)

    # intentamos reconstruir ciclo de periodo m mirando los últimos k*m puntos
    k = 8  # cuántas repeticiones del ciclo esperamos ver
    if len(pts) < k*m:
        return None
    tail = pts[-(k*m):]  # últimas k*m iteraciones
    # reshape a (k, m, 2) para promediar cada posición de ciclo
    tail = tail.reshape((k, m, 2))
    # Comprobar coherencia: las k repeticiones deben ser similares
    diffs = np.max(np.linalg.norm(tail - tail[0], axis=2))
    # si las repeticiones son consistentes -> tomar promedio sobre k repeticiones
    if diffs < 1e-5:
        cycle = np.mean(tail, axis=0)  # (m,2)
        return cycle
    # si no hay buena repetición, intentar una búsqueda por clustering simple:
    # tomar los últimos m puntos y reordenarlos por aparición aproximada
    last_m = pts[-m:]
    # comprobar si last_m es repetición tolerable
    if np.max(np.linalg.norm(last_m - last_m[0], axis=1)) < 1e3:
        # fallback: devolver last_m
        return last_m
    return None

def monodromy_det_plus_I(a, b, m, x0=0.1, y0=0.1):
    """
    Calcula det(P(a) + I) donde P es la matriz monodromía sobre el ciclo de periodo m.
    Si no se puede obtener el ciclo (diverge o no converge), devuelve None.
    """
    cycle = get_cycle_points(a, b, m, x0=x0, y0=y0, n_trans=n_trans, n_sample=n_sample, max_abs=max_abs)
    if cycle is None:
        return None
    # producto de jacobianos en orden
    P = np.eye(2)
    for (x, y) in cycle:
        J = jacobian_at(x, y, a, b)
        P = J @ P
    val = np.linalg.det(P + np.eye(2))
    return val

def find_a_for_flip(b, m, a_low, a_high):
    """
    Encuentra a tal que det(P(a)+I)=0 en [a_low, a_high] para periodo m (m=2^n).
    Usa brentq sobre la función g(a). Maneja casos de divergencia.
    """
    def g(a):
        val = monodromy_det_plus_I(a, b, m)
        if val is None:
            # si diverge, forzamos signo consistente retornando +inf o -inf según extremos
            # brentq necesita cambios de signo; aquí devolvemos +large para que la búsqueda lo evite.
            return 1e6
        return val

    # comprobar que hay cambio de signo; si no, intentar ajustar ligeramente los extremos
    ga = g(a_low); gb = g(a_high)
    if ga is None or gb is None:
        raise RuntimeError("No se pudieron evaluar extremos (divergencia). Ajusta el intervalo.")
    if ga * gb > 0:
        raise RuntimeError(f"No hay cambio de signo en g en [{a_low}, {a_high}] (g(a_low)={ga}, g(a_high)={gb}). Ajusta el intervalo.")
    # brentq
    a_root = brentq(g, a_low, a_high, xtol=tol_root, maxiter=200)
    return a_root

# ------------------ rutina principal ------------------
if __name__ == "__main__":
    # intervalos iniciales (ajusta según b)
    # para b=0.3 sabes que a1~0.3645, a2~0.9125, a3~~1.05..1.1, a4~~1.2..1.35 etc.
    # pondremos un intervalo amplio pero razonable
    a_intervals = [
        (0.30, 0.40),   # para a1 ~0.3645
        (0.8, 1.0),     # para a2 ~0.9125
        (0.95, 1.12),   # para a3
        (1.05, 1.30),   # para a4
        (1.15, 1.45),   # para a5 (si es necesario)
    ]

    a_found = []
    for n, (al, ah) in enumerate(a_intervals, start=1):
        m = 2**n
        print(f"Buscando a_{n} (periodo {m}) en [{al}, {ah}] ...")
        try:
            a_n = find_a_for_flip(b, m, al, ah)
        except Exception as e:
            print(f"  ERROR buscando a_{n}: {e}")
            break
        print(f"  -> a_{n} = {a_n:.12f}")
        a_found.append(a_n)

    if len(a_found) >= 2:
        # usar los dos últimos para extrapolar Feigenbaum
        a_infty = a_found[-1] + (a_found[-1] - a_found[-2]) / delta_feigenbaum
        print("\nResultados encontrados:")
        for i, val in enumerate(a_found, start=1):
            print(f" a_{i} = {val:.12f}")
        print(f"Estimación a_infty(b={b}) ≈ {a_infty:.12f}")
    else:
        print("No se encontraron suficientes bifurcaciones para extrapolar a_infty.")
