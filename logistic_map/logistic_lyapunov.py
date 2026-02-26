import numpy as np

def lyapunov_logistic(r, x0=0.5, n_trans=5000, n_iter=100000):
    """
    Calcula el exponente de Lyapunov del mapa logístico para un r dado.
    """
    x = x0
    
    # Transiente (descartar)
    for _ in range(n_trans):
        x = r * x * (1 - x)
    
    # Cálculo del exponente de Lyapunov
    lyap = 0.0
    for _ in range(n_iter):
        x = r * x * (1 - x)
        lyap += np.log(abs(r * (1 - 2*x)) + 1e-16)  # evitar log(0)
    
    return lyap / n_iter


def find_r_infty(r_min=3.568, r_max=3.572, tol=1e-10, max_iter=1000):
    """
    Encuentra r_infty (transición al caos) resolviendo λ(r)=0
    mediante bisección.
    """
    lam_min = lyapunov_logistic(r_min)
    lam_max = lyapunov_logistic(r_max)
    
    if lam_min * lam_max > 0:
        raise ValueError("El intervalo no encierra la transición (λ no cambia de signo).")
    
    for _ in range(max_iter):
        r_mid = 0.5 * (r_min + r_max)
        lam_mid = lyapunov_logistic(r_mid)
        
        if abs(lam_mid) < tol:
            return r_mid, lam_mid
        
        if lam_min * lam_mid < 0:
            r_max = r_mid
            lam_max = lam_mid
        else:
            r_min = r_mid
            lam_min = lam_mid
    
    return r_mid, lam_mid


if __name__ == "__main__":
    r_inf, lam = find_r_infty()
    print("r_infty ≈", r_inf)
    print("Lyapunov en r_infty ≈", lam)