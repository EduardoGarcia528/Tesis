import numpy as np
import matplotlib.pyplot as plt
from funciones import permutation_entropy

# Tu función (ya la tienes):
# from tu_modulo import permutation_entropy

def logistic_map(r, x0, n):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def colored_noise(N, beta, rng=None):
    """
    Genera ruido gaussiano coloreado con PSD ~ 1 / f^beta.
    beta = 0 (blanco), 1 (rosa), 2 (café/brown),
    -1 (azul), -2 (violeta).
    """
    rng = np.random.default_rng() if rng is None else rng

    # Frecuencias para rFFT
    freqs = np.fft.rfftfreq(N, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # evita f=0

    # Espectro blanco complejo
    real = rng.normal(size=len(freqs))
    imag = rng.normal(size=len(freqs))
    X = real + 1j * imag
    X[0] = rng.normal() + 0j  # componente DC real

    # Escalado de amplitud: si PSD ~ 1/f^beta, entonces |X| ~ 1/f^(beta/2)
    X *= freqs ** (-beta / 2.0)

    # Señal en el tiempo (real)
    x = np.fft.irfft(X, n=N)

    # Normaliza a media 0 y varianza 1
    x = (x - x.mean()) / x.std(ddof=0)
    return x

def plot_PE_vs_tau(m=5, taus=range(1, 51), N=20000, seed=42):
    rng = np.random.default_rng(seed)

    noise_specs = [
        ("Ruido blanco",   0,  "black"),
        ("Ruido café",     2,  "saddlebrown"),
        ("Ruido violeta", -2,  "violet"),
        ("Ruido rosa",     1,  "hotpink"),
        ("Ruido azul",    -1,  "dodgerblue"),
    ]

    taus = np.array(list(taus), dtype=int)

    plt.figure(figsize=(8, 5))

    for label, beta, color in noise_specs:
        x = colored_noise(N, beta, rng=rng)

        pe_vals = []
        for tau in taus:
            pe_vals.append(permutation_entropy(x, m, tau))

        plt.plot(
            taus,
            pe_vals,
            label=label,
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.7
        )

    plt.xlabel(r"$\tau$", fontsize=12)
    plt.ylabel("PE", fontsize=12)
    plt.title(f"PE vs $\\tau$ con embedding fijo $m={m}$", fontsize=13)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()

# ---- Ejemplo de uso ----
plot_PE_vs_tau(m=7, taus=range(1, 15), N=30000, seed=42)