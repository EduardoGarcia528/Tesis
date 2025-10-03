from numba import njit
import numpy as np


@njit
def lehmer_code(perm):
    """Codifica una permutación en un índice único usando Lehmer code"""
    m = len(perm)
    code = 0
    factor = 1
    for i in range(m-1, -1, -1):
        c = 0
        for j in range(i+1, m):
            if perm[j] < perm[i]:
                c += 1
        code += c * factor
        factor *= (m - i)
    return code

@njit
def stable_argsort_by_value_then_index(x):
    m = x.shape[0]
    idx = np.arange(m)
    # insertion sort por clave (valor, índice)
    for i in range(1, m):
        key = idx[i]
        j = i - 1
        while j >= 0:
            a = x[idx[j]]
            b = x[key]
            if (a > b) or (a == b and idx[j] > key):  # (valor) y luego (índice)
                idx[j+1] = idx[j]
                j -= 1
            else:
                break
        idx[j+1] = key
    return idx

@njit
def permutation_entropy(arr, m=3, tau=1):
    n = len(arr)
    if n < m:
        return np.nan
    # m!:
    fact = 1
    for k in range(2, m+1):
        fact *= k
    counts = np.zeros(fact, dtype=np.int64)
    denom = n - (m-1)*tau
    for i in range(denom):
        subseq = np.empty(m, np.float64)
        for j in range(m):
            subseq[j] = arr[i + j*tau]
        idx = stable_argsort_by_value_then_index(subseq)
        code = lehmer_code(idx)      # tu misma función
        counts[code] += 1
    # entropía normalizada (independiente de base)
    probs = counts[counts > 0] / denom
    n_prohibidos = fact - len(probs)
    H = -np.sum(probs * np.log(probs))
    Hnorm = H / np.log(fact)
    return Hnorm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import soundfile as sf  # pip install soundfile
# Si prefieres scipy: from scipy.io import wavfile

def plot_pe_from_wav(
    wav_path,
    permutation_entropy_fn,
    m=3,
    tau=1,
    win=2048,
    hop=512,
    bpm_quarter=120.0,
    sr_expected=44100,
    smooth_pe_win=None,  # p.ej. 21 para un suavizado ligero de PE
    max_points_wave=500_000,  # para mostrar la onda sin saturar
    title=None,
    corte_segundos=128
):
    """
    Genera una figura estilo Bandt & Pompe Fig.1 para un audio .wav.
    - wav_path: ruta al .wav
    - permutation_entropy_fn: tu función PE(arr, m=3, tau=1) que devuelve PE normalizada [0,1]
    - m, tau: parámetros de PE (por defecto m=3, tau=1 como pediste)
    - win: tamaño de ventana en muestras (por defecto 2048)
    - hop: desplazamiento entre ventanas en muestras (por defecto 512; puedes usar 1 bajo tu propio riesgo)
    - bpm_quarter: tempo de la negra en bpm (120.0 → 1 redonda = 2 s)
    - sr_expected: frecuencia de muestreo esperada (44.1 kHz)
    - smooth_pe_win: entero opcional para suavizado (media móvil sobre PE)
    - max_points_wave: límite de puntos para representar la onda (se diezmará si excede)
    - title: título opcional de la figura
    """
    # --------- Cargar audio ----------
    # soundfile preserva dtype y sr; si hay 2 canales, convertimos a mono por promedio
    audio, sr = sf.read(wav_path)  # audio shape: (N,) o (N, C)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if corte_segundos is not None:
        audio = audio[:sr_expected*corte_segundos]  # recortar si es muy largo
    N = len(audio)
    if sr != sr_expected:
        # No frenamos; advertimos y seguimos con sr real
        print(f"[Aviso] sr del archivo = {sr} Hz (esperado {sr_expected} Hz). Se usará {sr} Hz.")

    # Normalización (opcional y segura para ver amplitud)
    if audio.dtype.kind in ['i', 'u']:
        # Int a float [-1,1]
        max_int = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float64) / max_int
    else:
        audio = audio.astype(np.float64)

    # --------- Conversión de tiempo a "redondas" ----------
    # Negra = bpm_quarter -> periodo de negra = 60/bpm s
    # Redonda = 4 negras => 4*(60/bpm) s
    seconds_per_quarter = 60.0 / bpm_quarter
    seconds_per_whole = 4.0 * seconds_per_quarter  # 120 bpm => 2.0 s
    samples_per_whole = int(round(seconds_per_whole * sr))

    # --------- Ventaneo para PE ----------
    if win > N:
        raise ValueError("La ventana 'win' es mayor que la longitud del audio.")
    if hop < 1:
        raise ValueError("El 'hop' debe ser >= 1.")

    num_frames = 1 + (N - win) // hop
    # Índices de centro de cada ventana para ubicar en eje x
    centers = (np.arange(num_frames) * hop) + (win // 2)
    t_sec = centers / sr
    t_whole = t_sec / seconds_per_whole  # eje x en unidades de redonda

    # Calcular PE por ventanas (con tu función)
    pe_vals = np.empty(num_frames, dtype=np.float64)
    # Bucle simple (puedes paralelizar si quieres)
    for i in range(num_frames):
        start = i * hop
        end = start + win
        segment = audio[start:end]
        if i == num_frames//4 or i == num_frames//2 or i == 3*num_frames//4:
            print(i)
        pe_vals[i] = permutation_entropy_fn(segment, m=m, tau=tau)  # ya normalizada [0,1]

    # Suavizado opcional (media móvil)
    if smooth_pe_win is not None and smooth_pe_win > 1:
        k = smooth_pe_win
        kernel = np.ones(k, dtype=np.float64) / k
        pe_vals = np.convolve(pe_vals, kernel, mode='same')

    # --------- Preparar la onda para mostrar (diezmado rápido si hace falta) ----------
    wave_x_whole = np.arange(N) / sr / seconds_per_whole
    wave_y = audio

    if N > max_points_wave:
        # Decimado simple para vista (no afecta a PE)
        decim = int(np.ceil(N / max_points_wave))
        wave_x_whole = wave_x_whole[::decim]
        wave_y = wave_y[::decim]

    # --------- Graficar ----------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1]})

    # Panel 1: Onda
    ax0 = axes[0]
    ax0.plot(wave_x_whole, wave_y, linewidth=0.6)
    ax0.set_ylabel("Amplitud")
    ax0.set_title(title if title else "Audio (onda) y Entropía Permutacional (ventana deslizante)")

    # Panel 2: PE
    ax1 = axes[1]
    ax1.plot(t_whole, pe_vals, linewidth=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("PE (normalizada)")
    ax1.set_xlabel("Tiempo (unidades de redonda)")

    # --------- Grid en negras (4 por redonda) ----------
    # Ticks mayores: cada redonda (1.0 en el eje x)
    max_x = max(wave_x_whole[-1], t_whole[-1]) if len(t_whole) > 0 else wave_x_whole[-1]
    ax1.set_xlim(0, max_x)

    # Mayor: 1 redonda
    axes[-1].xaxis.set_major_locator(MultipleLocator(base=1.0))
    # Menor: 4 por unidad (negras)
    axes[-1].xaxis.set_minor_locator(MultipleLocator(base=1.0/4.0))

    for ax in axes:
        ax.grid(which='both', linestyle='--', alpha=0.35)
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=3)

    # Info en subtítulos
    info = f"PE: m={m}, tau={tau}, win={win} samp, hop={hop} samp | sr={sr} Hz | 1 redonda={seconds_per_whole:.3f}s"
    ax1.set_title(info, fontsize=9, pad=8)

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":

    # Supón que ya tienes: permutation_entropy(arr, m=3, tau=1) -> [0,1]
    fig, axes = plot_pe_from_wav(
       "partitura-copia.wav",
        permutation_entropy_fn=permutation_entropy,
        m=7,
        tau=1,
        win=44100,      # puedes probar 4096 si quieres ventanas un poco más largas
        hop=5512,       # empieza aquí; si tu máquina aguanta, baja a 128 o 64
        bpm_quarter=120.0,
        title="Sonata (PE deslizante en unidades de redonda)",
        corte_segundos=None
    )
    plt.show()

