import numpy as np

def colored_noise(beta, N, random_state=None):
    """
    Genera una serie real de longitud N con espectro de potencias ~ f^(-beta).
    La serie sale con media 0 y varianza 1.
    """
    rng = np.random.default_rng(random_state)
    spectrum = np.zeros(N, dtype=complex)

    # Número de frecuencias positivas
    n_pos = N // 2
    freqs = np.arange(1, n_pos + 1)

    # Amplitud ~ f^(-beta/2)
    amplitude = 1.0 / (freqs ** (beta / 2.0))
    phases = rng.uniform(0, 2 * np.pi, size=n_pos)
    pos = amplitude * np.exp(1j * phases)

    # Colocamos las frecuencias positivas
    spectrum[1:n_pos + 1] = pos
    # Simetría hermítica para señal real
    spectrum[-n_pos:] = np.conj(pos[::-1])

    # Nyquist real si N es par
    if N % 2 == 0:
        spectrum[N // 2] = spectrum[N // 2].real + 0j

    x = np.fft.ifft(spectrum).real
    x -= x.mean()
    x /= x.std()
    return x

def map_to_vocab(signal, vocab):
    """
    Mapea una serie real 1D a notas en 'vocab' (array 1D de enteros MIDI).
    Usa los rangos (ranks) para obtener una distribución aproximadamente uniforme
    sobre el vocabulario.
    """
    signal = np.asarray(signal)
    vocab = np.asarray(vocab)

    N = len(signal)
    # Ranks: 0,1,...,N-1 según el orden de la señal
    order = np.argsort(signal)
    ranks = np.empty(N, dtype=float)
    ranks[order] = np.arange(N, dtype=float)

    u = ranks / (N - 1)  # en [0,1]
    idx = np.round(u * (len(vocab) - 1)).astype(int)
    return vocab[idx]

def generate_midi_melody(beta, N, vocab, random_state=None):
    """
    Genera una melodía MIDI (array de enteros) de longitud N:
    - Con espectro de potencias ~ f^(-beta)
    - Donde cada nota pertenece al vocabulario 'vocab'
    """
    x = colored_noise(beta, N, random_state=random_state)
    melody = map_to_vocab(x, vocab)
    return melody

def power_spectrum(x, fs=1.0):
    N = len(x)
    X = np.fft.rfft(x)
    psd = np.abs(X)**2 / N
    freqs = np.fft.rfftfreq(N, 1/fs)
    return freqs[1:], psd[1:]  # quitamos f=0


import numpy as np
import matplotlib.pyplot as plt
from funciones import remove_consecutive_duplicates, permutation_entropy
from DFA_direct import iaaft, run_dfa_from_file

f = np.array([57, 59, 60, 62, 64, 65, 67])  # La2, Si2, Do3, Re3, Mi3, Fa3, Sol3
vocab_do_mayor = np.concatenate((f - 12,  # una octava abajo
                                 f,
                                 f + 12,
                                 f + 12 * 2,
                                 f + 12 * 3))

N = 800  # longitud de la melodía

# Ruido browniano ~ 1/f^2 (β=2)
mel_brown = generate_midi_melody(beta=2.0, N=N,
                                 vocab=vocab_do_mayor,
                                 random_state=0)
mel_brown = remove_consecutive_duplicates(mel_brown)
# xi = run_dfa_from_file(mel_brown.astype(float) ,'False',"DFA","xi",True)
print(len(mel_brown))
# mel_brown = iaaft(mel_brown, 1)[0,:]
np.save('mel_brown.npy',mel_brown)
plt.plot(mel_brown, marker = '.')
plt.xlim(0,100)
plt.show()


freqs, psd = power_spectrum(mel_brown)

mask = (freqs > 0) & (psd > 0)
freqs_fit = freqs[mask]
psd_fit = psd[mask]

# Ajuste lineal en escala log-log: log10(psd) = m * log10(freqs) + b
x = np.log10(freqs_fit)
y = np.log10(psd_fit)
m, b = np.polyfit(x, y, 1) 

y_fit = m * x + b
psd_model = 10**y_fit

plt.loglog(freqs, psd, markersize=3.5,
            linewidth=1.0, color='brown' , alpha=0.9)
plt.loglog(freqs_fit, psd_model, linewidth=1.0, color='black',
            alpha=1.0)
plt.title(f"slope = {m}")
plt.show()

PEs = []
for k in range(3,7):
    PE = permutation_entropy(mel_brown, m=k, tau=1)
    PEs.append(PE)
plt.plot(range(3,7),PEs)
plt.xlabel('m')
plt.ylim(0,1)
plt.ylabel('PE')
plt.show()