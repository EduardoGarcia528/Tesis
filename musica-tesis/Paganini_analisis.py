import numpy as np
import matplotlib.pyplot as plt
from DFA_direct import iaaft

def autocorrelation(x, maxlag):
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac)//2:]
    return ac[:maxlag] / ac[0]

def mutual_information(x, maxlag, bins=16):
    mi = np.zeros(maxlag)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)

    for tau in range(1, maxlag+1):
        x1 = x[:-tau]
        x2 = x[tau:]
        H, _, _ = np.histogram2d(x1, x2, bins=bins)
        Pxy = H / np.sum(H)
        Px = np.sum(Pxy, axis=1)
        Py = np.sum(Pxy, axis=0)

        nz = Pxy > 0
        mi[tau-1] = np.sum(Pxy[nz] * np.log(Pxy[nz] / (Px[:,None]*Py[None,:])[nz]))

    return mi

# ---- Main function ----
def analyze_melody(melody, maxlag=50, ns_iaaft=10):
    melody = np.asarray(melody)

    # original
    ac_orig = autocorrelation(melody, maxlag)
    mi_orig = mutual_information(melody, maxlag)

    # shuffled
    shuff = np.random.permutation(melody)
    ac_shuff = autocorrelation(shuff, maxlag)
    mi_shuff = mutual_information(shuff, maxlag)

    # IAAFT (average over surrogates)
    xs = iaaft(melody, ns_iaaft, verbose=False)
    ac_iaaft = np.mean([autocorrelation(x, maxlag) for x in xs], axis=0)
    mi_iaaft = np.mean([mutual_information(x, maxlag) for x in xs], axis=0)

    # ---- Plots ----
    lags = np.arange(maxlag)

    plt.figure()
    plt.plot(lags, ac_orig, label="Original")
    plt.plot(lags, ac_shuff, label="Shuffle")
    plt.plot(lags, ac_iaaft, label="IAAFT")
    plt.xlabel("Time delay")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(lags+1, mi_orig, label="Original")
    plt.plot(lags+1, mi_shuff, label="Shuffle")
    plt.plot(lags+1, mi_iaaft, label="IAAFT")
    plt.xlabel("Time delay")
    plt.ylabel("Mutual Information")
    plt.legend()
    plt.show()


melodia = np.load(r"new_data/24.npy", allow_pickle=True)
analyze_melody(np.diff(melodia), maxlag=40, ns_iaaft=10)

