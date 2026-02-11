import numpy as np
import matplotlib.pyplot as plt
from DFA_direct import iaaft
from funciones import permutation_entropy
from markov_music_generator import generate_markov_melody, generate_markov_k
import os

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
def analyze_melody(melody, maxlag=50, ns_iaaft=10, filename="melody_analysis"):

    # original
    ac_orig = autocorrelation(melody, maxlag)
    mi_orig = mutual_information(melody, maxlag)
    pe_orig = np.array([])
    for m in range(2,11):
        pe_orig = np.append(pe_orig, permutation_entropy(melody, m=m))

    # shuffled
    shuff = np.random.permutation(melody)
    ac_shuff = autocorrelation(shuff, maxlag)
    mi_shuff = mutual_information(shuff, maxlag)
    pe_shuff = np.array([])
    for m in range(2,11):
        pe_shuff = np.append(pe_shuff, permutation_entropy(shuff, m=m))

    # IAAFT (average over surrogates)
    xs = iaaft(melody, ns_iaaft, verbose=False)
    ac_iaaft = np.mean([autocorrelation(x, maxlag) for x in xs], axis=0)
    mi_iaaft = np.mean([mutual_information(x, maxlag) for x in xs], axis=0)
    pe_iaaft = np.array([])
    for m in range(2,11):
        pe_iaaft = np.append(pe_iaaft, np.mean([permutation_entropy(x, m=m) for x in xs]))

    # ---- Plots ----
    lags = np.arange(maxlag)

    plt.figure()
    plt.plot(lags, ac_orig, label="Original")
    plt.plot(lags, ac_shuff, label="Shuffle")
    plt.plot(lags, ac_iaaft, label="IAAFT")
    plt.xlabel("Time delay")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.title(filename)
    plt.show()

    plt.figure()
    plt.plot(lags+1, mi_orig, label="Original")
    plt.plot(lags+1, mi_shuff, label="Shuffle")
    plt.plot(lags+1, mi_iaaft, label="IAAFT")
    plt.xlabel("Time delay")
    plt.ylabel("Mutual Information")
    plt.legend()
    plt.title(filename)
    plt.show()

    plt.figure()
    plt.plot(range(2,11), pe_orig, marker='o', label="Original")
    plt.plot(range(2,11), pe_shuff, marker='s', label="Shuffle")
    plt.plot(range(2,11), pe_iaaft, marker='^', label="IAAFT")
    plt.xlabel("Permutation entropy order")
    plt.ylabel("Permutation Entropy")
    plt.legend()
    plt.title(filename)
    plt.show()

def compute_measures(melody, maxlag=50, ns_iaaft=10):
    # original
    ac_orig = autocorrelation(melody, maxlag)
    mi_orig = mutual_information(melody, maxlag)
    pe_orig = np.array([permutation_entropy(melody, m=m) for m in range(2,11)])

    # shuffled
    shuff = np.random.permutation(melody)
    ac_shuff = autocorrelation(shuff, maxlag)
    mi_shuff = mutual_information(shuff, maxlag)
    pe_shuff = np.array([permutation_entropy(shuff, m=m) for m in range(2,11)])

    # IAAFT
    xs = iaaft(melody, ns_iaaft, verbose=False)
    ac_iaaft = np.mean([autocorrelation(x, maxlag) for x in xs], axis=0)
    mi_iaaft = np.mean([mutual_information(x, maxlag) for x in xs], axis=0)
    pe_iaaft = np.array([
        np.mean([permutation_entropy(x, m=m) for x in xs])
        for m in range(2,11)
    ])

    return {
        "ac": (ac_orig, ac_shuff, ac_iaaft),
        "mi": (mi_orig, mi_shuff, mi_iaaft),
        "pe": (pe_orig, pe_shuff, pe_iaaft),
    }

def plot_comparison(meas_orig, meas_markov, filename, maxlag):
    lags = np.arange(maxlag)
    orders = np.arange(2,11)

    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharey='row')

    titles = ["Original", "Markov"]
    datasets = [meas_orig, meas_markov]

    for col, data in enumerate(datasets):
        axs[0, col].plot(lags, data["ac"][0], label="Original")
        axs[0, col].plot(lags, data["ac"][1], label="Shuffle")
        axs[0, col].plot(lags, data["ac"][2], label="IAAFT")
        axs[0, col].set_title(titles[col])

        axs[1, col].plot(lags+1, data["mi"][0])
        axs[1, col].plot(lags+1, data["mi"][1])
        axs[1, col].plot(lags+1, data["mi"][2])

        axs[2, col].plot(orders, data["pe"][0], marker='o')
        axs[2, col].plot(orders, data["pe"][1], marker='s')
        axs[2, col].plot(orders, data["pe"][2], marker='^')

    axs[0,0].set_ylabel("Autocorrelation")
    axs[1,0].set_ylabel("Mutual Information")
    axs[2,0].set_ylabel("Permutation Entropy")

    axs[2,0].set_xlabel("Order / Delay")
    axs[2,1].set_xlabel("Order / Delay")

    axs[0,0].legend(loc="upper right", fontsize=8)
    fig.suptitle(filename, fontsize=14)
    plt.tight_layout()
    plt.savefig(r"D:/La formula secreta de la cangreburger/Documentos/uaem/octavo semestre/Tesis/Audios/data/new_scores/images/"+f"{filename}_comparison_k_3.png")
    # plt.show()
    plt.close()


folder = r"D:/La formula secreta de la cangreburger/Documentos/uaem/octavo semestre/Tesis/Audios/data/new_scores/melodies"

for filename in os.listdir(folder):
    if filename.endswith(".npy"):
        path = os.path.join(folder, filename)
        melodia = np.load(path, allow_pickle=True)
        melodia = np.asarray(melodia, dtype=float)
        melodia = melodia[~np.isnan(melodia)]

        k = 3
        markov_melody = generate_markov_k(
            melodia, k=k,
            length=len(melodia),
            start_state=melodia[:k],
            end_note=None
        )

        plt.plot(melodia, label="Original")
        plt.plot(markov_melody, label="Markov k="+str(k))
        plt.xlim(0,500)
        plt.title(f"Melody Comparison: {filename} (k={k})")
        plt.legend()
        plt.savefig(r"D:/La formula secreta de la cangreburger/Documentos/uaem/octavo semestre/Tesis/Audios/data/new_scores/images/"+f"{filename}_time_series_k_3.png")
        # plt.show()
        plt.close()

        meas_orig = compute_measures(melodia, maxlag=20)
        meas_markov = compute_measures(markov_melody, maxlag=20)

        plot_comparison(meas_orig, meas_markov, filename, maxlag=20)

# for filename in os.listdir(folder):
#     if filename.endswith(".npy"):
#         path = os.path.join(folder, filename)
#         melodia = np.load(path,allow_pickle=True)
# # melodia = np.load(r"new_data/1_markov.npy", allow_pickle=True)
#         melodia = np.asarray(melodia, dtype=np.float64)
#         melodia = melodia[~np.isnan(melodia)]
#         k = 1
#         # markov_melody = generate_markov_melody(melodia, length=len(melodia), start_note=melodia[0], end_note=melodia[-1])
#         markov_melody = generate_markov_k(melodia, k = k,length=len(melodia), start_state=melodia[:k], end_note=melodia[-1])
#         analyze_melody(melodia, maxlag=20, ns_iaaft=10,filename=filename+"_original")
#         analyze_melody(markov_melody, maxlag=20, ns_iaaft=10,filename=filename+"_markov")

