import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import mi_libreria as ml

# ============================================================
# Configuración
# ============================================================

folder = Path("melodies")   
file_ids = range(1, 24)         

n_surrogates = 600
tau_values = [1]                
seed = 1234

rng = np.random.default_rng(seed)


# ============================================================
# Utilidades
# ============================================================

def load_melodies(folder, file_ids=range(1, 24)):
    melodies = {}

    for k in file_ids:
        path = folder / f"{k}.npy"
        if not path.exists():
            print(f"Archivo no encontrado: {path}")
            continue

        x = np.load(path, allow_pickle=True)
        x = np.asarray(x).squeeze()

        if x.ndim != 1:
            x = x.ravel()

        x = x.astype(float)
        x = x[np.isfinite(x)]

        melodies[k] = x

    return melodies


def safe_scalar(y):
    """
    Convierte una salida escalar, np.float, array escalar, etc. a float.
    """
    y = np.asarray(y).squeeze()
    if y.size != 1:
        raise ValueError(f"La medida no regresó un escalar. Shape: {y.shape}")
    return float(y)


def safe_eval(func, x):
    try:
        return safe_scalar(func(x))
    except Exception as e:
        print(f"Error en medida: {e}")
        return np.nan


def zscore(obs, null_values):
    null_values = np.asarray(null_values, dtype=float)
    mu = np.nanmean(null_values)
    sigma = np.nanstd(null_values, ddof=1)

    if sigma == 0 or np.isnan(sigma):
        z = np.nan
    else:
        z = (-obs + mu) / sigma

    return mu, sigma, z


# ============================================================
# Definición de medidas
# ============================================================

def make_measures(tau_values=(1,)):
    measures = OrderedDict()

    measures["mPE_m5_tau1"] = lambda x: ml.modified_permutation_entropy(
        x, m=3, tau=1
    )

    measures["gamma_rank_max5_mu2"] = lambda x: ml.gamma_index_rank_ties(
        x, max_gamma=1, mu=2)[1][0]

    measures["H_orbit_m3"] = lambda x: ml.H_orbit(x, m=3)

    for tau in tau_values:
        measures[f"S_eff_tau{tau}"] = lambda x, tau=tau: ml.indice_S_eff_fast(
            x, None, tau=tau, delta=False
        )

    return measures


# ============================================================
# Cálculo observado vs modelos nulos
# ============================================================

def compute_observed_and_nulls(
    folder,
    file_ids=range(1, 24),
    n_surrogates=100,
    tau_values=(1,),
    seed=1234
):
    rng = np.random.default_rng(seed)

    melodies = load_melodies(folder, file_ids)
    measures = make_measures(tau_values)

    rows = []

    for piece_id, x in melodies.items():
        print(f"Procesando pieza {piece_id}, N={len(x)}")

        # Nulo IAAFT
        iaaft_surr = ml.iaaft(x, n_surrogates)

        # Nulo shuffle
        shuffle_surr = [rng.permutation(x) for _ in range(n_surrogates)]

        for measure_name, func in measures.items():
            obs = safe_eval(func, x)

            iaaft_vals = np.array(
                [safe_eval(func, xs) for xs in iaaft_surr],
                dtype=float
            )

            shuffle_vals = np.array(
                [safe_eval(func, xs) for xs in shuffle_surr],
                dtype=float
            )

            iaaft_mu, iaaft_sigma, iaaft_z = zscore(obs, iaaft_vals)
            shuffle_mu, shuffle_sigma, shuffle_z = zscore(obs, shuffle_vals)

            rows.append({
                "piece": piece_id,
                "measure": measure_name,
                "obs": obs,

                "iaaft_mu": iaaft_mu,
                "iaaft_sigma": iaaft_sigma,
                "iaaft_z": iaaft_z,

                "shuffle_mu": shuffle_mu,
                "shuffle_sigma": shuffle_sigma,
                "shuffle_z": shuffle_z,

                "iaaft_values": iaaft_vals,
                "shuffle_values": shuffle_vals,
            })

    df = pd.DataFrame(rows)
    return df


df = compute_observed_and_nulls(
    folder=folder,
    file_ids=file_ids,
    n_surrogates=n_surrogates,
    tau_values=tau_values,
    seed=seed
)

# Resumen sin guardar las listas largas de nulos
df_summary = df.drop(columns=["iaaft_values", "shuffle_values"])
df_summary.to_csv("resumen_medidas_vs_nulos.csv", index=False)

df_summary.head()

import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path


def _finite_array(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def plot_all_measures_for_piece(
    df,
    piece_id,
    ncols=2,
    bins="fd",
    hist_style="step",
    common_xlim=False,
    savepath=None
):
    """
    Una figura por pieza.
    Cada subplot corresponde a una medida.

    En cada subplot:
    - distribución nula IAAFT
    - distribución nula shuffle
    - línea vertical del valor observado
    """

    d = df[df["piece"] == piece_id].copy()

    if len(d) == 0:
        raise ValueError(f"No hay datos para la pieza {piece_id}")

    measures = list(d["measure"].unique())
    n_measures = len(measures)

    nrows = math.ceil(n_measures / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.8 * nrows)
    )

    axes = np.asarray(axes).ravel()

    handles_for_legend = None
    labels_for_legend = None

    for ax, measure in zip(axes, measures):
        row = d[d["measure"] == measure].iloc[0]

        obs = float(row["obs"])
        iaaft_vals = _finite_array(row["iaaft_values"])
        shuffle_vals = _finite_array(row["shuffle_values"])

        combined = np.concatenate([
            iaaft_vals,
            shuffle_vals,
            np.array([obs], dtype=float)
        ])

        combined = combined[np.isfinite(combined)]

        if len(combined) < 2:
            ax.set_title(measure)
            ax.text(
                0.5,
                0.5,
                "Datos insuficientes",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.axis("off")
            continue

        edges = np.histogram_bin_edges(combined, bins=bins)

        if hist_style == "filled":
            ax.hist(
                iaaft_vals,
                bins=edges,
                density=True,
                alpha=0.45,
                label="IAAFT"
            )

            ax.hist(
                shuffle_vals,
                bins=edges,
                density=True,
                alpha=0.45,
                label="Shuffle"
            )

        elif hist_style == "step":
            ax.hist(
                iaaft_vals,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                label="IAAFT"
            )

            ax.hist(
                shuffle_vals,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                label="Shuffle"
            )

        else:
            raise ValueError("hist_style debe ser 'filled' o 'step'.")

        ax.axvline(
            obs,
            linestyle="--",
            linewidth=2.0,
            label="Observado"
        )

        ax.set_title(measure)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.grid(alpha=0.25)

        if common_xlim:
            xmin, xmax = np.nanmin(combined), np.nanmax(combined)
            if xmin == xmax:
                pad = 1e-6 if xmin == 0 else 0.05 * abs(xmin)
            else:
                pad = 0.05 * (xmax - xmin)

            ax.set_xlim(xmin - pad, xmax + pad)

        if handles_for_legend is None:
            handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

    for ax in axes[n_measures:]:
        ax.axis("off")

    fig.suptitle(f"Pieza MIDI {piece_id}: observado vs modelos nulos", fontsize=15)

    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995)
    )

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()

def plot_all_pieces_all_measures(
    df,
    ncols=2,
    bins="fd",
    hist_style="step",
    output_folder="figuras_por_pieza"
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pieces = sorted(df["piece"].unique())

    for piece_id in pieces:
        savepath = output_folder / f"pieza_{piece_id}_todas_las_medidas.png"

        plot_all_measures_for_piece(
            df,
            piece_id=piece_id,
            ncols=ncols,
            bins=bins,
            hist_style=hist_style,
            savepath=savepath
        )


plot_all_pieces_all_measures(
    df,
    ncols=2,
    bins="fd",
    hist_style="filled",
    output_folder="figuras_por_pieza_filled"
)