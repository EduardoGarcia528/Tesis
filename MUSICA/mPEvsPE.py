import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress

import mi_libreria as ml

# ============================================================
# Configuración
# ============================================================

BETA_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

M_VALUES = np.arange(3, 8)   # m = 3,4,5,6,7
TAU = 1

N = 5000
N_REPS = 50
RANDOM_SEED = 1234

MELODY_DIR = Path("melodies")
PIECE_IDS = range(1, 24)

# Distribución MIDI uniforme.
# Cambia a np.arange(21, 109) si quieres rango piano.
MIDI_VALUES = np.arange(128)

rng = np.random.default_rng(RANDOM_SEED)

# ============================================================
# Ruido coloreado por exponente beta
# ============================================================

def colored_noise_beta(N, beta=0.0, rng=None, normalize=True):
    """
    Genera ruido gaussiano con espectro de potencias aproximado

        S(f) ~ 1 / f^beta.

    beta = 0   -> blanco
    beta = 1   -> rosa
    beta = 2   -> café / browniano

    Método:
    1. Se genera ruido blanco.
    2. Se toma FFT real.
    3. Se multiplica la amplitud por f^(-beta/2),
       porque la potencia es amplitud^2.
    4. Se regresa al dominio temporal.

    Parámetros
    ----------
    N : int
        Longitud de la serie.
    beta : float
        Exponente espectral.
    rng : np.random.Generator
        Generador aleatorio.
    normalize : bool
        Si True, regresa la serie con media 0 y desviación estándar 1.

    Regresa
    -------
    x : np.ndarray
        Serie temporal.
    """
    if rng is None:
        rng = np.random.default_rng()

    white = rng.normal(size=N)

    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N, d=1.0)

    scale = np.ones_like(freqs)
    nonzero = freqs > 0

    scale[nonzero] = freqs[nonzero] ** (-beta / 2.0)

    # Eliminar componente DC para evitar desplazamientos grandes.
    scale[0] = 0.0

    X_colored = X * scale
    x = np.fft.irfft(X_colored, n=N)

    if normalize:
        x = x - np.mean(x)
        std = np.std(x)
        if std > 0:
            x = x / std

    return x

def estimate_beta_periodogram(x, fmin=0.01, fmax=0.45):
    """
    Estima beta ajustando log P(f) contra log f.

    Si S(f) ~ 1/f^beta, entonces:

        log S(f) = const - beta log f.

    Por tanto:

        beta_hat = -slope.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    power = np.abs(X) ** 2

    mask = (freqs > fmin) & (freqs < fmax) & np.isfinite(power) & (power > 0)

    logf = np.log(freqs[mask])
    logp = np.log(power[mask])

    slope, intercept, r, p, stderr = linregress(logf, logp)

    beta_hat = -slope

    return beta_hat, r**2

# ============================================================
# Limpieza, mapeo y métricas
# ============================================================

def clean_melody(arr, remove_negative=True):
    """
    Limpieza mínima de melodías MIDI.

    Si tus silencios están codificados como -1, esto los elimina.
    Si no quieres eliminar negativos, usa remove_negative=False.
    """
    arr = np.asarray(arr)

    arr = arr[np.isfinite(arr)]

    if remove_negative:
        arr = arr[arr >= 0]

    return arr.astype(int)


def make_uniform_midi_target(N, midi_values=MIDI_VALUES, rng=None):
    """
    Construye una distribución uniforme discreta de longitud N
    sobre los valores MIDI indicados.
    """
    if rng is None:
        rng = np.random.default_rng()

    midi_values = np.asarray(midi_values, dtype=int)
    K = len(midi_values)

    q, r = divmod(N, K)

    target = np.repeat(midi_values, q)

    if r > 0:
        extra = rng.choice(midi_values, size=r, replace=False)
        target = np.concatenate([target, extra])

    return target.astype(int)


def rank_map_to_target_distribution(x, target_values):
    """
    Mapea una serie continua x a una distribución discreta objetivo.

    El mapeo preserva rangos:

        valores bajos de x  -> notas bajas del target
        valores altos de x  -> notas altas del target

    La distribución marginal de salida es exactamente la de target_values.
    """
    x = np.asarray(x, dtype=float)
    target_values = np.asarray(target_values)

    if len(x) != len(target_values):
        raise ValueError("x y target_values deben tener la misma longitud.")

    order_x = np.argsort(x, kind="mergesort")
    sorted_target = np.sort(target_values)

    y = np.empty(len(x), dtype=sorted_target.dtype)
    y[order_x] = sorted_target

    return y


def compute_PE_mPE_raw(arr, m, tau=1):
    """
    Calcula PE y mPE sin normalización.
    """
    PE = float(ml.permutation_entropy(arr, m, tau=tau, norm=False))
    mPE = float(ml.modified_permutation_entropy(arr, m, tau=tau, norm=False))

    return PE, mPE


def tie_statistics(arr):
    """
    Estadísticos simples de empates y repeticiones locales.
    Útiles para interpretar mPE.
    """
    arr = np.asarray(arr)

    if len(arr) < 2:
        return {
            "p_equal_lag1": np.nan,
            "mean_abs_step": np.nan,
            "median_abs_step": np.nan,
            "n_unique": len(np.unique(arr)),
        }

    diffs = np.diff(arr)

    p_equal_lag1 = np.mean(diffs == 0)
    abs_step = np.abs(diffs)

    return {
        "p_equal_lag1": float(p_equal_lag1),
        "mean_abs_step": float(np.mean(abs_step)),
        "median_abs_step": float(np.median(abs_step)),
        "n_unique": int(len(np.unique(arr))),
    }

# ============================================================
# Experimento 1:
# Ruido continuo con beta >= 0
# ============================================================

records_exp1 = []

for rep in range(N_REPS):
    for beta in BETA_VALUES:

        local_rng = np.random.default_rng(RANDOM_SEED + 10_000 * rep + int(1000 * beta))

        x = colored_noise_beta(N, beta=beta, rng=local_rng, normalize=True)

        beta_hat, beta_fit_r2 = estimate_beta_periodogram(x)

        stats = tie_statistics(x)

        for m in M_VALUES:
            PE, mPE = compute_PE_mPE_raw(x, m, tau=TAU)

            records_exp1.append({
                "experiment": "continuous_beta_noise",
                "target_distribution": "continuous",
                "piece_id": np.nan,
                "rep": rep,
                "beta": beta,
                "beta_hat": beta_hat,
                "beta_fit_r2": beta_fit_r2,
                "m": m,
                "PE_raw": PE,
                "mPE_raw": mPE,
                "delta_mPE_PE_raw": mPE - PE,
                **stats,
            })

df_exp1 = pd.DataFrame(records_exp1)

df_exp1.head()

# ============================================================
# Experimento 2:
# Ruido beta >= 0 -> distribución uniforme MIDI
# ============================================================

records_exp2 = []

for rep in range(N_REPS):
    for beta in BETA_VALUES:

        local_rng = np.random.default_rng(RANDOM_SEED + 20_000 * rep + int(1000 * beta))

        x = colored_noise_beta(N, beta=beta, rng=local_rng, normalize=True)

        beta_hat, beta_fit_r2 = estimate_beta_periodogram(x)

        target_uniform = make_uniform_midi_target(N, midi_values=MIDI_VALUES, rng=local_rng)

        y = rank_map_to_target_distribution(x, target_uniform)

        stats = tie_statistics(y)

        for m in M_VALUES:
            PE, mPE = compute_PE_mPE_raw(y, m, tau=TAU)

            records_exp2.append({
                "experiment": "beta_noise_to_uniform_MIDI",
                "target_distribution": "uniform_MIDI",
                "piece_id": np.nan,
                "rep": rep,
                "beta": beta,
                "beta_hat": beta_hat,
                "beta_fit_r2": beta_fit_r2,
                "m": m,
                "PE_raw": PE,
                "mPE_raw": mPE,
                "delta_mPE_PE_raw": mPE - PE,
                **stats,
            })

df_exp2 = pd.DataFrame(records_exp2)

df_exp2.head()

# ============================================================
# Experimento 3:
# Ruido beta >= 0 -> distribución empírica de cada pieza
# ============================================================

records_exp3 = []

for piece_id in PIECE_IDS:

    melody_path = MELODY_DIR / f"{piece_id}.npy"

    if not melody_path.exists():
        print(f"No existe: {melody_path}")
        continue

    melody = clean_melody(np.load(melody_path), remove_negative=True)
    N_piece = len(melody)

    if N_piece < max(M_VALUES) + 2:
        print(f"Pieza {piece_id} demasiado corta: N={N_piece}")
        continue

    for rep in range(N_REPS):
        for beta in BETA_VALUES:

            local_rng = np.random.default_rng(
                RANDOM_SEED
                + 30_000 * rep
                + 1_000 * piece_id
                + int(1000 * beta)
            )

            x = colored_noise_beta(N_piece, beta=beta, rng=local_rng, normalize=True)

            beta_hat, beta_fit_r2 = estimate_beta_periodogram(x)

            # Mapeo exacto a la distribución empírica de la pieza.
            y = rank_map_to_target_distribution(x, melody)

            stats = tie_statistics(y)

            for m in M_VALUES:
                PE, mPE = compute_PE_mPE_raw(y, m, tau=TAU)

                records_exp3.append({
                    "experiment": "beta_noise_to_real_piece_MIDI",
                    "target_distribution": "empirical_piece_MIDI",
                    "piece_id": piece_id,
                    "rep": rep,
                    "beta": beta,
                    "beta_hat": beta_hat,
                    "beta_fit_r2": beta_fit_r2,
                    "m": m,
                    "PE_raw": PE,
                    "mPE_raw": mPE,
                    "delta_mPE_PE_raw": mPE - PE,
                    "N_piece": N_piece,
                    **stats,
                })

df_exp3 = pd.DataFrame(records_exp3)

df_exp3.head()

# ============================================================
# Unir resultados
# ============================================================

df_all = pd.concat([df_exp1, df_exp2, df_exp3], ignore_index=True)

df_all.to_csv("beta_positive_PE_mPE_raw_all_results.csv", index=False)

df_all.head()

# ============================================================
# Resumen por condición
# ============================================================

summary = (
    df_all
    .groupby(
        ["experiment", "target_distribution", "piece_id", "beta", "m"],
        dropna=False
    )
    .agg(
        beta_hat_mean=("beta_hat", "mean"),
        beta_hat_std=("beta_hat", "std"),
        beta_fit_r2_mean=("beta_fit_r2", "mean"),

        PE_raw_mean=("PE_raw", "mean"),
        PE_raw_std=("PE_raw", "std"),

        mPE_raw_mean=("mPE_raw", "mean"),
        mPE_raw_std=("mPE_raw", "std"),

        delta_raw_mean=("delta_mPE_PE_raw", "mean"),
        delta_raw_std=("delta_mPE_PE_raw", "std"),

        p_equal_lag1_mean=("p_equal_lag1", "mean"),
        p_equal_lag1_std=("p_equal_lag1", "std"),

        mean_abs_step_mean=("mean_abs_step", "mean"),
        median_abs_step_mean=("median_abs_step", "mean"),

        n_unique_mean=("n_unique", "mean"),

        n=("PE_raw", "size"),
    )
    .reset_index()
)

summary.to_csv("beta_positive_PE_mPE_raw_summary_by_beta.csv", index=False)

summary.head()

# ============================================================
# Correlaciones contra beta
# ============================================================

def correlation_by_beta_means(df, metrics=("PE_raw", "mPE_raw", "delta_mPE_PE_raw")):
    records = []

    group_cols = ["experiment", "target_distribution", "piece_id", "m"]

    for group_key, g in df.groupby(group_cols, dropna=False):

        experiment, target_distribution, piece_id, m = group_key

        beta_means = (
            g.groupby("beta", dropna=False)
            .agg({metric: "mean" for metric in metrics})
            .reset_index()
            .sort_values("beta")
        )

        if len(beta_means) < 3:
            continue

        beta_values = beta_means["beta"].values

        for metric in metrics:

            y = beta_means[metric].values

            pearson_r, pearson_p = pearsonr(beta_values, y)
            spearman_r, spearman_p = spearmanr(beta_values, y)

            slope, intercept, lin_r, lin_p, stderr = linregress(beta_values, y)

            records.append({
                "experiment": experiment,
                "target_distribution": target_distribution,
                "piece_id": piece_id,
                "m": m,
                "metric": metric,

                "pearson_r": pearson_r,
                "pearson_p": pearson_p,

                "spearman_r": spearman_r,
                "spearman_p": spearman_p,

                "slope_vs_beta": slope,
                "intercept": intercept,
                "linear_r2": lin_r ** 2,
                "linear_p": lin_p,
                "slope_stderr": stderr,
            })

    return pd.DataFrame(records)


corrs = correlation_by_beta_means(df_all)

corrs.to_csv("beta_positive_PE_mPE_raw_correlations.csv", index=False)

corrs.head()

# ============================================================
# Resumen de correlaciones por piezas reales
# ============================================================

piece_corr_summary = (
    corrs[~corrs["piece_id"].isna()]
    .groupby(["experiment", "target_distribution", "metric", "m"])
    .agg(
        pearson_r_mean=("pearson_r", "mean"),
        pearson_r_std=("pearson_r", "std"),
        pearson_r_min=("pearson_r", "min"),
        pearson_r_max=("pearson_r", "max"),

        spearman_r_mean=("spearman_r", "mean"),
        spearman_r_std=("spearman_r", "std"),
        spearman_r_min=("spearman_r", "min"),
        spearman_r_max=("spearman_r", "max"),

        slope_mean=("slope_vs_beta", "mean"),
        slope_std=("slope_vs_beta", "std"),
        slope_min=("slope_vs_beta", "min"),
        slope_max=("slope_vs_beta", "max"),

        linear_r2_mean=("linear_r2", "mean"),
        linear_r2_std=("linear_r2", "std"),

        n_pieces=("piece_id", "nunique"),
    )
    .reset_index()
)

piece_corr_summary.to_csv("beta_positive_PE_mPE_raw_piece_correlation_summary.csv", index=False)

piece_corr_summary

def plot_metric_vs_beta(summary_df, experiment, target_distribution, metric="PE_raw", piece_id=np.nan):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    g = summary_df[
        (summary_df["experiment"] == experiment)
        & (summary_df["target_distribution"] == target_distribution)
    ].copy()

    if pd.isna(piece_id):
        g = g[g["piece_id"].isna()]
        title_piece = ""
    else:
        g = g[g["piece_id"] == piece_id]
        title_piece = f", pieza {piece_id}"

    plt.figure(figsize=(7, 4))

    for m in sorted(g["m"].unique()):
        gm = g[g["m"] == m].sort_values("beta")

        plt.errorbar(
            gm["beta"],
            gm[mean_col],
            yerr=gm[std_col],
            marker="o",
            capsize=3,
            label=f"m={m}"
        )

    plt.xlabel(r"$\beta$")
    plt.ylabel(metric)
    plt.title(f"{metric} vs beta: {experiment}{title_piece}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Ruido continuo
plot_metric_vs_beta(
    summary,
    experiment="continuous_beta_noise",
    target_distribution="continuous",
    metric="PE_raw"
)

plot_metric_vs_beta(
    summary,
    experiment="continuous_beta_noise",
    target_distribution="continuous",
    metric="mPE_raw"
)

# MIDI uniforme
plot_metric_vs_beta(
    summary,
    experiment="beta_noise_to_uniform_MIDI",
    target_distribution="uniform_MIDI",
    metric="PE_raw"
)

plot_metric_vs_beta(
    summary,
    experiment="beta_noise_to_uniform_MIDI",
    target_distribution="uniform_MIDI",
    metric="mPE_raw"
)

# MIDI empírico, pieza 1
plot_metric_vs_beta(
    summary,
    experiment="beta_noise_to_real_piece_MIDI",
    target_distribution="empirical_piece_MIDI",
    metric="PE_raw",
    piece_id=1
)

plot_metric_vs_beta(
    summary,
    experiment="beta_noise_to_real_piece_MIDI",
    target_distribution="empirical_piece_MIDI",
    metric="mPE_raw",
    piece_id=1
)

def plot_corr_vs_m(corr_df, experiment, target_distribution, metric="PE_raw", piece_id=np.nan):
    g = corr_df[
        (corr_df["experiment"] == experiment)
        & (corr_df["target_distribution"] == target_distribution)
        & (corr_df["metric"] == metric)
    ].copy()

    if pd.isna(piece_id):
        g = g[g["piece_id"].isna()]
        title_piece = ""
    else:
        g = g[g["piece_id"] == piece_id]
        title_piece = f", pieza {piece_id}"

    g = g.sort_values("m")

    plt.figure(figsize=(7, 4))
    plt.plot(g["m"], g["pearson_r"], marker="o", label="Pearson")
    plt.plot(g["m"], g["spearman_r"], marker="s", label="Spearman")

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("m")
    plt.ylabel(r"correlación con $\beta$")
    plt.title(f"{metric}: {experiment}{title_piece}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Continuo
plot_corr_vs_m(
    corrs,
    experiment="continuous_beta_noise",
    target_distribution="continuous",
    metric="PE_raw"
)

plot_corr_vs_m(
    corrs,
    experiment="continuous_beta_noise",
    target_distribution="continuous",
    metric="mPE_raw"
)

# MIDI uniforme
plot_corr_vs_m(
    corrs,
    experiment="beta_noise_to_uniform_MIDI",
    target_distribution="uniform_MIDI",
    metric="PE_raw"
)

plot_corr_vs_m(
    corrs,
    experiment="beta_noise_to_uniform_MIDI",
    target_distribution="uniform_MIDI",
    metric="mPE_raw"
)

# Pieza 1
plot_corr_vs_m(
    corrs,
    experiment="beta_noise_to_real_piece_MIDI",
    target_distribution="empirical_piece_MIDI",
    metric="PE_raw",
    piece_id=1
)

plot_corr_vs_m(
    corrs,
    experiment="beta_noise_to_real_piece_MIDI",
    target_distribution="empirical_piece_MIDI",
    metric="mPE_raw",
    piece_id=1
)


def plot_delta_vs_beta(summary_df, experiment, target_distribution, piece_id=np.nan):
    g = summary_df[
        (summary_df["experiment"] == experiment)
        & (summary_df["target_distribution"] == target_distribution)
    ].copy()

    if pd.isna(piece_id):
        g = g[g["piece_id"].isna()]
        title_piece = ""
    else:
        g = g[g["piece_id"] == piece_id]
        title_piece = f", pieza {piece_id}"

    plt.figure(figsize=(7, 4))

    for m in sorted(g["m"].unique()):
        gm = g[g["m"] == m].sort_values("beta")

        plt.errorbar(
            gm["beta"],
            gm["delta_raw_mean"],
            yerr=gm["delta_raw_std"],
            marker="o",
            capsize=3,
            label=f"m={m}"
        )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel(r"$\beta$")
    plt.ylabel("mPE_raw - PE_raw")
    plt.title(f"Diferencia cruda mPE - PE: {experiment}{title_piece}")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_delta_vs_beta(
    summary,
    experiment="continuous_beta_noise",
    target_distribution="continuous"
)

plot_delta_vs_beta(
    summary,
    experiment="beta_noise_to_uniform_MIDI",
    target_distribution="uniform_MIDI"
)

plot_delta_vs_beta(
    summary,
    experiment="beta_noise_to_real_piece_MIDI",
    target_distribution="empirical_piece_MIDI",
    piece_id=1
)

def plot_ties_vs_beta(summary_df, experiment, target_distribution, piece_id=np.nan):
    g = summary_df[
        (summary_df["experiment"] == experiment)
        & (summary_df["target_distribution"] == target_distribution)
    ].copy()

    if pd.isna(piece_id):
        g = g[g["piece_id"].isna()]
        title_piece = ""
    else:
        g = g[g["piece_id"] == piece_id]
        title_piece = f", pieza {piece_id}"

    # p_equal no depende de m, pero está repetido para cada m.
    # Tomamos m mínimo para no duplicar.
    m0 = g["m"].min()
    g = g[g["m"] == m0].sort_values("beta")

    plt.figure(figsize=(7, 4))

    plt.errorbar(
        g["beta"],
        g["p_equal_lag1_mean"],
        yerr=g["p_equal_lag1_std"],
        marker="o",
        capsize=3,
    )

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$P(y_t = y_{t+1})$")
    plt.title(f"Empates consecutivos: {experiment}{title_piece}")
    plt.tight_layout()
    plt.show()


plot_ties_vs_beta(
    summary,
    experiment="beta_noise_to_uniform_MIDI",
    target_distribution="uniform_MIDI"
)

plot_ties_vs_beta(
    summary,
    experiment="beta_noise_to_real_piece_MIDI",
    target_distribution="empirical_piece_MIDI",
    piece_id=1
)

def plot_piece_corr_summary(piece_corr_summary, metric="mPE_raw"):
    g = piece_corr_summary[
        piece_corr_summary["metric"] == metric
    ].sort_values("m")

    plt.figure(figsize=(7, 4))

    plt.errorbar(
        g["m"],
        g["pearson_r_mean"],
        yerr=g["pearson_r_std"],
        marker="o",
        capsize=3,
        label="Pearson"
    )

    plt.errorbar(
        g["m"],
        g["spearman_r_mean"],
        yerr=g["spearman_r_std"],
        marker="s",
        capsize=3,
        label="Spearman"
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("m")
    plt.ylabel(r"correlación con $\beta$")
    plt.title(f"Distribución empírica real: {metric}")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_piece_corr_summary(piece_corr_summary, metric="PE_raw")
plot_piece_corr_summary(piece_corr_summary, metric="mPE_raw")
plot_piece_corr_summary(piece_corr_summary, metric="delta_mPE_PE_raw")

# Correlaciones globales: continuo y MIDI uniforme
corrs_global = corrs[corrs["piece_id"].isna()].copy()

corrs_global.sort_values(["experiment", "metric", "m"])

# Correlaciones promedio sobre piezas reales
piece_corr_summary.sort_values(["metric", "m"])

# Medias por beta para ruido continuo
summary[
    (summary["experiment"] == "continuous_beta_noise")
    & (summary["target_distribution"] == "continuous")
    & (summary["piece_id"].isna())
].sort_values(["m", "beta"])

# Medias por beta para MIDI uniforme
summary[
    (summary["experiment"] == "beta_noise_to_uniform_MIDI")
    & (summary["target_distribution"] == "uniform_MIDI")
    & (summary["piece_id"].isna())
].sort_values(["m", "beta"])

# Medias por beta para una pieza real
summary[
    (summary["experiment"] == "beta_noise_to_real_piece_MIDI")
    & (summary["target_distribution"] == "empirical_piece_MIDI")
    & (summary["piece_id"] == 1)
].sort_values(["m", "beta"])