# -*- coding: utf-8 -*-
"""
Análisis de invariancia de escala de distribuciones de Z-score de mPE.

Supone archivos .pkl dentro de CACHE_DIR, con:
    - una columna llamada 'z'
    - nombres que contienen _D<m> y _tau<tau>
      por ejemplo: ..._D3_tau1....pkl

El script:
1) Calcula momentos y estadísticos robustos para cada (m, tau).
2) Estandariza cada distribución por mediana e IQR:
       y = (z - mediana) / IQR
   para probar invariancia de forma bajo transformaciones afines.
3) Compara, dentro de cada m, todos los pares de tau mediante:
       - KS sobre las distribuciones estandarizadas
       - distancia de Wasserstein
       - distancia de energía
       - comparación de cuantiles (QQ): pendiente, intercepto y R^2
4) Produce figuras de:
       - distribuciones originales
       - colapso de escala
       - momentos vs tau
       - QQ frente a tau de referencia
5) Guarda tablas CSV con momentos y comparaciones.

NOTA:
El KS estándar se usa como diagnóstico de forma. Como los datos han sido
estandarizados usando parámetros estimados de cada muestra, su p-value debe
interpretarse con cautela; las distancias y el colapso gráfico son igualmente
importantes.
"""

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# CONFIGURACIÓN
# ============================================================
CACHE_DIR = "cache_pe_score"
OUT_DIR = "analisis_invariancia_escala_mPE"

M_VALUES = [3, 4, 5, 6, 7]
TAU_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

# Tu script original devuelve -z. Mantener True reproduce esa convención.
FLIP_Z_SIGN = True

# Número de puntos para curvas de densidad KDE y QQ
N_GRID = 500
N_QQ = 200

# Si hay colas extremas, limita sólo la VISUALIZACIÓN de KDE a estos cuantiles.
# Los cálculos estadísticos siguen usando todos los datos.
PLOT_QMIN = 0.002
PLOT_QMAX = 0.998

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# CARGA Y PARSEO
# ============================================================
def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", str(text))]


def parse_m_tau(fname):
    """Extrae m desde _D<m> y tau desde _tau<tau>."""
    mm = re.search(r"_D(\d+)", fname)
    mt = re.search(r"_tau(\d+)", fname)
    if mm is None or mt is None:
        return None, None
    return int(mm.group(1)), int(mt.group(1))


def load_z_values(path):
    df = pd.read_pickle(path)
    if "z" not in df.columns:
        raise ValueError(f"El archivo no contiene columna 'z': {path}")

    z = df["z"].to_numpy(dtype=float)
    z = z[np.isfinite(z)]

    if FLIP_Z_SIGN:
        z = -z

    return z


def load_all_distributions():
    data = {}

    files = sorted(
        [f for f in os.listdir(CACHE_DIR) if f.lower().endswith(".pkl")],
        key=natural_key
    )

    for fname in files:
        m, tau = parse_m_tau(fname)
        if m is None:
            continue
        if m not in M_VALUES or tau not in TAU_VALUES:
            continue

        key = (m, tau)
        if key in data:
            raise ValueError(
                f"Hay más de un archivo para m={m}, tau={tau}. "
                f"Revisa los nombres dentro de {CACHE_DIR}."
            )

        data[key] = load_z_values(os.path.join(CACHE_DIR, fname))

    expected = {(m, tau) for m in M_VALUES for tau in TAU_VALUES}
    missing = sorted(expected - set(data))
    if missing:
        print("ADVERTENCIA: faltan distribuciones:", missing)

    if not data:
        raise FileNotFoundError(
            f"No encontré archivos compatibles en '{CACHE_DIR}'."
        )

    return data


# ============================================================
# ESTADÍSTICA
# ============================================================
def robust_standardize(x):
    """
    Quita localización y escala con mediana e IQR.
    Esta opción es robusta frente a colas y outliers.
    """
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25

    if iqr <= 0:
        raise ValueError("IQR nulo: no se puede estandarizar la distribución.")

    return (x - med) / iqr


def sample_statistics(x):
    """Momentos clásicos + estadísticos robustos."""
    n = len(x)
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    std = np.std(x, ddof=1)
    median = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    mad = stats.median_abs_deviation(x, scale="normal")

    # Momentos centrales muestrales (sin estandarizar)
    centered = x - mean
    mu2 = np.mean(centered**2)
    mu3 = np.mean(centered**3)
    mu4 = np.mean(centered**4)

    # Momentos estandarizados: invariantes a z -> a z + b, a > 0
    skew = stats.skew(x, bias=False)
    kurt_excess = stats.kurtosis(x, fisher=True, bias=False)

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "variance": var,
        "std": std,
        "IQR": iqr,
        "MAD_normal": mad,
        "mu2": mu2,
        "mu3": mu3,
        "mu4": mu4,
        "skewness": skew,
        "excess_kurtosis": kurt_excess,
    }


def build_moments_table(data):
    rows = []
    for (m, tau), x in sorted(data.items()):
        row = {"m": m, "tau": tau}
        row.update(sample_statistics(x))
        rows.append(row)
    return pd.DataFrame(rows)


def qq_affine_fit(x, y, n_q=N_QQ):
    """
    Compara cuantiles originales de dos distribuciones.

    Si y = a*x + b en distribución, entonces:
        Q_y(p) = a Q_x(p) + b
    y el QQ será una recta.

    Devuelve pendiente a, intercepto b y R^2.
    """
    probs = np.linspace(0.01, 0.99, n_q)
    qx = np.quantile(x, probs)
    qy = np.quantile(y, probs)

    slope, intercept, r_value, _, _ = stats.linregress(qx, qy)
    return slope, intercept, r_value**2, qx, qy


def pairwise_shape_comparisons(data):
    rows = []

    for m in M_VALUES:
        available_tau = [tau for tau in TAU_VALUES if (m, tau) in data]

        for tau1, tau2 in combinations(available_tau, 2):
            x = data[(m, tau1)]
            y = data[(m, tau2)]

            xs = robust_standardize(x)
            ys = robust_standardize(y)

            ks = stats.ks_2samp(xs, ys, alternative="two-sided", method="auto")
            wasser = stats.wasserstein_distance(xs, ys)
            energy = stats.energy_distance(xs, ys)

            slope, intercept, r2, _, _ = qq_affine_fit(x, y)

            # Cocientes de escala robustos y clásicos.
            # Para invariancia de escala, ambos deberían describir un factor
            # aproximadamente consistente entre las dos distribuciones.
            iqr_x = np.subtract(*np.percentile(x, [75, 25]))
            iqr_y = np.subtract(*np.percentile(y, [75, 25]))
            std_x = np.std(x, ddof=1)
            std_y = np.std(y, ddof=1)

            rows.append({
                "m": m,
                "tau1": tau1,
                "tau2": tau2,
                "KS_standardized": ks.statistic,
                "KS_pvalue_standardized": ks.pvalue,
                "Wasserstein_standardized": wasser,
                "Energy_standardized": energy,
                "QQ_slope_tau2_vs_tau1": slope,
                "QQ_intercept": intercept,
                "QQ_R2": r2,
                "IQR_ratio_tau2_over_tau1": iqr_y / iqr_x,
                "std_ratio_tau2_over_tau1": std_y / std_x,
            })

    return pd.DataFrame(rows)


# ============================================================
# FIGURAS
# ============================================================
def kde_on_grid(x, grid):
    if len(np.unique(x)) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    kde = stats.gaussian_kde(x)
    return kde(grid)


def plot_original_distributions(data):
    fig, axs = plt.subplots(len(M_VALUES), 1, figsize=(8, 3.2 * len(M_VALUES)))
    axs = np.atleast_1d(axs)

    for ax, m in zip(axs, M_VALUES):
        arrays = [data[(m, tau)] for tau in TAU_VALUES if (m, tau) in data]
        if not arrays:
            ax.axis("off")
            continue

        pooled = np.concatenate(arrays)
        xmin, xmax = np.quantile(pooled, [PLOT_QMIN, PLOT_QMAX])
        grid = np.linspace(xmin, xmax, N_GRID)

        for tau in TAU_VALUES:
            if (m, tau) not in data:
                continue
            x = data[(m, tau)]
            ax.plot(grid, kde_on_grid(x, grid), label=rf"$\tau={tau}$")

        ax.set_title(rf"Distribuciones originales — $m={m}$")
        ax.set_xlabel("Z-score (mPE)")
        ax.set_ylabel("Densidad")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_distribuciones_originales.png"), dpi=300)
    plt.close(fig)


def plot_scale_collapse(data):
    fig, axs = plt.subplots(len(M_VALUES), 1, figsize=(8, 3.2 * len(M_VALUES)))
    axs = np.atleast_1d(axs)

    for ax, m in zip(axs, M_VALUES):
        standardized = {
            tau: robust_standardize(data[(m, tau)])
            for tau in TAU_VALUES if (m, tau) in data
        }

        if not standardized:
            ax.axis("off")
            continue

        pooled = np.concatenate(list(standardized.values()))
        xmin, xmax = np.quantile(pooled, [PLOT_QMIN, PLOT_QMAX])
        grid = np.linspace(xmin, xmax, N_GRID)

        for tau, x in standardized.items():
            ax.plot(grid, kde_on_grid(x, grid), label=rf"$\tau={tau}$")

        ax.set_title(
            rf"Colapso de escala — $m={m}$: "
            r"$(Z-\mathrm{mediana})/\mathrm{IQR}$"
        )
        ax.set_xlabel(r"$(Z-\mathrm{mediana})/\mathrm{IQR}$")
        ax.set_ylabel("Densidad")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_colapso_invariancia_escala.png"), dpi=300)
    plt.close(fig)


def plot_moments(moment_df):
    metrics = [
        ("mean", "Media"),
        ("std", "Desviación estándar"),
        ("skewness", "Asimetría"),
        ("excess_kurtosis", "Curtosis (exceso)"),
    ]

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for m in M_VALUES:
            sub = moment_df[moment_df["m"] == m].sort_values("tau")
            if sub.empty:
                continue
            ax.plot(sub["tau"], sub[metric], marker="o", label=rf"$m={m}$")

        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(TAU_VALUES)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUT_DIR, f"03_momento_{metric}.png"),
            dpi=300
        )
        plt.close(fig)


def plot_qq_against_reference(data):
    """QQ de cada tau contra el menor tau disponible, por cada m."""
    fig, axs = plt.subplots(len(M_VALUES), 1, figsize=(7, 4 * len(M_VALUES)))
    axs = np.atleast_1d(axs)

    for ax, m in zip(axs, M_VALUES):
        available_tau = [tau for tau in TAU_VALUES if (m, tau) in data]
        if len(available_tau) < 2:
            ax.axis("off")
            continue

        tau_ref = min(available_tau)
        x = data[(m, tau_ref)]

        for tau in available_tau:
            if tau == tau_ref:
                continue

            y = data[(m, tau)]
            slope, intercept, r2, qx, qy = qq_affine_fit(x, y)

            ax.plot(qx, qy, ".", ms=3,
                    label=rf"$\tau={tau}$: $R^2={r2:.4f}$, $a={slope:.3g}$")

            xx = np.array([qx.min(), qx.max()])
            ax.plot(xx, slope * xx + intercept, "--", linewidth=1)

        ax.set_title(rf"QQ frente a $\tau={tau_ref}$ — $m={m}$")
        ax.set_xlabel(rf"Cuantiles de $\tau={tau_ref}$")
        ax.set_ylabel("Cuantiles de la otra distribución")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_QQ_afin_por_m.png"), dpi=300)
    plt.close(fig)


def plot_pairwise_shape_distances(pair_df):
    """Resumen visual de distancias entre distribuciones ya estandarizadas."""
    if pair_df.empty:
        return

    for metric, ylabel in [
        ("KS_standardized", "KS tras estandarización"),
        ("Wasserstein_standardized", "Wasserstein tras estandarización"),
        ("Energy_standardized", "Distancia de energía tras estandarización"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.8))

        labels = []
        xpos = []
        vals = []
        tick = 0

        for m in M_VALUES:
            sub = pair_df[pair_df["m"] == m]
            for _, row in sub.iterrows():
                labels.append(rf"$m={m}$\n{int(row.tau1)}-{int(row.tau2)}")
                xpos.append(tick)
                vals.append(row[metric])
                tick += 1

        ax.scatter(xpos, vals)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"05_{metric}.png"), dpi=300)
        plt.close(fig)


# ============================================================
# RESUMEN POR m: ¿QUÉ TAN CONSTANTE ES LA FORMA?
# ============================================================
def summarize_shape_stability(moment_df, pair_df):
    rows = []

    for m in M_VALUES:
        mm = moment_df[moment_df["m"] == m]
        pp = pair_df[pair_df["m"] == m]
        if mm.empty:
            continue

        # Rango relativo de escala: cuánto cambia std e IQR con tau.
        std_ratio_maxmin = mm["std"].max() / mm["std"].min()
        iqr_ratio_maxmin = mm["IQR"].max() / mm["IQR"].min()

        # La skewness y la kurtosis son invariantes ante cambios afines de escala.
        skew_range = mm["skewness"].max() - mm["skewness"].min()
        kurt_range = mm["excess_kurtosis"].max() - mm["excess_kurtosis"].min()

        rows.append({
            "m": m,
            "std_max_over_min": std_ratio_maxmin,
            "IQR_max_over_min": iqr_ratio_maxmin,
            "skewness_range_over_tau": skew_range,
            "kurtosis_range_over_tau": kurt_range,
            "mean_pairwise_KS_standardized": pp["KS_standardized"].mean() if not pp.empty else np.nan,
            "max_pairwise_KS_standardized": pp["KS_standardized"].max() if not pp.empty else np.nan,
            "mean_pairwise_Wasserstein_standardized": pp["Wasserstein_standardized"].mean() if not pp.empty else np.nan,
            "min_QQ_R2": pp["QQ_R2"].min() if not pp.empty else np.nan,
            "mean_QQ_R2": pp["QQ_R2"].mean() if not pp.empty else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    data = load_all_distributions()

    moments = build_moments_table(data)
    comparisons = pairwise_shape_comparisons(data)
    summary = summarize_shape_stability(moments, comparisons)

    moments.to_csv(
        os.path.join(OUT_DIR, "momentos_por_m_tau.csv"),
        index=False
    )
    comparisons.to_csv(
        os.path.join(OUT_DIR, "comparaciones_invariancia_escala.csv"),
        index=False
    )
    summary.to_csv(
        os.path.join(OUT_DIR, "resumen_invariancia_por_m.csv"),
        index=False
    )

    plot_original_distributions(data)
    plot_scale_collapse(data)
    plot_moments(moments)
    plot_qq_against_reference(data)
    plot_pairwise_shape_distances(comparisons)

    print("\n=== MOMENTOS POR (m, tau) ===")
    cols = ["m", "tau", "n", "mean", "std", "median", "IQR",
            "skewness", "excess_kurtosis"]
    print(moments[cols].to_string(index=False))

    print("\n=== RESUMEN DE INVARIANCIA POR m ===")
    print(summary.to_string(index=False))

    print(f"\nResultados guardados en: {OUT_DIR}/")
    print("\nInterpretación sugerida:")
    print("  * Si las KDE estandarizadas colapsan, hay evidencia visual de invariancia de forma.")
    print("  * KS, Wasserstein y Energy cercanos a 0 => distribuciones estandarizadas similares.")
    print("  * QQ con R^2 cercano a 1 => una distribución es aproximadamente una transformación afín de la otra.")
    print("  * skewness y exceso de curtosis aproximadamente constantes con tau => forma estable.")
    print("  * std e IQR pueden variar: precisamente cuantifican el cambio de escala.")


if __name__ == "__main__":
    main()
