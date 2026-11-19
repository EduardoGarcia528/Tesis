# -*- coding: utf-8 -*-
"""Experimentos 1-3: mediana, ley de potencia para IQR y comparación de colapsos.

Compatible con los .pkl de analisis_invariancia_escala_mPE(1).py.
Cada archivo debe tener una columna 'z' y un nombre con _D<m> y _tau<tau>.

Requisitos: numpy, pandas, scipy, matplotlib.
Ejecutar: python experimentos_escala_mPE.py
"""
from pathlib import Path
from itertools import combinations
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# CONFIGURACIÓN: idéntica convención que el script original
# ============================================================
CACHE_DIR = Path("cache_pe_score")
OUT_DIR = Path("experimentos_ley_escala_mPE")
M_VALUES = [3, 4, 5, 6, 7]
TAU_VALUES = [1, 2, 3, 4,5,6,7,8]
FLIP_Z_SIGN = True  # conserva z -> -z del script original

N_GRID = 350
PLOT_QMIN, PLOT_QMAX = 0.002, 0.998
MAX_KDE_POINTS = 4000   # sólo afecta las FIGURAS, no los cálculos
SEED = 12345


# ============================================================
# CARGA DE LAS MISMAS DISTRIBUCIONES
# ============================================================
def parse_m_tau(filename):
    mm = re.search(r"_D(\d+)", filename)
    mt = re.search(r"_tau(\d+)", filename)
    return (int(mm.group(1)), int(mt.group(1))) if mm and mt else None


def load_all_distributions():
    if not CACHE_DIR.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {CACHE_DIR.resolve()}")
    data = {}
    for path in sorted(CACHE_DIR.glob("*.pkl")):
        key = parse_m_tau(path.name)
        if key is None or key[0] not in M_VALUES or key[1] not in TAU_VALUES:
            continue
        if key in data:
            raise ValueError(f"Archivo duplicado para (m,tau)={key}: {path}")
        df = pd.read_pickle(path)
        if "z" not in df.columns:
            raise ValueError(f"Falta columna 'z' en {path}")
        x = df["z"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if FLIP_Z_SIGN:
            x = -x
        if len(x) < 4:
            print(f"ADVERTENCIA: menos de 4 datos en {path}; se ignora")
            continue
        data[key] = x
    if not data:
        raise FileNotFoundError("No hay archivos .pkl compatibles con M_VALUES y TAU_VALUES")
    missing = [(m, t) for m in M_VALUES for t in TAU_VALUES if (m, t) not in data]
    if missing:
        print("ADVERTENCIA: faltan combinaciones (m,tau):", missing)
    return data


# ============================================================
# EXPERIMENTO 1: MEDIANA FRENTE A TAU
# ============================================================
def make_stats_table(data):
    rows = []
    for (m, tau), x in sorted(data.items()):
        q25, q75 = np.quantile(x, [0.25, 0.75])
        b = q75 - q25
        if b <= 0:
            raise ValueError(f"IQR nulo o negativo para (m,tau)=({m},{tau})")
        rows.append({"m": m, "tau": tau, "n": len(x),
                     "mediana": float(np.median(x)), "IQR": float(b),
                     "std": float(np.std(x, ddof=1))})
    return pd.DataFrame(rows)


def fit_by_m(stats_df):
    """Exponente libre (regresión log-log) y anclado a tau_ref.

    Ajuste libre: log(IQR_tau) = c + alpha_OLS * log(tau/tau_ref).
    Ajuste anclado: log(IQR_tau/IQR_ref) = alpha_ref * log(tau/tau_ref).
    El segundo se utiliza en el colapso de potencia y coincide exactamente
    con la estandarización afín para tau_ref.
    """
    models = {}
    rows = []
    for m, sub in stats_df.groupby("m", sort=True):
        sub = sub.sort_values("tau")
        if len(sub) < 3:
            print(f"ADVERTENCIA: m={m} tiene {len(sub)} tau; se requieren al menos 3 para los ajustes")
            continue
        tau = sub["tau"].to_numpy(float)
        med = sub["mediana"].to_numpy(float)
        b = sub["IQR"].to_numpy(float)
        t_ref, a_ref, b_ref = tau[0], med[0], b[0]
        if np.any(tau <= 0):
            raise ValueError("Los tau deben ser positivos para hacer ajustes log-log")
        u = np.log(tau / t_ref)
        v = np.log(b / b_ref)

        reg = stats.linregress(u, np.log(b))
        alpha_ols = float(reg.slope)
        r2_ols = float(reg.rvalue ** 2) if np.isfinite(reg.rvalue) else np.nan
        alpha_ref = float(np.dot(u, v) / np.dot(u, u))
        err_ref = v - alpha_ref * u
        rmse_ref = float(np.sqrt(np.mean(err_ref ** 2)))
        b_pred = b_ref * (tau / t_ref) ** alpha_ref
        rel_err_max = float(np.max(np.abs(b_pred / b - 1)))

        # Cuantifica desplazamiento de la mediana en unidades de IQR_ref.
        center_drift = float((np.max(med) - np.min(med)) / b_ref)
        center_ref_max = float(np.max(np.abs(med - a_ref)) / b_ref)
        median_slope = float(stats.linregress(tau, med).slope)
        models[m] = {"tau_ref": t_ref, "a_ref": a_ref, "b_ref": b_ref,
                     "alpha_ref": alpha_ref, "alpha_ols": alpha_ols,
                     "r2_ols": r2_ols, "rmse_log_iqr_ref": rmse_ref,
                     "median_drift_IQRref": center_drift}
        rows.append({"m": m, "n_tau": len(tau), "tau_ref": t_ref,
                     "mediana_ref": a_ref, "IQR_ref": b_ref,
                     "pendiente_mediana_vs_tau": median_slope,
                     "rango_mediana_sobre_IQR_ref": center_drift,
                     "max_cambio_mediana_vs_ref_sobre_IQR_ref": center_ref_max,
                     "alpha_OLS": alpha_ols, "R2_loglog_OLS": r2_ols,
                     "alpha_anclado": alpha_ref, "RMSE_log_IQR_anclado": rmse_ref,
                     "max_error_relativo_IQR_anclado": rel_err_max})
    return models, pd.DataFrame(rows)


def plot_medians(stats_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, sub in stats_df.groupby("m", sort=True):
        sub = sub.sort_values("tau")
        ax.plot(sub["tau"], sub["mediana"], "o-", label=fr"$m={m}$")
    ax.set(xlabel=r"Retraso $\tau$", ylabel="Mediana del Z-score de mPE",
           title="Experimento 1: desplazamiento de la mediana")
    ax.set_xticks(sorted(stats_df["tau"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_mediana_vs_tau.png", dpi=250)
    plt.close(fig)


def plot_iqr_law(stats_df, models):
    fig, axs = plt.subplots(len(models), 1, figsize=(7.5, 3.8 * len(models)))
    for ax, (m, model) in zip(np.atleast_1d(axs), models.items()):
        sub = stats_df[stats_df["m"] == m].sort_values("tau")
        tau = sub["tau"].to_numpy(float)
        b = sub["IQR"].to_numpy(float)
        t_dense = np.geomspace(tau.min(), tau.max(), 180)
        b_fit = model["b_ref"] * (t_dense / model["tau_ref"]) ** model["alpha_ref"]
        ax.loglog(tau, b, "o", ms=6, label="IQR observado")
        ax.loglog(t_dense, b_fit, "--", label=fr"Potencia anclada, $\alpha={model['alpha_ref']:.3g}$")
        ax.set(title=fr"$m={m}$; $R^2$ log-log libre = {model['r2_ols']:.3f}",
               xlabel=r"Retraso $\tau$", ylabel="IQR del Z-score")
        ax.set_xticks(tau)
        ax.set_xticklabels([str(int(t)) if t.is_integer() else f"{t:g}" for t in tau])
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_IQR_ley_potencia_loglog.png", dpi=250)
    plt.close(fig)


# ============================================================
# EXPERIMENTO 3: COMPARAR 3 COLAPSOS
# ============================================================
def transform_distributions(data, stats_df, models):
    """Tres modelos, todos con la misma referencia (med_ref, IQR_ref):

    afin: (z-med_tau)/IQR_tau  [ajuste libre por tau]
    potencia_fijo: (z-med_ref)/(IQR_ref*(tau/tau_ref)**alpha_ref)
    potencia_mediana: (z-med_tau)/(IQR_ref*(tau/tau_ref)**alpha_ref)

    El último es un diagnóstico: aisla si el fallo de potencia_fijo se
    debe principalmente al desplazamiento del centro o de la escala.
    """
    result = {name: {} for name in ("afin", "potencia_fijo", "potencia_mediana")}
    index = stats_df.set_index(["m", "tau"])
    for (m, tau), x in data.items():
        if m not in models:
            continue
        med_tau = float(index.loc[(m, tau), "mediana"])
        iqr_tau = float(index.loc[(m, tau), "IQR"])
        model = models[m]
        b_power = model["b_ref"] * (tau / model["tau_ref"]) ** model["alpha_ref"]
        result["afin"][(m, tau)] = (x - med_tau) / iqr_tau
        result["potencia_fijo"][(m, tau)] = (x - model["a_ref"]) / b_power
        result["potencia_mediana"][(m, tau)] = (x - med_tau) / b_power
    return result


def compare_pairwise(transformed):
    rows = []
    for model_name, datasets in transformed.items():
        for m in sorted({k[0] for k in datasets}):
            available_tau = sorted(t for mm, t in datasets if mm == m)
            for t1, t2 in combinations(available_tau, 2):
                x, y = datasets[(m, t1)], datasets[(m, t2)]
                ks = stats.ks_2samp(x, y, method="auto").statistic
                w = stats.wasserstein_distance(x, y)
                rows.append({"modelo": model_name, "m": m, "tau1": t1, "tau2": t2,
                             "KS": float(ks), "Wasserstein": float(w)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (detail.groupby(["m", "modelo"], as_index=False)
               .agg(KS_medio=("KS", "mean"), KS_max=("KS", "max"),
                    W_medio=("Wasserstein", "mean"), W_max=("Wasserstein", "max"),
                    n_pares=("KS", "size")))
    return detail, summary


def kde_curve(x, grid, seed=SEED):
    if len(x) > MAX_KDE_POINTS:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=MAX_KDE_POINTS, replace=False)
    if len(np.unique(x)) < 2:
        return np.full_like(grid, np.nan)
    return stats.gaussian_kde(x)(grid)


def plot_collapses(transformed, models):
    names = [("afin", "Afín: mediana e IQR libres"),
             ("potencia_fijo", "Potencia: centro fijo"),
             ("potencia_mediana", "Potencia + mediana libre")]
    for m in models:
        available_tau = sorted(t for mm, t in transformed["afin"] if mm == m)
        pooled = np.concatenate([transformed[name][(m, t)] for name, _ in names for t in available_tau])
        lo, hi = np.quantile(pooled, [PLOT_QMIN, PLOT_QMAX])
        if hi <= lo:
            print(f"Se omite figura de colapsos para m={m}: rango degenerado")
            continue
        grid = np.linspace(lo, hi, N_GRID)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
        for ax, (name, title) in zip(axs, names):
            for t in available_tau:
                x = transformed[name][(m, t)]
                ax.plot(grid, kde_curve(x, grid, SEED + int(t)), label=fr"$\tau={t}$")
            ax.set(title=title, xlabel="Variable transformada", xlim=(lo, hi))
            ax.grid(alpha=0.25)
        axs[0].set_ylabel("Densidad")
        axs[-1].legend(fontsize=9)
        fig.suptitle(fr"$m={m}$; $\alpha_{{anclado}}={models[m]['alpha_ref']:.3f}$")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"03_colapsos_m{m}.png", dpi=250)
        plt.close(fig)


def plot_distances(summary):
    if summary.empty:
        return
    nice = {"afin": "Afín mediana/IQR", "potencia_fijo": "Potencia: centro fijo",
            "potencia_mediana": "Potencia: mediana libre"}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for model_name, sub in summary.groupby("modelo"):
        sub = sub.sort_values("m")
        axs[0].plot(sub["m"], sub["KS_medio"], "o-", label=nice[model_name])
        axs[1].plot(sub["m"], sub["W_medio"], "o-", label=nice[model_name])
    for ax, metric in zip(axs, ("KS medio", "Wasserstein medio")):
        ax.set(xlabel="Dimensión m", ylabel=metric)
        ax.set_xticks(sorted(summary["m"].unique()))
        ax.grid(alpha=0.3)
    axs[1].legend(fontsize=9)
    fig.suptitle("Experimento 3: distancias entre todos los pares de retrasos")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_comparacion_KS_Wasserstein.png", dpi=250)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_distributions()
    stats_df = make_stats_table(data)
    models, fits_df = fit_by_m(stats_df)
    if not models:
        raise ValueError("No hay valores de m con al menos tres retrasos disponibles")

    stats_df.to_csv(OUT_DIR / "estadisticos_por_m_tau.csv", index=False)
    fits_df.to_csv(OUT_DIR / "ajustes_mediana_y_potencia_por_m.csv", index=False)
    plot_medians(stats_df)
    plot_iqr_law(stats_df, models)

    transformed = transform_distributions(data, stats_df, models)
    pair_df, summary_df = compare_pairwise(transformed)
    pair_df.to_csv(OUT_DIR / "comparaciones_pares_tres_modelos.csv", index=False)
    summary_df.to_csv(OUT_DIR / "resumen_comparacion_modelos.csv", index=False)
    plot_collapses(transformed, models)
    plot_distances(summary_df)

    print("\n=== EXPERIMENTOS 1 Y 2: MEDIANA Y LEY DE POTENCIA ===")
    columns = ["m", "rango_mediana_sobre_IQR_ref", "alpha_OLS", "R2_loglog_OLS",
               "alpha_anclado", "RMSE_log_IQR_anclado"]
    print(fits_df[columns].to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("\n=== EXPERIMENTO 3: COMPARACION DE COLAPSOS ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\nCSV y figuras guardados en: {OUT_DIR.resolve()}")
    print("NOTA: n_tau=3 o 4 es insuficiente para afirmar una ley de potencia robusta.")
    print("KS y Wasserstein son descriptivos; los parámetros de colapso se estiman con los datos.")


if __name__ == "__main__":
    main()
