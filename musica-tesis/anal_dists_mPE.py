# -*- coding: utf-8 -*-
"""
Analisis definitivo de Z-scores de mPE ya calculados y guardados en cache_zscore.

Suposiciones:
- Ya existen los pkl generados por tu script previo.
- Cada fila del pkl corresponde a una pieza ("serie") bajo una condicion.
- Cada pieza aporta un Z-score.
- Las comparaciones entre nulos y entre representaciones se hacen de forma PAREADA
  usando (composer, serie).

Salidas:
- CSVs con tablas de resumen y matrices de distancias.
- Impresion en consola de resultados resumidos listos para inspeccion.
"""

import os
import re
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon, binomtest

# =========================================================
# CONFIGURACION
# =========================================================

CACHE_DIR = "cache_zscore"
OUTPUT_DIR = "analysis_definitivo_zscores"

# Deben coincidir con los parametros usados al guardar los cache .pkl
NORMALIZE = True
ALTERNATIVE = "less"

# Bootstrap para razones de compresion
N_BOOT = 2000
BOOT_CI = 95
BOOT_RANDOM_STATE = 123456

# Cuantos pares de compositores imprimir como mas cercanos / lejanos
TOP_K_PAIRS = 10

# Si quieres silencio casi total, pon False
VERBOSE = True

# Las 4 condiciones del esquema definitivo
PANEL_CONFIGS = [
    {
        "key": "notes_shuffle",
        "measure": "mPE",
        "D": 5,
        "tau": 1,
        "type_null": "shuffle",
        "representation": "MIDI",
        "null_model": "shuffle",
        "title": "notes + shuffle",
    },
    {
        "key": "intervals_shuffle",
        "measure": "mPE_interval",
        "D": 5,
        "tau": 1,
        "type_null": "shuffle",
        "representation": "intervals",
        "null_model": "shuffle",
        "title": "intervals + shuffle",
    },
    {
        "key": "notes_iaaft",
        "measure": "mPE",
        "D": 5,
        "tau": 1,
        "type_null": "iaaft",
        "representation": "MIDI",
        "null_model": "iaaft",
        "title": "notes + IAAFT",
    },
    {
        "key": "intervals_iaaft",
        "measure": "mPE_interval",
        "D": 5,
        "tau": 1,
        "type_null": "iaaft",
        "representation": "intervals",
        "null_model": "iaaft",
        "title": "intervals + IAAFT",
    },
]


# =========================================================
# UTILIDADES BASICAS
# =========================================================

def natural_key(text):
    return tuple(
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", str(text))
    )


def build_cache_key(measure, D, tau, type_null, normalize, alternative):
    # Replica la logica de tu script original
    if "PE" in measure:
        return f"{measure}_D{D}_tau{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    elif "Cd" in measure:
        return f"{measure}_d{D}_mu{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    else:
        return f"{measure}_{type_null}_{alternative}_norm{int(normalize)}"


def get_cache_path(cache_dir, measure, D, tau, type_null, normalize, alternative):
    fname = build_cache_key(measure, D, tau, type_null, normalize, alternative) + ".pkl"
    return os.path.join(cache_dir, fname)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_scale(x):
    """
    Escala robusta para estandarizacion:
    1) MAD
    2) IQR / 1.349
    3) std
    4) 1.0 si todo falla
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return 1.0

    s = mad(x)
    if np.isfinite(s) and s > 0:
        return s

    s = iqr(x)
    if np.isfinite(s) and s > 0:
        return s / 1.349

    if x.size >= 2:
        s = np.std(x, ddof=1)
        if np.isfinite(s) and s > 0:
            return float(s)

    return 1.0


def robust_standardize(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    med = np.median(x)
    scale = robust_scale(x)
    return (x - med) / scale


def safe_log(x, eps=1e-12):
    if not np.isfinite(x):
        return np.nan
    return np.log(max(float(x), eps))


def pretty_float(x, nd=4):
    if pd.isna(x):
        return "NaN"
    return f"{x:.{nd}f}"


def summary_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
        "iqr": iqr(x),
        "mad": mad(x),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def paired_bootstrap_ratio(x, y, stat="iqr", n_boot=2000, ci=95, random_state=1234):
    """
    Bootstrap pareado por indices (mismas piezas re-muestreadas juntas).
    Devuelve ratio puntual, CI inferior, CI superior.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.nan, np.nan, np.nan

    if stat == "iqr":
        stat_func = iqr
    elif stat == "mad":
        stat_func = mad
    else:
        raise ValueError("stat debe ser 'iqr' o 'mad'")

    sx = stat_func(x)
    sy = stat_func(y)

    if not np.isfinite(sx) or not np.isfinite(sy) or sy <= 0:
        return np.nan, np.nan, np.nan

    point = sx / sy

    if x.size < 2:
        return point, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    n = x.size
    ratios = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        sxb = stat_func(xb)
        syb = stat_func(yb)

        if np.isfinite(sxb) and np.isfinite(syb) and syb > 0:
            ratios.append(sxb / syb)

    if len(ratios) == 0:
        return point, np.nan, np.nan

    alpha = 100 - ci
    lo = np.percentile(ratios, alpha / 2)
    hi = np.percentile(ratios, 100 - alpha / 2)

    return point, float(lo), float(hi)


def paired_tests(diffs):
    """
    Contrastes pareados:
    - Wilcoxon signed-rank
    - prueba de signos (binomial)
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]

    out = {
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
        "sign_p": np.nan,
        "n_pos": 0,
        "n_neg": 0,
        "n_zero": 0,
    }

    if diffs.size == 0:
        return out

    out["n_pos"] = int(np.sum(diffs > 0))
    out["n_neg"] = int(np.sum(diffs < 0))
    out["n_zero"] = int(np.sum(diffs == 0))

    # Wilcoxon
    nz = diffs[diffs != 0]
    if nz.size >= 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
            out["wilcoxon_stat"] = float(stat)
            out["wilcoxon_p"] = float(p)
        except Exception:
            out["wilcoxon_stat"] = np.nan
            out["wilcoxon_p"] = np.nan

    # Prueba de signos
    n_eff = out["n_pos"] + out["n_neg"]
    if n_eff >= 1:
        try:
            bt = binomtest(out["n_pos"], n=n_eff, p=0.5, alternative="two-sided")
            out["sign_p"] = float(bt.pvalue)
        except Exception:
            out["sign_p"] = np.nan

    return out


def compression_winner(ratio, label_a, label_b):
    if not np.isfinite(ratio):
        return "indeterminado"
    if ratio < 1:
        return label_a
    if ratio > 1:
        return label_b
    return "igual"


# =========================================================
# CARGA DE PANELES DESDE CACHE
# =========================================================

def load_panel_from_cache(cfg):
    path = get_cache_path(
        cache_dir=CACHE_DIR,
        measure=cfg["measure"],
        D=cfg["D"],
        tau=cfg["tau"],
        type_null=cfg["type_null"],
        normalize=NORMALIZE,
        alternative=ALTERNATIVE,
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontro el cache:\n{path}\n"
            "Verifica CACHE_DIR, D, tau, NORMALIZE, ALTERNATIVE y type_null."
        )

    df = pd.read_pickle(path).copy()
    required = {
        "composer", "birth_year", "composer_index", "serie",
        "length", "z", "p_value", "p_raw",
        "pe_obs", "mu_null", "sigma_null"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"El cache {path} no tiene las columnas esperadas. Faltan: {sorted(missing)}"
        )

    df["condition_key"] = cfg["key"]
    df["representation"] = cfg["representation"]
    df["null_model"] = cfg["null_model"]
    df["condition_title"] = cfg["title"]

    if VERBOSE:
        n_comp = df["composer"].nunique()
        n_series = len(df)
        print(f"[LOAD] {cfg['key']:>18s} | compositores={n_comp:2d} | filas={n_series:4d} | {path}")

    return df


def build_master_piece_table(panel_dict):
    """
    Construye una tabla ancha por pieza:
    una fila = (composer, serie)
    columnas = z_notes_shuffle, z_intervals_shuffle, z_notes_iaaft, z_intervals_iaaft, ...
    """
    pieces = []

    for key, df in panel_dict.items():
        sub = df[
            [
                "composer", "birth_year", "composer_index", "serie", "length",
                "z", "p_value", "p_raw", "pe_obs", "mu_null", "sigma_null"
            ]
        ].copy()

        rename_map = {
            "z": f"z_{key}",
            "p_value": f"p_value_{key}",
            "p_raw": f"p_raw_{key}",
            "pe_obs": f"pe_obs_{key}",
            "mu_null": f"mu_null_{key}",
            "sigma_null": f"sigma_null_{key}",
            "length": f"length_{key}",
        }
        sub = sub.rename(columns=rename_map)
        pieces.append(sub)

    merge_keys = ["composer", "birth_year", "composer_index", "serie"]
    master = reduce(
        lambda left, right: pd.merge(left, right, on=merge_keys, how="outer"),
        pieces
    )

    # ordenar
    master["birth_year_num"] = pd.to_numeric(master["birth_year"], errors="coerce")
    master["composer_index_num"] = pd.to_numeric(master["composer_index"], errors="coerce")

    master = master.sort_values(
        by=["composer_index_num", "birth_year_num", "composer", "serie"],
        key=lambda col: col.map(natural_key) if col.name in ["composer", "serie"] else col
    ).reset_index(drop=True)

    return master


def get_ordered_composers(master):
    meta = (
        master[["composer", "birth_year", "composer_index"]]
        .drop_duplicates()
        .copy()
    )
    meta["birth_year_num"] = pd.to_numeric(meta["birth_year"], errors="coerce")
    meta["composer_index_num"] = pd.to_numeric(meta["composer_index"], errors="coerce")

    meta = meta.sort_values(
        by=["composer_index_num", "birth_year_num", "composer"],
        key=lambda col: col.map(natural_key) if col.name == "composer" else col
    )

    return meta["composer"].tolist(), meta[["composer", "birth_year", "composer_index"]].reset_index(drop=True)


# =========================================================
# RESUMENES POR COMPOSITOR Y CONDICION
# =========================================================

def composer_condition_summary(master, composer_order):
    rows = []

    condition_keys = [cfg["key"] for cfg in PANEL_CONFIGS]

    for composer in composer_order:
        sub = master[master["composer"] == composer].copy()

        base = {
            "composer": composer,
            "birth_year": sub["birth_year"].iloc[0] if len(sub) else np.nan,
            "composer_index": sub["composer_index"].iloc[0] if len(sub) else np.nan,
        }

        for key in condition_keys:
            x = sub[f"z_{key}"].to_numpy(dtype=float) if f"z_{key}" in sub else np.array([], dtype=float)
            st = summary_stats(x)

            base[f"n_{key}"] = st["n"]
            base[f"median_{key}"] = st["median"]
            base[f"iqr_{key}"] = st["iqr"]
            base[f"mad_{key}"] = st["mad"]
            base[f"mean_{key}"] = st["mean"]
            base[f"std_{key}"] = st["std"]
            base[f"min_{key}"] = st["min"]
            base[f"max_{key}"] = st["max"]

        rows.append(base)

    return pd.DataFrame(rows)


# =========================================================
# COMPARACIONES PAREADAS POR COMPOSITOR
# =========================================================

def paired_contrast_by_composer(
    master,
    composer_order,
    col_a,
    col_b,
    label_a,
    label_b,
    contrast_name,
    n_boot=N_BOOT,
    boot_ci=BOOT_CI,
    boot_seed=BOOT_RANDOM_STATE,
):
    rows = []

    for ic, composer in enumerate(composer_order):
        sub = master[master["composer"] == composer][
            ["composer", "birth_year", "composer_index", "serie", col_a, col_b]
        ].copy()

        sub = sub.dropna(subset=[col_a, col_b]).sort_values("serie", key=lambda s: s.map(natural_key))

        x = sub[col_a].to_numpy(dtype=float)
        y = sub[col_b].to_numpy(dtype=float)
        diffs = x - y

        stx = summary_stats(x)
        sty = summary_stats(y)

        iqr_ratio, iqr_lo, iqr_hi = paired_bootstrap_ratio(
            x, y, stat="iqr", n_boot=n_boot, ci=boot_ci, random_state=boot_seed + 1000 * ic + 1
        )
        mad_ratio, mad_lo, mad_hi = paired_bootstrap_ratio(
            x, y, stat="mad", n_boot=n_boot, ci=boot_ci, random_state=boot_seed + 1000 * ic + 2
        )

        w1 = np.nan
        w1_shape = np.nan
        if len(x) > 0 and len(y) > 0:
            w1 = float(wasserstein_distance(x, y))
            x_std = robust_standardize(x)
            y_std = robust_standardize(y)
            w1_shape = float(wasserstein_distance(x_std, y_std))

        tests = paired_tests(diffs)

        row = {
            "contrast": contrast_name,
            "composer": composer,
            "birth_year": sub["birth_year"].iloc[0] if len(sub) else np.nan,
            "composer_index": sub["composer_index"].iloc[0] if len(sub) else np.nan,
            "n_pairs": int(len(sub)),

            f"median_{label_a}": stx["median"],
            f"median_{label_b}": sty["median"],
            "median_diff_a_minus_b": float(np.median(diffs)) if len(diffs) else np.nan,

            f"iqr_{label_a}": stx["iqr"],
            f"iqr_{label_b}": sty["iqr"],
            "iqr_ratio_a_over_b": iqr_ratio,
            "iqr_ratio_ci_low": iqr_lo,
            "iqr_ratio_ci_high": iqr_hi,
            "iqr_more_compressed": compression_winner(iqr_ratio, label_a, label_b),

            f"mad_{label_a}": stx["mad"],
            f"mad_{label_b}": sty["mad"],
            "mad_ratio_a_over_b": mad_ratio,
            "mad_ratio_ci_low": mad_lo,
            "mad_ratio_ci_high": mad_hi,
            "mad_more_compressed": compression_winner(mad_ratio, label_a, label_b),

            "wasserstein_w1": w1,
            "wasserstein_shape_w1": w1_shape,

            "wilcoxon_stat": tests["wilcoxon_stat"],
            "wilcoxon_p": tests["wilcoxon_p"],
            "sign_p": tests["sign_p"],
            "n_pos_diff": tests["n_pos"],
            "n_neg_diff": tests["n_neg"],
            "n_zero_diff": tests["n_zero"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["composer_index", "birth_year", "composer"]).reset_index(drop=True)
    return df


def print_compact_contrast_table(df, title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    cols_show = [
        "composer",
        "n_pairs",
        "median_diff_a_minus_b",
        "iqr_ratio_a_over_b",
        "mad_ratio_a_over_b",
        "wasserstein_w1",
        "wasserstein_shape_w1",
        "wilcoxon_p",
        "sign_p",
        "iqr_more_compressed",
        "mad_more_compressed",
    ]
    print(df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_contrast_aggregate(df, label_a, label_b, title):
    print("\n" + "-" * 110)
    print(f"Resumen agregado: {title}")
    print("-" * 110)

    valid_iqr = df["iqr_ratio_a_over_b"].dropna()
    valid_mad = df["mad_ratio_a_over_b"].dropna()
    valid_w1 = df["wasserstein_w1"].dropna()
    valid_ws = df["wasserstein_shape_w1"].dropna()

    n_iqr_a_comp = int((valid_iqr < 1).sum())
    n_iqr_b_comp = int((valid_iqr > 1).sum())

    n_mad_a_comp = int((valid_mad < 1).sum())
    n_mad_b_comp = int((valid_mad > 1).sum())

    n_wilcox_sig = int((df["wilcoxon_p"] < 0.05).sum())
    n_sign_sig = int((df["sign_p"] < 0.05).sum())

    print(f"Compositores donde {label_a} esta mas comprimido que {label_b} segun IQR : {n_iqr_a_comp}")
    print(f"Compositores donde {label_b} esta mas comprimido que {label_a} segun IQR : {n_iqr_b_comp}")
    print(f"Mediana de IQR ratio ({label_a}/{label_b})                         : {pretty_float(valid_iqr.median())}")

    print(f"Compositores donde {label_a} esta mas comprimido que {label_b} segun MAD : {n_mad_a_comp}")
    print(f"Compositores donde {label_b} esta mas comprimido que {label_a} segun MAD : {n_mad_b_comp}")
    print(f"Mediana de MAD ratio ({label_a}/{label_b})                         : {pretty_float(valid_mad.median())}")

    print(f"Mediana de Wasserstein W1                                           : {pretty_float(valid_w1.median())}")
    print(f"Mediana de Wasserstein de forma                                     : {pretty_float(valid_ws.median())}")
    print(f"Compositores con Wilcoxon p < 0.05                                  : {n_wilcox_sig}")
    print(f"Compositores con Sign test p < 0.05                                 : {n_sign_sig}")


# =========================================================
# MATRICES ENTRE COMPOSITORES
# =========================================================

def composer_pairwise_matrices(master, composer_order, zcol):
    """
    Devuelve:
    - matriz W1 entre compositores
    - matriz de compresion basada en |log(IQR_i)-log(IQR_j)|
    - tabla larga con todos los pares
    """
    n = len(composer_order)
    D = np.full((n, n), np.nan, dtype=float)
    C = np.full((n, n), np.nan, dtype=float)

    series_by_comp = {}
    iqr_by_comp = {}
    mad_by_comp = {}

    for composer in composer_order:
        x = master.loc[master["composer"] == composer, zcol].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        series_by_comp[composer] = x
        iqr_by_comp[composer] = iqr(x)
        mad_by_comp[composer] = mad(x)

    pair_rows = []

    for i, ci in enumerate(composer_order):
        xi = series_by_comp[ci]
        D[i, i] = 0.0
        C[i, i] = 0.0

        for j in range(i + 1, n):
            cj = composer_order[j]
            xj = series_by_comp[cj]

            d = np.nan
            if len(xi) > 0 and len(xj) > 0:
                d = float(wasserstein_distance(xi, xj))

            c = np.abs(safe_log(iqr_by_comp[ci]) - safe_log(iqr_by_comp[cj]))

            D[i, j] = D[j, i] = d
            C[i, j] = C[j, i] = c

            pair_rows.append({
                "composer_i": ci,
                "composer_j": cj,
                "wasserstein_w1": d,
                "compression_logiqr_distance": c,
                "iqr_i": iqr_by_comp[ci],
                "iqr_j": iqr_by_comp[cj],
                "mad_i": mad_by_comp[ci],
                "mad_j": mad_by_comp[cj],
            })

    D_df = pd.DataFrame(D, index=composer_order, columns=composer_order)
    C_df = pd.DataFrame(C, index=composer_order, columns=composer_order)
    pairs_df = pd.DataFrame(pair_rows).sort_values("wasserstein_w1", ascending=True).reset_index(drop=True)

    return D_df, C_df, pairs_df


def print_top_pairs(pairs_df, title, metric_col, top_k=10, ascending=True):
    if pairs_df.empty:
        return

    direction = "mas cercanos" if ascending else "mas lejanos"
    print("\n" + "-" * 110)
    print(f"{title} | pares {direction} segun {metric_col}")
    print("-" * 110)

    sub = pairs_df.sort_values(metric_col, ascending=ascending).head(top_k).copy()
    cols = ["composer_i", "composer_j", metric_col, "iqr_i", "iqr_j"]
    print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =========================================================
# EXPORTACION
# =========================================================

def save_csv(df, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=True if isinstance(df.index, pd.Index) and df.index.name is not None else False, encoding="utf-8")
    return path


def save_csv_noindex(df, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUTPUT_DIR)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    # -----------------------------------------------------
    # 1) Cargar paneles
    # -----------------------------------------------------
    panel_dict = {}
    for cfg in PANEL_CONFIGS:
        panel_dict[cfg["key"]] = load_panel_from_cache(cfg)

    # -----------------------------------------------------
    # 2) Tabla maestra por pieza
    # -----------------------------------------------------
    master = build_master_piece_table(panel_dict)
    composer_order, composer_meta = get_ordered_composers(master)

    save_csv_noindex(master, "master_piece_table.csv")
    save_csv_noindex(composer_meta, "composer_meta.csv")

    print("\n" + "=" * 110)
    print("TABLA MAESTRA POR PIEZA")
    print("=" * 110)
    print(f"Filas (pieza): {len(master)}")
    print(f"Compositores:  {len(composer_order)}")
    print("Columnas de Z disponibles:")
    for cfg in PANEL_CONFIGS:
        print(f"  - z_{cfg['key']}")

    # -----------------------------------------------------
    # 3) Resumen por compositor y condicion
    # -----------------------------------------------------
    summary_df = composer_condition_summary(master, composer_order)
    save_csv_noindex(summary_df, "composer_condition_summary.csv")

    print("\n" + "=" * 110)
    print("RESUMEN POR COMPOSITOR Y CONDICION")
    print("=" * 110)

    cols_to_show = ["composer", "birth_year"]
    for key in [cfg["key"] for cfg in PANEL_CONFIGS]:
        cols_to_show += [f"n_{key}", f"median_{key}", f"iqr_{key}", f"mad_{key}"]

    print(summary_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # -----------------------------------------------------
    # 4) Contrastes pareados: modelos nulos
    # -----------------------------------------------------
    null_notes_df = paired_contrast_by_composer(
        master=master,
        composer_order=composer_order,
        col_a="z_notes_iaaft",
        col_b="z_notes_shuffle",
        label_a="IAAFT",
        label_b="shuffle",
        contrast_name="null_models_within_notes",
    )
    save_csv_noindex(null_notes_df, "contrast_null_models_within_notes.csv")
    print_compact_contrast_table(null_notes_df, "COMPARACION DE MODELOS NULOS DENTRO DE MIDI (IAAFT vs shuffle)")
    print_contrast_aggregate(null_notes_df, "IAAFT", "shuffle", "Modelos nulos dentro de MIDI")

    null_intervals_df = paired_contrast_by_composer(
        master=master,
        composer_order=composer_order,
        col_a="z_intervals_iaaft",
        col_b="z_intervals_shuffle",
        label_a="IAAFT",
        label_b="shuffle",
        contrast_name="null_models_within_intervals",
    )
    save_csv_noindex(null_intervals_df, "contrast_null_models_within_intervals.csv")
    print_compact_contrast_table(null_intervals_df, "COMPARACION DE MODELOS NULOS DENTRO DE INTERVALOS (IAAFT vs shuffle)")
    print_contrast_aggregate(null_intervals_df, "IAAFT", "shuffle", "Modelos nulos dentro de intervalos")

    # -----------------------------------------------------
    # 5) Contrastes pareados: representaciones
    # -----------------------------------------------------
    rep_iaaft_df = paired_contrast_by_composer(
        master=master,
        composer_order=composer_order,
        col_a="z_notes_iaaft",
        col_b="z_intervals_iaaft",
        label_a="MIDI",
        label_b="intervals",
        contrast_name="representations_within_iaaft",
    )
    save_csv_noindex(rep_iaaft_df, "contrast_representations_within_iaaft.csv")
    print_compact_contrast_table(rep_iaaft_df, "COMPARACION DE REPRESENTACIONES DENTRO DE IAAFT (MIDI vs intervals)")
    print_contrast_aggregate(rep_iaaft_df, "MIDI", "intervals", "Representaciones dentro de IAAFT")

    rep_shuffle_df = paired_contrast_by_composer(
        master=master,
        composer_order=composer_order,
        col_a="z_notes_shuffle",
        col_b="z_intervals_shuffle",
        label_a="MIDI",
        label_b="intervals",
        contrast_name="representations_within_shuffle",
    )
    save_csv_noindex(rep_shuffle_df, "contrast_representations_within_shuffle.csv")
    print_compact_contrast_table(rep_shuffle_df, "COMPARACION DE REPRESENTACIONES DENTRO DE SHUFFLE (MIDI vs intervals)")
    print_contrast_aggregate(rep_shuffle_df, "MIDI", "intervals", "Representaciones dentro de shuffle")

    # -----------------------------------------------------
    # 6) Matrices entre compositores por condicion
    # -----------------------------------------------------
    print("\n" + "=" * 110)
    print("COMPARACIONES ENTRE COMPOSITORES")
    print("=" * 110)

    for cfg in PANEL_CONFIGS:
        zcol = f"z_{cfg['key']}"
        D_df, C_df, pairs_df = composer_pairwise_matrices(master, composer_order, zcol)

        save_csv(D_df, f"composer_wasserstein_matrix_{cfg['key']}.csv")
        save_csv(C_df, f"composer_compression_matrix_{cfg['key']}.csv")
        save_csv_noindex(pairs_df, f"composer_pairs_{cfg['key']}.csv")

        print("\n" + "=" * 110)
        print(f"CONDICION: {cfg['title']}")
        print("=" * 110)

        print_top_pairs(
            pairs_df, title=cfg["title"], metric_col="wasserstein_w1",
            top_k=TOP_K_PAIRS, ascending=True
        )
        print_top_pairs(
            pairs_df, title=cfg["title"], metric_col="wasserstein_w1",
            top_k=TOP_K_PAIRS, ascending=False
        )
        print_top_pairs(
            pairs_df, title=cfg["title"], metric_col="compression_logiqr_distance",
            top_k=TOP_K_PAIRS, ascending=True
        )
        print_top_pairs(
            pairs_df, title=cfg["title"], metric_col="compression_logiqr_distance",
            top_k=TOP_K_PAIRS, ascending=False
        )

    # -----------------------------------------------------
    # 7) Resumen muy compacto final
    # -----------------------------------------------------
    compact_global = pd.DataFrame({
        "contrast": [
            "null_models_within_notes",
            "null_models_within_intervals",
            "representations_within_iaaft",
            "representations_within_shuffle",
        ],
        "median_w1": [
            null_notes_df["wasserstein_w1"].median(),
            null_intervals_df["wasserstein_w1"].median(),
            rep_iaaft_df["wasserstein_w1"].median(),
            rep_shuffle_df["wasserstein_w1"].median(),
        ],
        "median_shape_w1": [
            null_notes_df["wasserstein_shape_w1"].median(),
            null_intervals_df["wasserstein_shape_w1"].median(),
            rep_iaaft_df["wasserstein_shape_w1"].median(),
            rep_shuffle_df["wasserstein_shape_w1"].median(),
        ],
        "median_iqr_ratio_a_over_b": [
            null_notes_df["iqr_ratio_a_over_b"].median(),
            null_intervals_df["iqr_ratio_a_over_b"].median(),
            rep_iaaft_df["iqr_ratio_a_over_b"].median(),
            rep_shuffle_df["iqr_ratio_a_over_b"].median(),
        ],
        "median_mad_ratio_a_over_b": [
            null_notes_df["mad_ratio_a_over_b"].median(),
            null_intervals_df["mad_ratio_a_over_b"].median(),
            rep_iaaft_df["mad_ratio_a_over_b"].median(),
            rep_shuffle_df["mad_ratio_a_over_b"].median(),
        ],
        "n_wilcoxon_sig_p_lt_0_05": [
            int((null_notes_df["wilcoxon_p"] < 0.05).sum()),
            int((null_intervals_df["wilcoxon_p"] < 0.05).sum()),
            int((rep_iaaft_df["wilcoxon_p"] < 0.05).sum()),
            int((rep_shuffle_df["wilcoxon_p"] < 0.05).sum()),
        ],
    })
    save_csv_noindex(compact_global, "compact_global_summary.csv")

    print("\n" + "=" * 110)
    print("RESUMEN GLOBAL MUY COMPACTO")
    print("=" * 110)
    print(compact_global.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 110)
    print(f"Archivos guardados en: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 110)


if __name__ == "__main__":
    main()