# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# =========================================================
# CONFIGURACIÓN
# =========================================================
EXPORT_DIR = "csv_medidas_paneles"   # carpeta donde guardaste los CSV
MIN_PAIRS = 3                        # mínimo de pares para calcular correlación

# =========================================================
# AUXILIARES
# =========================================================
def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', str(text))]

def sanitize_panel_name(filename):
    """
    Convierte el nombre del archivo en una etiqueta legible del panel.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    return name

def load_panel_csvs(export_dir):
    """
    Lee todos los CSV de medidas guardados en export_dir.

    Retorna
    -------
    panel_dict : dict
        panel_dict[nombre_panel] = DataFrame
        índice: Serie_i
        columnas: compositores
        valores: medida observada (pe_obs)
    """
    if not os.path.isdir(export_dir):
        raise FileNotFoundError(f"No existe la carpeta: {export_dir}")

    files = [f for f in os.listdir(export_dir) if f.lower().endswith("csv")]
    files = sorted(files, key=natural_key)

    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron CSV en: {export_dir}")

    panel_dict = {}

    for fname in files:
        path = os.path.join(export_dir, fname)
        df = pd.read_csv(path, index_col=0)

        # Asegurar que todo sea numérico
        df = df.apply(pd.to_numeric, errors="coerce")

        # Ordenar índice y columnas de forma natural
        df = df.sort_index(key=lambda idx: [natural_key(x) for x in idx])
        df = df.reindex(columns=sorted(df.columns, key=natural_key))

        panel_name = sanitize_panel_name(fname)
        panel_dict[panel_name] = df

    return panel_dict

def panel_to_long(df, panel_name):
    """
    Convierte un DataFrame ancho (Series x Compositores) a largo:
    columnas = [serie, composer, value]
    """
    long_df = (
        df.copy()
        .rename_axis("serie")
        .reset_index()
        .melt(id_vars="serie", var_name="composer", value_name="value")
    )
    long_df["panel"] = panel_name
    return long_df

def correlation_summary(x, y, min_pairs=MIN_PAIRS):
    """
    Calcula Pearson y Spearman con limpieza de NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < min_pairs:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan
        }

    # Si una variable es constante, scipy puede fallar o devolver nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan
        }

    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)

    return {
        "n": n,
        "pearson_r": rp,
        "pearson_p": pp,
        "spearman_rho": rs,
        "spearman_p": ps
    }

def compute_global_correlations(panel_dict, min_pairs=MIN_PAIRS):
    """
    Calcula correlaciones globales entre todos los pares de paneles.

    Global = usar todas las observaciones emparejadas por (composer, serie).
    """
    panel_names = list(panel_dict.keys())
    long_dict = {name: panel_to_long(df, name) for name, df in panel_dict.items()}

    rows = []

    for i in range(len(panel_names)):
        for j in range(i + 1, len(panel_names)):
            p1 = panel_names[i]
            p2 = panel_names[j]

            df1 = long_dict[p1][["serie", "composer", "value"]].rename(columns={"value": "value_1"})
            df2 = long_dict[p2][["serie", "composer", "value"]].rename(columns={"value": "value_2"})

            merged = pd.merge(df1, df2, on=["serie", "composer"], how="inner")

            stats = correlation_summary(
                merged["value_1"].to_numpy(),
                merged["value_2"].to_numpy(),
                min_pairs=min_pairs
            )

            rows.append({
                "panel_1": p1,
                "panel_2": p2,
                **stats
            })

    return pd.DataFrame(rows)

def compute_correlations_by_composer(panel_dict, min_pairs=MIN_PAIRS):
    """
    Calcula correlaciones por compositor entre todos los pares de paneles.

    Para cada compositor, correlaciona las series comunes entre dos paneles.
    """
    panel_names = list(panel_dict.keys())
    all_composers = sorted(
        set().union(*[set(df.columns) for df in panel_dict.values()]),
        key=natural_key
    )

    rows = []

    for composer in all_composers:
        for i in range(len(panel_names)):
            for j in range(i + 1, len(panel_names)):
                p1 = panel_names[i]
                p2 = panel_names[j]

                df1 = panel_dict[p1]
                df2 = panel_dict[p2]

                if composer not in df1.columns or composer not in df2.columns:
                    continue

                s1 = df1[composer].rename("value_1")
                s2 = df2[composer].rename("value_2")

                merged = pd.concat([s1, s2], axis=1, join="inner").reset_index()
                merged = merged.rename(columns={"index": "serie"})

                stats = correlation_summary(
                    merged["value_1"].to_numpy(),
                    merged["value_2"].to_numpy(),
                    min_pairs=min_pairs
                )

                rows.append({
                    "composer": composer,
                    "panel_1": p1,
                    "panel_2": p2,
                    **stats
                })

    return pd.DataFrame(rows)

def print_global_results(df_global):
    print("\n" + "="*90)
    print("CORRELACIONES GLOBALES")
    print("="*90)

    if df_global.empty:
        print("No hay resultados globales.")
        return

    for _, row in df_global.iterrows():
        print(f"\n{row['panel_1']}  vs  {row['panel_2']}")
        print(f"  n pares        = {int(row['n'])}")
        print(f"  Pearson  r     = {row['pearson_r']:.6f}   p = {row['pearson_p']:.6g}")
        print(f"  Spearman rho   = {row['spearman_rho']:.6f}   p = {row['spearman_p']:.6g}")

def print_results_by_composer(df_by_composer):
    print("\n" + "="*90)
    print("CORRELACIONES POR COMPOSITOR")
    print("="*90)

    if df_by_composer.empty:
        print("No hay resultados por compositor.")
        return

    composers = sorted(df_by_composer["composer"].unique(), key=natural_key)

    for composer in composers:
        sub = df_by_composer[df_by_composer["composer"] == composer].copy()

        print("\n" + "-"*90)
        print(f"COMPOSITOR: {composer}")
        print("-"*90)

        for _, row in sub.iterrows():
            print(f"{row['panel_1']}  vs  {row['panel_2']}")
            print(f"  n pares        = {int(row['n'])}")
            print(f"  Pearson  r     = {row['pearson_r']:.6f}   p = {row['pearson_p']:.6g}")
            print(f"  Spearman rho   = {row['spearman_rho']:.6f}   p = {row['spearman_p']:.6g}")
            print()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    panel_dict = load_panel_csvs(EXPORT_DIR)

    # print("Paneles detectados:")
    # for name in panel_dict:
    #     df = panel_dict[name]
    #     print(f"  - {name}: shape = {df.shape}")

    df_global = compute_global_correlations(panel_dict, min_pairs=MIN_PAIRS)
    df_by_composer = compute_correlations_by_composer(panel_dict, min_pairs=MIN_PAIRS)

    print_global_results(df_global)
    # print_results_by_composer(df_by_composer)

    # Guardado opcional
    # df_global.to_csv("correlaciones_globales_paneles.csv", index=False, encoding="utf-8")
    # df_by_composer.to_csv("correlaciones_por_compositor_paneles.csv", index=False, encoding="utf-8")