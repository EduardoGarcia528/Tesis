# -*- coding: utf-8 -*-
"""
sacar_gammas_dists_dataset.py

Versión de sacar_gammas.py adaptada al conjunto de datos usado por dists.py.

- Carga:
    data/composers_hybrid.pkl
    data/datos_composers_hybrid.pkl
  y/o
    data/composers_complexity.pkl
    data/datos_composers_complexity.pkl

- Calcula, en una sola corrida por METHOD, todos:
    C_1, ..., C_{MAX_GAMMA+1}
    gamma_1, ..., gamma_{MAX_GAMMA}

- Guarda PKL con nombres compatibles con el esquema de dists.py, añadiendo METHOD:
    cache_zscore/Cd_rank_d{d}_mu{mu}_{type_null}_{alternative}_{method}.pkl
    cache_zscore/gamma_rank_ties_d{d}_mu{mu}_{type_null}_{alternative}_{method}.pkl

- Guarda CSV con el mismo esquema de nombres usado por save_panel_measure_csv en dists.py:
    csv_medidas_paneles/Cd_rank_D{d}_tau{mu}_{type_null}_{method}.csv
    csv_medidas_paneles/gamma_rank_ties_D{d}_tau{mu}_{type_null}_{method}.csv

Notas:
- Por defecto VALUE_MODE="raw" guarda C_d y gamma_d directamente.
- Si quieres reproducir la convención de algunos paneles antiguos de dists.py,
  usa VALUE_MODE="one_minus" para guardar 1-C_d y 1-gamma_d.
- Por defecto los CSV exportan la columna "z", igual que dists.py.
"""

import os
import re
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import mi_libreria as ml


# =========================================================
# PARÁMETROS GENERALES
# =========================================================

DATA_DIR = r"data"
CACHE_DIR = "cache_zscore"
EXPORT_DIR = "csv_medidas_paneles"

# Puedes dejar sólo ["hybrid"] o sólo ["complexity"] si no quieres procesar ambos.
METHODS = ["hybrid", "complexity"]

MAX_GAMMA = 6
MU = 2

TYPE_NULL = "shuffle"      # "shuffle" o "iaaft"
N_SURROGATES = 800
RANDOM_STATE = 12345
ALTERNATIVE = "less"       # "less", "greater", "two-sided"
NORMALIZE = True           # Se conserva como columna de metadatos

FORCE_RECOMPUTE = False

# Si quieres calcular sobre intervalos melódicos np.diff(x), usa True.
# Si quieres calcular sobre notas crudas, usa False.
USE_INTERVALS = False

# "raw" guarda C_d y gamma_d directamente.
# "one_minus" guarda 1 - C_d y 1 - gamma_d.
VALUE_MODE = "raw"

# Igual que dists.py: los CSV de panel exportan z.
# Opciones útiles: "z", "pe_obs", "obs", "mu_null", "sigma_null", "p_value", "p_raw".
CSV_VALUE_COL = "z"
WRITE_CSV = True


# =========================================================
# AUXILIARES DE ORDENAMIENTO Y NOMBRES
# =========================================================

def natural_key(text):
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", str(text))
    ]


def composer_sort_key(item):
    composer, meta = item
    idx = meta.get("Indice", np.inf)
    byear = meta.get("Birth_year", "999999")

    try:
        byear_num = int(byear)
    except Exception:
        byear_num = 999999

    return idx, byear_num, composer


def composer_labels(datos_composers):
    ordered = sorted(datos_composers.items(), key=composer_sort_key)

    names = []
    labels = []

    for composer, meta in ordered:
        byear = meta.get("Birth_year", "")
        names.append(composer)
        labels.append(f"{composer} {byear}".strip())

    return names, labels


def sanitize_filename(text):
    text = str(text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def measure_bases(use_intervals=False):
    if use_intervals:
        return "Cd_rank_interval", "gamma_rank_ties_interval"
    return "Cd_rank", "gamma_rank_ties"


def cache_filename(base_measure, d, mu, type_null, alternative, method):
    """
    Nombre tipo dists.py, pero extendido a todos los C_d y gamma_d.

    No incluye NORMALIZE en el nombre porque dists.py tampoco lo incluye
    en build_cache_key(...). NORMALIZE se conserva como columna del DataFrame.
    """
    return (
        f"{base_measure}_"
        f"d{d}_"
        f"mu{mu}_"
        f"{type_null}_"
        f"{alternative}_"
        f"{method}.pkl"
    )


def cache_path(base_measure, d, mu, type_null, alternative, method, cache_dir):
    return os.path.join(
        cache_dir,
        cache_filename(
            base_measure=base_measure,
            d=d,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            method=method,
        ),
    )


def get_target_cache_paths(
    max_gamma,
    mu,
    type_null,
    alternative,
    method,
    cache_dir,
    use_intervals=False,
):
    base_C, base_g = measure_bases(use_intervals=use_intervals)

    paths = OrderedDict()

    # C_1, C_2, ..., C_{max_gamma + 1}
    for d in range(1, max_gamma + 2):
        paths[("C", d)] = cache_path(
            base_measure=base_C,
            d=d,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            method=method,
            cache_dir=cache_dir,
        )

    # gamma_1, gamma_2, ..., gamma_{max_gamma}
    for j in range(1, max_gamma + 1):
        paths[("gamma", j)] = cache_path(
            base_measure=base_g,
            d=j,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            method=method,
            cache_dir=cache_dir,
        )

    return paths


# =========================================================
# ESTADÍSTICA NULA
# =========================================================

def empirical_p_raw(values_surrogates, value_obs, mu_null, alternative="two-sided"):
    values_surrogates = np.asarray(values_surrogates, dtype=float)

    if alternative == "greater":
        return np.mean(values_surrogates >= value_obs)

    if alternative == "less":
        return np.mean(values_surrogates <= value_obs)

    if alternative == "two-sided":
        return np.mean(
            np.abs(values_surrogates - mu_null)
            >= np.abs(value_obs - mu_null)
        )

    raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")


def scalar_null_stats(value_obs, values_surrogates, alternative):
    values_surrogates = np.asarray(values_surrogates, dtype=float)

    mu_null = np.mean(values_surrogates)
    sigma_null = np.std(values_surrogates, ddof=1)

    if sigma_null == 0:
        z = np.nan
    else:
        z = (value_obs - mu_null) / sigma_null

    if alternative == "greater":
        p_value = (
            np.sum(values_surrogates >= value_obs) + 1
        ) / (len(values_surrogates) + 1)

    elif alternative == "less":
        p_value = (
            np.sum(values_surrogates <= value_obs) + 1
        ) / (len(values_surrogates) + 1)

    elif alternative == "two-sided":
        p_value = (
            np.sum(
                np.abs(values_surrogates - mu_null)
                >= np.abs(value_obs - mu_null)
            ) + 1
        ) / (len(values_surrogates) + 1)

    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")

    p_raw = empirical_p_raw(
        values_surrogates,
        value_obs,
        mu_null,
        alternative=alternative,
    )

    return {
        "obs": value_obs,
        "mu_null": mu_null,
        "sigma_null": sigma_null,
        "z": z,
        "p_value": p_value,
        "p_raw": p_raw,
    }


def apply_value_mode(values, value_mode):
    values = np.asarray(values, dtype=float)

    if value_mode == "raw":
        return values

    if value_mode == "one_minus":
        return 1.0 - values

    raise ValueError("VALUE_MODE debe ser 'raw' o 'one_minus'.")


# =========================================================
# CÁLCULO DE C Y GAMMA PARA UNA SERIE
# =========================================================

def gamma_C_stats_for_series(
    x,
    max_gamma,
    mu,
    n_surrogates,
    type_null,
    random_state=None,
    use_intervals=False,
    value_mode="raw",
):
    rng = np.random.default_rng(random_state)

    x = np.asarray(x, dtype=float)

    if use_intervals:
        x_input = np.diff(x)
    else:
        x_input = x.copy()

    C_obs, g_obs = ml.gamma_index_rank_ties(
        x_input,
        max_gamma=max_gamma,
        mu=mu,
    )

    C_obs = np.asarray(C_obs, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    # Se guardan C_1,...,C_{max_gamma+1} y gamma_1,...,gamma_{max_gamma}.
    # C[0] suele ser C_0=1 y no se exporta como panel.
    required_C_len = max_gamma + 2       # índices 0, 1, ..., max_gamma+1
    required_g_len = max_gamma           # índices 0, ..., max_gamma-1

    if len(C_obs) < required_C_len:
        raise ValueError(
            f"C_obs tiene longitud {len(C_obs)}, pero se requiere al menos "
            f"{required_C_len} para guardar C_1,...,C_{max_gamma + 1}."
        )

    if len(g_obs) < required_g_len:
        raise ValueError(
            f"g_obs tiene longitud {len(g_obs)}, pero se requiere al menos "
            f"{required_g_len} para guardar gamma_1,...,gamma_{max_gamma}."
        )

    C_obs = C_obs[:required_C_len]
    g_obs = g_obs[:required_g_len]

    C_surrogates = np.empty((n_surrogates, required_C_len), dtype=float)
    g_surrogates = np.empty((n_surrogates, required_g_len), dtype=float)

    if type_null == "shuffle":
        for k in range(n_surrogates):
            x_surr = rng.permutation(x_input)

            C_surr, g_surr = ml.gamma_index_rank_ties(
                x_surr,
                max_gamma=max_gamma,
                mu=mu,
            )

            C_surr = np.asarray(C_surr, dtype=float)
            g_surr = np.asarray(g_surr, dtype=float)

            if len(C_surr) < required_C_len:
                raise ValueError(
                    f"C_surr tiene longitud {len(C_surr)}, pero se requiere "
                    f"{required_C_len}."
                )

            if len(g_surr) < required_g_len:
                raise ValueError(
                    f"g_surr tiene longitud {len(g_surr)}, pero se requiere "
                    f"{required_g_len}."
                )

            C_surrogates[k, :] = C_surr[:required_C_len]
            g_surrogates[k, :] = g_surr[:required_g_len]

    elif type_null == "iaaft":
        x_surr_all = ml.iaaft(x_input, n_surrogates)
        x_surr_all = np.asarray(x_surr_all, dtype=float)

        for k in range(n_surrogates):
            C_surr, g_surr = ml.gamma_index_rank_ties(
                x_surr_all[k, :],
                max_gamma=max_gamma,
                mu=mu,
            )

            C_surr = np.asarray(C_surr, dtype=float)
            g_surr = np.asarray(g_surr, dtype=float)

            if len(C_surr) < required_C_len:
                raise ValueError(
                    f"C_surr tiene longitud {len(C_surr)}, pero se requiere "
                    f"{required_C_len}."
                )

            if len(g_surr) < required_g_len:
                raise ValueError(
                    f"g_surr tiene longitud {len(g_surr)}, pero se requiere "
                    f"{required_g_len}."
                )

            C_surrogates[k, :] = C_surr[:required_C_len]
            g_surrogates[k, :] = g_surr[:required_g_len]

    else:
        raise ValueError("type_null debe ser 'shuffle' o 'iaaft'.")

    C_obs = apply_value_mode(C_obs, value_mode)
    g_obs = apply_value_mode(g_obs, value_mode)

    C_surrogates = apply_value_mode(C_surrogates, value_mode)
    g_surrogates = apply_value_mode(g_surrogates, value_mode)

    return C_obs, g_obs, C_surrogates, g_surrogates


# =========================================================
# CONSTRUCCIÓN DE TODOS LOS DATAFRAMES
# =========================================================

def compute_all_C_gamma_dataframes(
    composers,
    datos_composers,
    max_gamma,
    mu,
    n_surrogates,
    type_null,
    normalize,
    alternative,
    random_state,
    method,
    use_intervals=False,
    value_mode="raw",
):
    composer_names, _ = composer_labels(datos_composers)
    base_C, base_g = measure_bases(use_intervals=use_intervals)

    rows_by_measure = OrderedDict()

    for d in range(1, max_gamma + 2):
        rows_by_measure[("C", d)] = []

    for j in range(1, max_gamma + 1):
        rows_by_measure[("gamma", j)] = []

    global_counter = 0
    seed_base = int(random_state) if random_state is not None else None

    for composer in composer_names:
        if composer not in composers:
            continue

        print(f"[METHOD={method}] [COMPOSER] {composer}")

        meta = datos_composers.get(composer, {})
        birth_year = meta.get("Birth_year", np.nan)
        composer_index = meta.get("Indice", np.nan)

        series_names = sorted(
            composers[composer].keys(),
            key=natural_key,
        )

        for sname in series_names:
            x = np.asarray(composers[composer][sname], dtype=float)

            if seed_base is None:
                seed_here = None
            else:
                seed_here = (
                    seed_base
                    + 100000 * int(max_gamma)
                    + 1000 * int(mu)
                    + global_counter
                )

            global_counter += 1

            print(f"  {sname} | seed={seed_here}")

            C_obs, g_obs, C_surr, g_surr = gamma_C_stats_for_series(
                x=x,
                max_gamma=max_gamma,
                mu=mu,
                n_surrogates=n_surrogates,
                type_null=type_null,
                random_state=seed_here,
                use_intervals=use_intervals,
                value_mode=value_mode,
            )

            length_used = len(np.diff(x)) if use_intervals else len(x)

            common_info = {
                "method": method,
                "composer": composer,
                "birth_year": birth_year,
                "composer_index": composer_index,
                "serie": sname,
                "length": len(x),
                "length_used": length_used,
                "max_gamma_run": max_gamma,
                "mu": mu,
                "tau": mu,  # compatibilidad con dists.py
                "n_surrogates": n_surrogates,
                "normalize": normalize,
                "alternative": alternative,
                "random_state": seed_here,
                "type_null": type_null,
                "use_intervals": use_intervals,
                "value_mode": value_mode,
            }

            # C_1, ..., C_{max_gamma+1}
            for d in range(1, max_gamma + 2):
                stats = scalar_null_stats(
                    value_obs=C_obs[d],
                    values_surrogates=C_surr[:, d],
                    alternative=alternative,
                )

                row = dict(common_info)
                row.update({
                    "measure": base_C,
                    "component": d,
                    "D": d,
                    "source_vector": "C",
                    "pe_obs": stats["obs"],   # compatibilidad con dists.py
                    "obs": stats["obs"],
                    "mu_null": stats["mu_null"],
                    "sigma_null": stats["sigma_null"],
                    "z": stats["z"],
                    "p_value": stats["p_value"],
                    "p_raw": stats["p_raw"],
                })

                rows_by_measure[("C", d)].append(row)

            # gamma_1, ..., gamma_{max_gamma}
            for j in range(1, max_gamma + 1):
                stats = scalar_null_stats(
                    value_obs=g_obs[j - 1],
                    values_surrogates=g_surr[:, j - 1],
                    alternative=alternative,
                )

                row = dict(common_info)
                row.update({
                    "measure": base_g,
                    "component": j,
                    "D": j,
                    "source_vector": "gamma",
                    "pe_obs": stats["obs"],   # compatibilidad con dists.py
                    "obs": stats["obs"],
                    "mu_null": stats["mu_null"],
                    "sigma_null": stats["sigma_null"],
                    "z": stats["z"],
                    "p_value": stats["p_value"],
                    "p_raw": stats["p_raw"],
                })

                rows_by_measure[("gamma", j)].append(row)

    dfs_by_measure = OrderedDict(
        (key, pd.DataFrame(rows))
        for key, rows in rows_by_measure.items()
    )

    return dfs_by_measure


# =========================================================
# GUARDADO DE PKL
# =========================================================

def save_all_C_gamma_pickles(
    dfs_by_measure,
    max_gamma,
    mu,
    type_null,
    alternative,
    method,
    cache_dir,
    use_intervals=False,
):
    os.makedirs(cache_dir, exist_ok=True)

    base_C, base_g = measure_bases(use_intervals=use_intervals)
    saved_paths = []

    for d in range(1, max_gamma + 2):
        df = dfs_by_measure[("C", d)]

        path = cache_path(
            base_measure=base_C,
            d=d,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            method=method,
            cache_dir=cache_dir,
        )

        df.to_pickle(path)
        saved_paths.append(path)
        print(f"[CACHE] Guardado: {path}")

    for j in range(1, max_gamma + 1):
        df = dfs_by_measure[("gamma", j)]

        path = cache_path(
            base_measure=base_g,
            d=j,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            method=method,
            cache_dir=cache_dir,
        )

        df.to_pickle(path)
        saved_paths.append(path)
        print(f"[CACHE] Guardado: {path}")

    return saved_paths


def load_all_C_gamma_pickles(
    max_gamma,
    mu,
    type_null,
    alternative,
    method,
    cache_dir,
    use_intervals=False,
):
    target_paths = get_target_cache_paths(
        max_gamma=max_gamma,
        mu=mu,
        type_null=type_null,
        alternative=alternative,
        method=method,
        cache_dir=cache_dir,
        use_intervals=use_intervals,
    )

    dfs_by_measure = OrderedDict()
    for key, path in target_paths.items():
        dfs_by_measure[key] = pd.read_pickle(path)
        print(f"[CACHE] Cargado: {path}")

    return dfs_by_measure, list(target_paths.values())


# =========================================================
# GUARDADO DE CSV TIPO dists.py
# =========================================================

def build_measure_dict_from_panel(df_panel, composer_names, value_col="z"):
    if value_col not in df_panel.columns:
        raise ValueError(
            f"CSV_VALUE_COL='{value_col}' no está en el DataFrame. "
            f"Columnas disponibles: {list(df_panel.columns)}"
        )

    measure_dict = {}

    for composer in composer_names:
        sub = df_panel[df_panel["composer"] == composer].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("serie", key=lambda s: s.map(natural_key))

        measure_dict[composer] = {}
        for _, row in sub.iterrows():
            measure_dict[composer][row["serie"]] = row[value_col]

    return measure_dict


def build_measure_dataframe_from_panel(df_panel, composer_names, value_col="z"):
    measure_dict = build_measure_dict_from_panel(
        df_panel=df_panel,
        composer_names=composer_names,
        value_col=value_col,
    )

    df_measure = pd.DataFrame(measure_dict)

    if not df_measure.empty:
        df_measure = df_measure.sort_index(
            key=lambda idx: [natural_key(x) for x in idx]
        )

    valid_cols = [c for c in composer_names if c in df_measure.columns]
    df_measure = df_measure.reindex(columns=valid_cols)

    return df_measure


def panel_csv_filename(measure, D, tau, type_null, method):
    fname = (
        f"{measure}_"
        f"D{D}_"
        f"tau{tau}_"
        f"{type_null}_"
        f"{method}.csv"
    )
    return sanitize_filename(fname)


def save_panel_measure_csv(
    df_panel,
    composer_names,
    measure,
    D,
    tau,
    type_null,
    method,
    export_dir=EXPORT_DIR,
    value_col="z",
):
    os.makedirs(export_dir, exist_ok=True)

    df_measure = build_measure_dataframe_from_panel(
        df_panel=df_panel,
        composer_names=composer_names,
        value_col=value_col,
    )

    fname = panel_csv_filename(
        measure=measure,
        D=D,
        tau=tau,
        type_null=type_null,
        method=method,
    )
    path = os.path.join(export_dir, fname)

    df_measure.to_csv(path, encoding="utf-8")
    print(f"[CSV] Guardado: {path}")

    return path


def save_all_C_gamma_csvs(
    dfs_by_measure,
    datos_composers,
    max_gamma,
    mu,
    type_null,
    method,
    export_dir,
    use_intervals=False,
    value_col="z",
):
    composer_names, _ = composer_labels(datos_composers)
    base_C, base_g = measure_bases(use_intervals=use_intervals)

    saved_paths = []

    for d in range(1, max_gamma + 2):
        path = save_panel_measure_csv(
            df_panel=dfs_by_measure[("C", d)],
            composer_names=composer_names,
            measure=base_C,
            D=d,
            tau=mu,
            type_null=type_null,
            method=method,
            export_dir=export_dir,
            value_col=value_col,
        )
        saved_paths.append(path)

    for j in range(1, max_gamma + 1):
        path = save_panel_measure_csv(
            df_panel=dfs_by_measure[("gamma", j)],
            composer_names=composer_names,
            measure=base_g,
            D=j,
            tau=mu,
            type_null=type_null,
            method=method,
            export_dir=export_dir,
            value_col=value_col,
        )
        saved_paths.append(path)

    return saved_paths


# =========================================================
# CACHÉ GLOBAL
# =========================================================

def get_or_compute_all_C_gamma_pickles(
    composers,
    datos_composers,
    max_gamma,
    mu,
    n_surrogates,
    type_null,
    normalize,
    alternative,
    random_state,
    method,
    cache_dir,
    force_recompute=False,
    use_intervals=False,
    value_mode="raw",
):
    target_paths = get_target_cache_paths(
        max_gamma=max_gamma,
        mu=mu,
        type_null=type_null,
        alternative=alternative,
        method=method,
        cache_dir=cache_dir,
        use_intervals=use_intervals,
    )

    all_exist = all(os.path.exists(path) for path in target_paths.values())

    if all_exist and not force_recompute:
        print(f"[CACHE] METHOD={method}: todos los PKL ya existen. No se recalcula.")
        return load_all_C_gamma_pickles(
            max_gamma=max_gamma,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            method=method,
            cache_dir=cache_dir,
            use_intervals=use_intervals,
        )

    print(f"[CACHE] METHOD={method}: calculando todos los C_d y gamma_d en una sola corrida...")

    dfs_by_measure = compute_all_C_gamma_dataframes(
        composers=composers,
        datos_composers=datos_composers,
        max_gamma=max_gamma,
        mu=mu,
        n_surrogates=n_surrogates,
        type_null=type_null,
        normalize=normalize,
        alternative=alternative,
        random_state=random_state,
        method=method,
        use_intervals=use_intervals,
        value_mode=value_mode,
    )

    saved_paths = save_all_C_gamma_pickles(
        dfs_by_measure=dfs_by_measure,
        max_gamma=max_gamma,
        mu=mu,
        type_null=type_null,
        alternative=alternative,
        method=method,
        cache_dir=cache_dir,
        use_intervals=use_intervals,
    )

    return dfs_by_measure, saved_paths


# =========================================================
# CARGA DEL DATASET TIPO dists.py
# =========================================================

def load_dataset_for_method(method, data_dir=DATA_DIR):
    composers_path = os.path.join(data_dir, f"composers_{method}.pkl")
    datos_path = os.path.join(data_dir, f"datos_composers_{method}.pkl")

    if not os.path.exists(composers_path):
        raise FileNotFoundError(
            f"No existe {composers_path}. Revisa DATA_DIR o METHODS."
        )

    if not os.path.exists(datos_path):
        raise FileNotFoundError(
            f"No existe {datos_path}. Revisa DATA_DIR o METHODS."
        )

    with open(composers_path, "rb") as f:
        composers = pickle.load(f)

    with open(datos_path, "rb") as f:
        datos_composers = pickle.load(f)

    return composers, datos_composers


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    all_pkl_paths = []
    all_csv_paths = []

    for method in METHODS:
        print("\n" + "=" * 72)
        print(f"METHOD = {method}")
        print("=" * 72)

        composers, datos_composers = load_dataset_for_method(
            method=method,
            data_dir=DATA_DIR,
        )

        dfs_by_measure, pkl_paths = get_or_compute_all_C_gamma_pickles(
            composers=composers,
            datos_composers=datos_composers,
            max_gamma=MAX_GAMMA,
            mu=MU,
            n_surrogates=N_SURROGATES,
            type_null=TYPE_NULL,
            normalize=NORMALIZE,
            alternative=ALTERNATIVE,
            random_state=RANDOM_STATE,
            method=method,
            cache_dir=CACHE_DIR,
            force_recompute=FORCE_RECOMPUTE,
            use_intervals=USE_INTERVALS,
            value_mode=VALUE_MODE,
        )

        all_pkl_paths.extend(pkl_paths)

        if WRITE_CSV:
            csv_paths = save_all_C_gamma_csvs(
                dfs_by_measure=dfs_by_measure,
                datos_composers=datos_composers,
                max_gamma=MAX_GAMMA,
                mu=MU,
                type_null=TYPE_NULL,
                method=method,
                export_dir=EXPORT_DIR,
                use_intervals=USE_INTERVALS,
                value_col=CSV_VALUE_COL,
            )
            all_csv_paths.extend(csv_paths)

    print("\nArchivos PKL generados/cargados:")
    for path in all_pkl_paths:
        print(path)

    if WRITE_CSV:
        print("\nArchivos CSV generados:")
        for path in all_csv_paths:
            print(path)
