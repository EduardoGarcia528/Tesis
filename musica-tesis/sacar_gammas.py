# -*- coding: utf-8 -*-
import os
import re
import pickle
import numpy as np
import pandas as pd
import mi_libreria as ml

# =========================================================
# PARÁMETROS GENERALES
# =========================================================

CACHE_DIR = "cache_zscore"

CARPETA_DATASET = r"data/dataset_procesado"

MAX_GAMMA = 6
MU = 2

TYPE_NULL = "shuffle"      # "shuffle" o "iaaft"
N_SURROGATES = 800
RANDOM_STATE = 12345
ALTERNATIVE = "less"       # "less", "greater", "two-sided"
NORMALIZE = True           # Se conserva para compatibilidad en el nombre

FORCE_RECOMPUTE = False

# Si quieres calcular sobre intervalos melódicos np.diff(x), usa True.
# Si quieres calcular sobre notas crudas, usa False.
USE_INTERVALS = False

# "raw" guarda C_d y gamma_d directamente.
# "one_minus" guarda 1 - C_d y 1 - gamma_d.
# Para lo que pediste, lo natural es "raw".
VALUE_MODE = "raw"


# =========================================================
# AUXILIARES
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


def empirical_p_raw(values_surrogates, value_obs, mu_null, alternative="two-sided"):
    values_surrogates = np.asarray(values_surrogates, dtype=float)

    if alternative == "greater":
        return np.mean(values_surrogates >= value_obs)

    elif alternative == "less":
        return np.mean(values_surrogates <= value_obs)

    elif alternative == "two-sided":
        return np.mean(
            np.abs(values_surrogates - mu_null)
            >= np.abs(value_obs - mu_null)
        )

    else:
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
        alternative=alternative
    )

    return {
        "obs": value_obs,
        "mu_null": mu_null,
        "sigma_null": sigma_null,
        "z": z,
        "p_value": p_value,
        "p_raw": p_raw
    }


def apply_value_mode(values, value_mode):
    values = np.asarray(values, dtype=float)

    if value_mode == "raw":
        return values

    elif value_mode == "one_minus":
        return 1.0 - values

    else:
        raise ValueError("VALUE_MODE debe ser 'raw' o 'one_minus'.")


def cache_filename(base_measure, d, mu, type_null, alternative, normalize):
    return (
        f"{base_measure}_"
        f"d{d}_"
        f"mu{mu}_"
        f"{type_null}_"
        f"{alternative}_"
        f"norm{int(normalize)}.pkl"
    )


def cache_path(base_measure, d, mu, type_null, alternative, normalize, cache_dir):
    fname = cache_filename(
        base_measure=base_measure,
        d=d,
        mu=mu,
        type_null=type_null,
        alternative=alternative,
        normalize=normalize
    )
    return os.path.join(cache_dir, fname)


def get_target_cache_paths(
    max_gamma,
    mu,
    type_null,
    alternative,
    normalize,
    cache_dir,
    use_intervals=False
):
    if use_intervals:
        base_C = "Cd_rank_interval"
        base_g = "gamma_rank_ties_interval"
    else:
        base_C = "Cd_rank"
        base_g = "gamma_rank_ties"

    paths = {}

    # C_1, C_2, ..., C_{max_gamma + 1}
    for d in range(1, max_gamma + 2):
        key = ("C", d)
        paths[key] = cache_path(
            base_measure=base_C,
            d=d,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            normalize=normalize,
            cache_dir=cache_dir
        )

    # gamma_1, gamma_2, ..., gamma_{max_gamma}
    for j in range(1, max_gamma + 1):
        key = ("gamma", j)
        paths[key] = cache_path(
            base_measure=base_g,
            d=j,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            normalize=normalize,
            cache_dir=cache_dir
        )

    return paths


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
    value_mode="raw"
):
    rng = np.random.default_rng(random_state)

    x = np.asarray(x, dtype=float)

    if use_intervals:
        x_input = np.diff(x)
    else:
        x_input = x.copy()

    # Cálculo observado.
    C_obs, g_obs = ml.gamma_index_rank_ties(
        x_input,
        max_gamma=max_gamma,
        mu=mu
    )

    C_obs = np.asarray(C_obs, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    # Queremos guardar:
    # C_1, ..., C_{max_gamma + 1}
    # gamma_1, ..., gamma_{max_gamma}
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
                mu=mu
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
                mu=mu
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
    use_intervals=False,
    value_mode="raw"
):
    composer_names, _ = composer_labels(datos_composers)

    rows_by_measure = {}

    # Inicializar contenedores:
    # C_1, ..., C_{max_gamma+1}
    for d in range(1, max_gamma + 2):
        rows_by_measure[("C", d)] = []

    # gamma_1, ..., gamma_{max_gamma}
    for j in range(1, max_gamma + 1):
        rows_by_measure[("gamma", j)] = []

    global_counter = 0
    seed_base = int(random_state) if random_state is not None else None

    for composer in composer_names:

        if composer not in composers:
            continue

        print(f"[COMPOSER] {composer}")

        meta = datos_composers.get(composer, {})
        birth_year = meta.get("Birth_year", np.nan)
        composer_index = meta.get("Indice", np.nan)

        series_names = sorted(
            composers[composer].keys(),
            key=natural_key
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
                value_mode=value_mode
            )

            length_used = len(np.diff(x)) if use_intervals else len(x)

            common_info = {
                "composer": composer,
                "birth_year": birth_year,
                "composer_index": composer_index,
                "serie": sname,
                "length": len(x),
                "length_used": length_used,
                "max_gamma_run": max_gamma,
                "mu": mu,
                "tau": mu,  # compatibilidad con código previo
                "n_surrogates": n_surrogates,
                "normalize": normalize,
                "alternative": alternative,
                "random_state": seed_here,
                "type_null": type_null,
                "use_intervals": use_intervals,
                "value_mode": value_mode
            }

            # Guardar C_1, ..., C_{max_gamma+1}
            for d in range(1, max_gamma + 2):

                stats = scalar_null_stats(
                    value_obs=C_obs[d],
                    values_surrogates=C_surr[:, d],
                    alternative=alternative
                )

                row = dict(common_info)
                row.update({
                    "measure": "Cd_rank_interval" if use_intervals else "Cd_rank",
                    "component": d,
                    "D": d,
                    "source_vector": "C",
                    "pe_obs": stats["obs"],   # compatibilidad con código viejo
                    "obs": stats["obs"],
                    "mu_null": stats["mu_null"],
                    "sigma_null": stats["sigma_null"],
                    "z": stats["z"],
                    "p_value": stats["p_value"],
                    "p_raw": stats["p_raw"]
                })

                rows_by_measure[("C", d)].append(row)

            # Guardar gamma_1, ..., gamma_{max_gamma}
            for j in range(1, max_gamma + 1):

                stats = scalar_null_stats(
                    value_obs=g_obs[j - 1],
                    values_surrogates=g_surr[:, j - 1],
                    alternative=alternative
                )

                row = dict(common_info)
                row.update({
                    "measure": (
                        "gamma_rank_ties_interval"
                        if use_intervals
                        else "gamma_rank_ties"
                    ),
                    "component": j,
                    "D": j,
                    "source_vector": "gamma",
                    "pe_obs": stats["obs"],   # compatibilidad con código viejo
                    "obs": stats["obs"],
                    "mu_null": stats["mu_null"],
                    "sigma_null": stats["sigma_null"],
                    "z": stats["z"],
                    "p_value": stats["p_value"],
                    "p_raw": stats["p_raw"]
                })

                rows_by_measure[("gamma", j)].append(row)

    dfs_by_measure = {
        key: pd.DataFrame(rows)
        for key, rows in rows_by_measure.items()
    }

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
    normalize,
    cache_dir,
    use_intervals=False
):
    os.makedirs(cache_dir, exist_ok=True)

    if use_intervals:
        base_C = "Cd_rank_interval"
        base_g = "gamma_rank_ties_interval"
    else:
        base_C = "Cd_rank"
        base_g = "gamma_rank_ties"

    saved_paths = []

    # C_1, ..., C_{max_gamma+1}
    for d in range(1, max_gamma + 2):
        df = dfs_by_measure[("C", d)]

        path = cache_path(
            base_measure=base_C,
            d=d,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            normalize=normalize,
            cache_dir=cache_dir
        )

        df.to_pickle(path)
        saved_paths.append(path)
        print(f"[CACHE] Guardado: {path}")

    # gamma_1, ..., gamma_{max_gamma}
    for j in range(1, max_gamma + 1):
        df = dfs_by_measure[("gamma", j)]

        path = cache_path(
            base_measure=base_g,
            d=j,
            mu=mu,
            type_null=type_null,
            alternative=alternative,
            normalize=normalize,
            cache_dir=cache_dir
        )

        df.to_pickle(path)
        saved_paths.append(path)
        print(f"[CACHE] Guardado: {path}")

    return saved_paths


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
    cache_dir,
    force_recompute=False,
    use_intervals=False,
    value_mode="raw"
):
    target_paths = get_target_cache_paths(
        max_gamma=max_gamma,
        mu=mu,
        type_null=type_null,
        alternative=alternative,
        normalize=normalize,
        cache_dir=cache_dir,
        use_intervals=use_intervals
    )

    all_exist = all(os.path.exists(path) for path in target_paths.values())

    if all_exist and not force_recompute:
        print("[CACHE] Todos los archivos ya existen. No se recalcula.")
        for path in target_paths.values():
            print(f"  {path}")
        return list(target_paths.values())

    print("[CACHE] Calculando todos los C_d y gamma_d en una sola corrida...")

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
        use_intervals=use_intervals,
        value_mode=value_mode
    )

    saved_paths = save_all_C_gamma_pickles(
        dfs_by_measure=dfs_by_measure,
        max_gamma=max_gamma,
        mu=mu,
        type_null=type_null,
        alternative=alternative,
        normalize=normalize,
        cache_dir=cache_dir,
        use_intervals=use_intervals
    )

    return saved_paths


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    with open(os.path.join(CARPETA_DATASET, "composers.pkl"), "rb") as f:
        composers = pickle.load(f)

    with open(os.path.join(CARPETA_DATASET, "datos_composers.pkl"), "rb") as f:
        datos_composers = pickle.load(f)

    saved_paths = get_or_compute_all_C_gamma_pickles(
        composers=composers,
        datos_composers=datos_composers,
        max_gamma=MAX_GAMMA,
        mu=MU,
        n_surrogates=N_SURROGATES,
        type_null=TYPE_NULL,
        normalize=NORMALIZE,
        alternative=ALTERNATIVE,
        random_state=RANDOM_STATE,
        cache_dir=CACHE_DIR,
        force_recompute=FORCE_RECOMPUTE,
        use_intervals=USE_INTERVALS,
        value_mode=VALUE_MODE
    )

    print("\nArchivos generados/cargados:")
    for path in saved_paths:
        print(path)