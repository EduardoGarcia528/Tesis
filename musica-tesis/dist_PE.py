# -*- coding: utf-8 -*-
import os
import pickle
import re
import hashlib
import numpy as np
import pandas as pd
import copy 
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import mi_libreria as ml

# =========================================================
# PARÁMETROS GENERALES
# =========================================================
DOT_SIZE = 22
EDGE_ALPHA = 0.55
JITTER_STD = 0.06
LINE_WIDTH = 0.9

FONT_GENERAL = 12
FONT_TICKS = 11
LEGEND_FONTSIZE = 11
TITLE_SIZE = 14
PERCENT_FONTSIZE = 11

BOTTOM_MARGIN = 0.28
LEFT_MARGIN = 0.08
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.90

# =========================================================
# PARÁMETROS DEL TEST NULO
# =========================================================
MEASURE = "mPE"
TYPE_NULL = "shuffle"
N_SURROGATES = 800
RANDOM_STATE = 12345
ALTERNATIVE = "less"   # 'two-sided', 'greater', 'less'
NORMALIZE = True

# rojo si ningún surrogate fue tan extremo como el dato
USE_P_RAW_ZERO_FOR_RED = True
ALPHA = 0.05   # sólo se usa si USE_P_RAW_ZERO_FOR_RED = False

# =========================================================
# CONFIGURACIÓN DE LOS 4 PANELES
# =========================================================

PANEL_CONFIGS = [
    {"measure":"H_tau5_v2","D": 2, "tau": 5,"type_null":"shuffle", "title": r"$Indice H_{\tau=5}$ (notes,shuffle)"},
    # {"measure":"gamma_rank_ties","D": 1, "tau": 2,"type_null":"shuffle", "title": r"$\gamma_1^{(R)}(\mu=2)$ (notes,shuffle)"},
    # {"measure":"gamma_rank_ties","D": 2, "tau": 2,"type_null":"shuffle", "title": r"$\gamma_2^{(R)}(\mu=2)$ (notes,shuffle)"},
    # {"measure":"gamma_rank_ties","D": 3, "tau": 2,"type_null":"shuffle", "title": r"$\gamma_3^{(R)}(\mu=2)$ (notes,shuffle)"},
    # {"measure":"gamma_rank_ties","D": 4, "tau": 2,"type_null":"shuffle", "title": r"$\gamma_4^{(R)}(\mu=2)$ (notes,shuffle)"},
    # {"measure":"Cd_rank","D": 4, "tau": 2,"type_null":"shuffle", "title": fr"mPE (notes,shuffle): $m=5,\ \tau=1$"},
    {"measure":"mPE","D":3, "tau": 1,"type_null":"shuffle", "title": fr"mPE (notes,shuffle): $m=3,\ \tau=1$"},
    {"measure":"mPE","D": 4, "tau": 1,"type_null":"shuffle", "title": fr"mPE (notes,shuffle): $m=4,\ \tau=1$"},
    {"measure":"mPE","D": 5, "tau": 1,"type_null":"shuffle", "title": fr"mPE (notes,shuffle): $m=5,\ \tau=1$"},
] 

# =========================================================
# CACHÉ
# =========================================================
CACHE_DIR = f"cache_zscore"
FORCE_RECOMPUTE = False   # True si quieres forzar recálculo

# =========================================================
# EXPORTAR MEDIDAS POR PANEL A CSV
# =========================================================
EXPORT_DIR = "csv_medidas_paneles"

def sanitize_filename(text):
    text = str(text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text

def build_measure_dict_from_panel(df_panel, composer_names):
    """
    Reconstruye una estructura tipo:
    measure_dict[composer][serie] = pe_obs
    respetando el orden natural de las series.
    """
    measure_dict = {}

    for composer in composer_names:
        sub = df_panel[df_panel["composer"] == composer].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("serie", key=lambda s: s.map(natural_key))

        measure_dict[composer] = {}
        for _, row in sub.iterrows():
            measure_dict[composer][row["serie"]] = row["z"]

    return measure_dict

def build_measure_dataframe_from_panel(df_panel, composer_names):
    """
    Convierte el panel a un DataFrame:
      columnas = compositores
      índice   = Serie_1, Serie_2, ...
      valores  = pe_obs
    """
    measure_dict = build_measure_dict_from_panel(df_panel, composer_names)
    df_measure = pd.DataFrame(measure_dict)

    # ordenar índice tipo Serie_1, Serie_2, ...
    df_measure = df_measure.sort_index(key=lambda idx: [natural_key(x) for x in idx])

    # ordenar columnas según composer_names
    valid_cols = [c for c in composer_names if c in df_measure.columns]
    df_measure = df_measure.reindex(columns=valid_cols)

    return df_measure

def save_panel_measure_csv(df_panel, composer_names, cfg, export_dir=EXPORT_DIR):
    os.makedirs(export_dir, exist_ok=True)

    df_measure = build_measure_dataframe_from_panel(df_panel, composer_names)

    fname = (
        f"{cfg['measure']}_"
        f"D{cfg['D']}_"
        f"tau{cfg['tau']}_"
        f"{cfg['type_null']}.csv"
    )
    fname = sanitize_filename(fname)
    path = os.path.join(export_dir, fname)

    df_measure.to_csv(path, encoding="utf-8")
    print(f"[CSV] Guardado: {path}")

    return df_measure, path
# =========================================================
# AUXILIARES
# =========================================================

def medianas_por_compositor(df_panel, composer_names):
    medianas = []
    for composer in composer_names:
        sub = df_panel[df_panel["composer"] == composer].copy()
        if sub.empty:
            medianas.append(np.nan)
            continue

        zvals = sub["z"].to_numpy(dtype=float)
        zvals = zvals[np.isfinite(zvals)]

        if zvals.size == 0:
            medianas.append(np.nan)
        else:
            medianas.append(np.median(zvals))

    return np.array(medianas, dtype=float)


#dataframe datos de compositores 
def extraer_dataset_musica():

    datos_composers = {}
    carpeta = r'data\Sequences\labels'
    archivos_en_carpeta = os.listdir(carpeta)
    index0 = 0
    indice = 0

    for archivo in archivos_en_carpeta:
        ruta_completa = os.path.join(carpeta, archivo)
        serie = pd.read_csv(ruta_completa, header = None)
        composer = archivo.split('-')[1].capitalize() # nombre compositor
        datos_composers[composer] = {} #genero bibio para composer
        datos_composers[composer]['Birth_year'] = archivo.split('-')[0] #año de nacimiento
        index1 = serie.iloc[0, 0].split('\t')[0] #el # del primer serie del composer
        index2 = int(serie.iloc[len(serie)-3, 0].split('\t')[0]) - index0 # # Piezas
        index0 = index2 + index0 # numero total de piezas anteriores
        datos_composers[composer]['# Piezas'] = index2 # Piezas
        datos_composers[composer]['Indice'] = indice
        indice += 1

    composers = {}
    M = 0
    carpeta = r'data\Sequences\Series'
    archivos_en_carpeta = os.listdir(carpeta)

    for archivo in archivos_en_carpeta:
        ruta_completa = os.path.join(carpeta, archivo)
        serie = pd.read_csv(ruta_completa)
        # escoge una serie
        composer = archivo.split('-')[1].capitalize() # nombre compositor
        composers[composer] = {}

        for pieza in range( datos_composers[composer]['# Piezas'] ):
            N = serie.iloc[0, 0].split('\t')[1] # # de elementos por pieza
            M = int(N) + M
            index_n1 = 0 
            index_n2 = int(N)+2 
            serie_n = serie[index_n1 + 2:index_n2].reset_index(drop=True) # resetear index
            serie = serie[index_n2 +1:] # recortar serie Original
            serie_n.index += 1 # que index empiece desde 1
            num_serie_T = serie.columns[0]  # numero de serie de todo el dataset
            num_serie = pieza + 1
            composers[composer]['Serie_'+str(num_serie)] = serie_n.squeeze().to_numpy().astype(float) # agregamos pieza al dicc composer con key como # serie

    ###
    ###

    composers_depurado = copy.deepcopy(composers)
    datos_composers_depurado = copy.deepcopy(datos_composers)

    for i,composer in enumerate(composers.keys()):
        d = 0
        for pieza in composers[composer].keys():
            if len(composers[composer][pieza])//2 < 400:
                del composers_depurado[composer][pieza]
                d = d + 1
        datos_composers_depurado[composer]['# Piezas'] = datos_composers[composer]['# Piezas'] - d


    # 40 promedio de numero de piezas por compositor
    composers_depurado_v2 = copy.deepcopy(composers_depurado)
    composers_depurado_v2_keychange = copy.deepcopy(composers_depurado_v2)
    datos_composers_depurado_v2 = copy.deepcopy(datos_composers_depurado)

    for composer in composers.keys():
        if datos_composers_depurado[composer]['# Piezas'] < 30:
            del composers_depurado_v2[composer]
            del datos_composers_depurado_v2[composer]
        
    for i,composer in enumerate(composers_depurado_v2.keys()):
        datos_composers_depurado_v2[composer]['Indice'] = i 

    for composer in composers_depurado_v2.keys():
        for i,serie in enumerate(composers_depurado_v2[composer].keys()):
            composers_depurado_v2_keychange[composer]['Serie_' + str(i+1)] = composers_depurado_v2_keychange[composer].pop(serie)

    print(" # de compositores restantes: ", len(composers_depurado_v2))

    return composers_depurado_v2, datos_composers_depurado_v2

def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', str(text))]

def composer_sort_key(item):
    composer, meta = item
    idx = meta.get("Indice", np.inf)
    byear = meta.get("Birth_year", "999999")
    try:
        byear_num = int(byear)
    except Exception:
        byear_num = 999999
    return (idx, byear_num, composer)

def composer_labels(datos_composers):
    ordered = sorted(datos_composers.items(), key=composer_sort_key)
    names = []
    labels = []
    for composer, meta in ordered:
        byear = meta.get("Birth_year", "")
        names.append(composer)
        labels.append(f"{composer} {byear}".strip())
    return names, labels

def empirical_p_raw(pe_surrogates, pe_obs, mu_null, alternative="two-sided"):
    pe_surrogates = np.asarray(pe_surrogates, dtype=float)

    if alternative == "greater":
        return np.mean(pe_surrogates >= pe_obs)
    elif alternative == "less":
        return np.mean(pe_surrogates <= pe_obs)
    elif alternative == "two-sided":
        return np.mean(np.abs(pe_surrogates - mu_null) >= np.abs(pe_obs - mu_null))
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")

def pe_stats_for_series(
    x,
    measure,
    D,
    tau,
    n_surrogates,
    type_null,
    alternative,
    random_state=None
):
    """
    Requiere permutation_entropy(...) ya definida en tu entorno.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    if "noNorm" in measure:
        fafaeg = False
    else:
        fafaeg = True
    if "PE" in measure:
        if "interval" in measure:
            pe_obs = ml.modified_permutation_entropy(np.diff(x), m=D, tau=tau,norm=fafaeg)
        else:
            pe_obs = ml.modified_permutation_entropy(x, m=D, tau=tau, norm=fafaeg)
    elif "H_tau" in measure:
        if "interval" in measure:
            pe_obs = ml.indice_S_eff_fast(np.diff(x), None, tau=tau)
        else:
            pe_obs = ml.indice_S_eff_fast(x, None, tau=tau,delta=True)
    elif "Cd" in measure:
        if "interval" in measure:
            # angulos = angulos_alpha(np.diff(x),False)
            C,_ = ml.gamma_index_rank_ties(np.diff(x),max_gamma=D,mu=tau)
            pe_obs = 1-C[-1]
        else:
            # angulos = angulos_alpha(x,False)
            C,_ = ml.gamma_index_rank_ties(x,max_gamma=D,mu=tau)
            pe_obs = 1-C[-1]
    elif "gamma" in measure:
        if "interval" in measure:
            # angulos = angulos_alpha(np.diff(x),False)
            _,C = ml.gamma_index_rank_ties(np.diff(x),max_gamma=D,mu=tau)
            pe_obs = 1-C[-1]
        else:
            # angulos = angulos_alpha(x,False)
            _,C = ml.gamma_index_rank_ties(x,max_gamma=D,mu=tau)
            pe_obs = 1-C[-1]
    elif "J_tau" in measure:
        if "interval" in measure:
            x_input = np.diff(x)
        else:
            x_input = x
        pe_obs = ml.indice_J(x_input, None, tau=tau)
    pe_surrogates = np.empty(n_surrogates, dtype=float)
    if type_null == "shuffle":
        for k in range(n_surrogates):
            if "interval" in measure:
                x_surr = rng.permutation(np.diff(x))
            else:
                x_surr = rng.permutation(x)
            if "PE" in measure:
                pe_surr = ml.modified_permutation_entropy(x_surr, m=D, tau=tau, norm=fafaeg)
            elif "Cd" in measure:
                # angulos_surr = angulos_alpha(x_surr,False)
                C,_ = ml.gamma_index_rank_ties(x_surr,max_gamma=D,mu=tau)
                pe_surr = 1-C[-1]
            elif "H_tau" in measure:
                pe_surr = ml.indice_S_eff_fast(x_surr, None, tau=tau,null="no",delta=True)
            elif "gamma" in measure:
                _,C = ml.gamma_index_rank_ties(x_surr,max_gamma=D,mu=tau)
                pe_surr = 1-C[-1]
            elif "J_tau" in measure: 
                pe_surr = ml.indice_J(x_surr, None, tau=tau)
            pe_surrogates[k] = pe_surr
    if type_null == "iaaft":
        if "interval" in measure:    
            x_surr = ml.iaaft(np.diff(x),n_surrogates)
        else:
            x_surr = ml.iaaft(x,n_surrogates)
        for k in range(n_surrogates):
            if "PE" in measure: 
                pe_surr = ml.modified_permutation_entropy(x_surr[k,:], m=D, tau=tau, norm=fafaeg)
            elif "gamma" in measure:
                _,C = ml.gamma_index_rank_ties(x_surr[k,:],max_gamma=D,mu=tau)
                pe_surr = 1-C[-1]
            elif "Cd" in measure:
                C,_ = ml.gamma_index_rank_ties(x_surr[k,:],max_gamma=D,mu=tau)
                pe_surr = 1-C[-1]
            elif "H_tau" in measure:
                pe_surr = ml.indice_S_eff_fast(x_surr[k,:], None, tau=tau,null="shuffle",delta=True)
            elif "J_tau" in measure:
                pe_surr = ml.indice_J(x_surr[k,:], None, tau=tau)
            pe_surrogates[k] = pe_surr



    mu_null = np.mean(pe_surrogates)
    sigma_null = np.std(pe_surrogates, ddof=1)

    if sigma_null == 0:
        z = np.nan
    else:
        z = (pe_obs - mu_null) / sigma_null

    # p-value corregido (+1)
    if alternative == "greater":
        p_value = (np.sum(pe_surrogates >= pe_obs) + 1) / (n_surrogates + 1)
    elif alternative == "less":
        p_value = (np.sum(pe_surrogates <= pe_obs) + 1) / (n_surrogates + 1)
    elif alternative == "two-sided":
        p_value = (np.sum(np.abs(pe_surrogates - mu_null) >= np.abs(pe_obs - mu_null)) + 1) / (n_surrogates + 1)
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")

    p_raw = empirical_p_raw(pe_surrogates, pe_obs, mu_null, alternative=alternative)

    return {
        "pe_obs": pe_obs,
        "mu_null": mu_null,
        "sigma_null": sigma_null,
        "z": z,
        "p_value": p_value,
        "p_raw": p_raw
    }

def build_cache_key(measure,D, tau, type_null, normalize, alternative):
    if "PE" in measure:
        return f"{measure}_D{D}_tau{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    elif "Cd" in measure:
        return f"{measure}_d{D+1}_mu{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    elif "gamma" in measure:  
        return f"{measure}_d{D}_mu{tau}_{type_null}_{alternative}_norm{int(normalize)}"
    else:
        return f"{measure}_{type_null}_{alternative}_norm{int(normalize)}"

def get_cache_path(measure,cache_dir, D, tau, type_null, normalize, alternative):
    os.makedirs(cache_dir, exist_ok=True)
    fname = build_cache_key(measure,D, tau, type_null, normalize, alternative) + ".pkl"
    return os.path.join(cache_dir, fname)

def compute_panel_dataframe(
    composers,
    datos_composers,
    measure,
    D,
    tau,
    n_surrogates,
    type_null,
    normalize,
    alternative,
    random_state
):
    """
    Calcula un DataFrame con una fila por serie.
    """
    composer_names, _ = composer_labels(datos_composers)

    rows = []
    global_counter = 0
    seed_base = int(random_state) if random_state is not None else None

    for composer in composer_names:
        if composer not in composers:
            continue
        print(composer)

        meta = datos_composers.get(composer, {})
        birth_year = meta.get("Birth_year", np.nan)
        composer_index = meta.get("Indice", np.nan)

        series_names = sorted(composers[composer].keys(), key=natural_key)

        for sname in series_names:
            x = np.asarray(composers[composer][sname], dtype=float)

            if seed_base is None:
                seed_here = None
            else:
                seed_here = seed_base + 100000 * D + 1000 * tau + global_counter

            global_counter += 1

            stats = pe_stats_for_series(
                x=x,
                measure=measure,
                D=D,
                tau=tau,
                n_surrogates=n_surrogates,
                type_null = type_null,
                alternative=alternative,
                random_state=seed_here
            )

            rows.append({
                "composer": composer,
                "birth_year": birth_year,
                "composer_index": composer_index,
                "serie": sname,
                "length": len(x),
                "D": D,
                "tau": tau,
                "n_surrogates": n_surrogates,
                "normalize": normalize,
                "alternative": alternative,
                "random_state": seed_here,
                "pe_obs": stats["pe_obs"],
                "mu_null": stats["mu_null"],
                "sigma_null": stats["sigma_null"],
                "z": stats["z"],
                "p_value": stats["p_value"],
                "p_raw": stats["p_raw"]
            })

    df = pd.DataFrame(rows)
    return df

def get_or_compute_panel_dataframe(
    composers,
    datos_composers,
    measure,
    D,
    tau,
    n_surrogates,
    type_null,
    normalize,
    alternative,
    random_state,
    cache_dir=CACHE_DIR,
    force_recompute=FORCE_RECOMPUTE
):
    """
    Si ya existe el caché para esos parámetros, lo carga.
    Si no existe, calcula, guarda y devuelve.
    """
    cache_path = get_cache_path(
        measure=measure,
        cache_dir=cache_dir,
        D=D,
        tau=tau,
        type_null=type_null,
        normalize=normalize,
        alternative=alternative,
    )

    if (not force_recompute) and os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        print(f"[CACHE] Cargado: {cache_path}")
        return df

    print(f"[CACHE] Calculando panel {measure} D={D}, tau={tau} {type_null}...")
    df = compute_panel_dataframe(
        composers=composers,
        datos_composers=datos_composers,
        measure=measure,
        D=D,
        tau=tau,
        n_surrogates=n_surrogates,
        type_null = type_null,
        normalize=normalize,
        alternative=alternative,
        random_state=random_state
    )
    df.to_pickle(cache_path)
    print(f"[CACHE] Guardado: {cache_path}")
    return df

def point_is_red(p_raw, p_value, use_p_raw_zero=True, alpha=0.05):
    if use_p_raw_zero:
        return (p_raw == 0.0)
    return (p_value <= alpha)

def plot_panel(ax, df_panel, composer_names, title,
               use_p_raw_zero=True, alpha=0.05):
    """
    df_panel: DataFrame con una fila por serie para un panel fijo (D, tau).
    """
    medianas = []
    all_z = []

    total_red_points = 0
    total_points = 0

    for i, composer in enumerate(composer_names, start=1):
        sub = df_panel[df_panel["composer"] == composer].copy()

        if sub.empty:
            medianas.append(np.nan)
            continue

        zvals = sub["z"].to_numpy(dtype=float)
        pvals = sub["p_value"].to_numpy(dtype=float)
        praws = sub["p_raw"].to_numpy(dtype=float)

        finite_mask = np.isfinite(zvals)
        zvals = zvals[finite_mask]
        pvals = pvals[finite_mask]
        praws = praws[finite_mask]

        if zvals.size == 0:
            medianas.append(np.nan)
            continue

        red_mask = np.array([
            point_is_red(pr, pv, use_p_raw_zero=use_p_raw_zero, alpha=alpha)
            for pr, pv in zip(praws, pvals)
        ], dtype=bool)

        colors = np.where(red_mask, "red", "blue")
        x = np.random.normal(i, JITTER_STD, size=zvals.size)

        ax.scatter(
            x, 1 - zvals,
            s=DOT_SIZE,
            alpha=EDGE_ALPHA,
            facecolors="none",
            edgecolors=colors,
            linewidths=0.8
        )

        med = np.median(zvals)
        medianas.append(med)
        all_z.extend(zvals.tolist())

        num_red = np.sum(red_mask)
        total_red_points += num_red
        total_points += zvals.size

        percentage_red = 100.0 * num_red / zvals.size
        if i % 2 != 0:
            y_offset = 0.02
        else:
            y_offset = 0.07
        ax.text(
            i, y_offset, f"{percentage_red:.1f}%",
            ha="center", va="bottom",
            fontsize=PERCENT_FONTSIZE,
            color="black",
            transform=ax.get_xaxis_transform()
        )

    idxs = np.arange(1, len(composer_names) + 1)
    medianas = np.array(medianas, dtype=float)

    ax.axhline(0, color="gray", linestyle="--", linewidth=LINE_WIDTH, alpha=0.9)

    if np.isfinite(medianas).any():
        ax.scatter(
            idxs, 1- medianas,
            color="black",
            s=DOT_SIZE,
            zorder=3,
            marker="o",
            label="Mediana"
        )

    if use_p_raw_zero:
        red_label = r"Significativa ($p_{\mathrm{raw}}=0$)"
    else:
        red_label = fr"Significativa ($p \leq {alpha}$)"

    ax.plot([], [], marker='o', color='none', markeredgecolor='red', label=red_label)
    ax.plot([], [], marker='o', color='none', markeredgecolor='blue', label='No significativa')
    ax.plot([], [], marker='none', color='none', label='%: series rojas')

    if total_points > 0:
        total_percentage_red = 100.0 * total_red_points / total_points
        ax.text(
            0.01, 0.98,
            f"Total: {total_percentage_red:.1f}%",
            ha="left", va="top",
            fontsize=FONT_GENERAL,
            color="black",
            transform=ax.transAxes
        )

    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(axis='y', alpha=0.4)
    ax.set_xticks(np.arange(1, len(composer_names) + 1))
    ax.tick_params(axis='both', labelsize=FONT_TICKS)

    return np.array(all_z, dtype=float)

# =========================================================
# OBTENER DATOS DESDE CACHÉ O CÁLCULO
# =========================================================
# Se asume que YA existen:
#   composers
#   datos_composers
#   permutation_entropy(...)

# composers, datos_composers = extraer_dataset_musica()
# Carpeta de salida
carpeta_salida = r"data/dataset_procesado"
# os.makedirs(carpeta_salida, exist_ok=True)

# # Guardar composers
# with open(os.path.join(carpeta_salida, "composers.pkl"), "wb") as f:
#     pickle.dump(composers, f, protocol=pickle.HIGHEST_PROTOCOL)

# # Guardar datos_composers
# with open(os.path.join(carpeta_salida, "datos_composers.pkl"), "wb") as f:
#     pickle.dump(datos_composers, f, protocol=pickle.HIGHEST_PROTOCOL)
# Cargar composers
with open(os.path.join(carpeta_salida, "composers.pkl"), "rb") as f:
    composers = pickle.load(f)

# Cargar datos_composers
with open(os.path.join(carpeta_salida, "datos_composers.pkl"), "rb") as f:
    datos_composers = pickle.load(f)
composer_names, labels = composer_labels(datos_composers)

panel_dfs = []
panel_measure_dfs = []
all_z_global = []

for cfg in PANEL_CONFIGS:
    df_panel = get_or_compute_panel_dataframe(
        composers=composers,
        datos_composers=datos_composers,
        measure=cfg["measure"],
        D=cfg["D"],
        tau=cfg["tau"],
        n_surrogates=N_SURROGATES,
        type_null=cfg["type_null"],
        normalize=NORMALIZE,
        alternative=ALTERNATIVE,
        random_state=RANDOM_STATE,
        cache_dir=CACHE_DIR,
        force_recompute=FORCE_RECOMPUTE
    )
    panel_dfs.append(df_panel)

    # construir DataFrame de medidas y guardarlo en CSV
    df_measure, csv_path = save_panel_measure_csv(
        df_panel=df_panel,
        composer_names=composer_names,
        cfg=cfg,
        export_dir=EXPORT_DIR
    )
    panel_measure_dfs.append(df_measure)

# =========================================================
# PLOTEO 2x2
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)

for ax, cfg, df_panel in zip(axs.ravel(), PANEL_CONFIGS, panel_dfs):
    zvals_panel = plot_panel(
        ax=ax,
        df_panel=df_panel,
        composer_names=composer_names,
        title=cfg["title"],
        use_p_raw_zero=USE_P_RAW_ZERO_FOR_RED,
        alpha=ALPHA
    )
    if zvals_panel.size > 0:
        all_z_global.extend(zvals_panel[np.isfinite(zvals_panel)].tolist())

all_z_global = np.array(all_z_global, dtype=float)
if all_z_global.size > 0:
    zmax = np.nanmax(np.abs(all_z_global))
    if np.isfinite(zmax) and zmax > 0:
        pad = 0.08 * zmax
        ylim = (-zmax - pad, zmax + pad)
        for ax in axs.ravel():
            ax.set_ylim(*ylim)

for ax in axs.ravel():
    ax.set_ylabel("Z-score of mPE", fontsize=FONT_GENERAL)

xticks = np.arange(1, len(labels) + 1)
for ax in axs[1, :]:
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=FONT_TICKS)

handles, lab = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles, lab,
    loc='upper center',
    ncol=3,
    fontsize=LEGEND_FONTSIZE,
    frameon=False
)

plt.subplots_adjust(
    left=LEFT_MARGIN,
    right=RIGHT_MARGIN,
    bottom=BOTTOM_MARGIN,
    top=TOP_MARGIN,
    wspace=0.08,
    hspace=0.18
)

plt.margins(x=0.02)
plt.show()

medianas_dict = {}

for cfg, df_panel in zip(PANEL_CONFIGS, panel_dfs):
    nombre_panel = cfg["title"]
    medianas_dict[nombre_panel] = medianas_por_compositor(df_panel, composer_names)

df_medianas = pd.DataFrame(medianas_dict, index=composer_names)

print("\nMedianas por compositor:")
print(df_medianas)

print("\nCorrelaciones de Spearman entre series de medianas:")
for i in range(len(df_medianas.columns)):
    for j in range(i + 1, len(df_medianas.columns)):
        c1 = df_medianas.columns[i]
        c2 = df_medianas.columns[j]

        pares = df_medianas[[c1, c2]].dropna()
        if len(pares) < 2:
            print(f"{c1} vs {c2}: insuficientes datos")
            continue

        r, p = spearmanr(pares[c1], pares[c2])
        print(f"{c1} vs {c2}: r = {r:.6f}, p = {p:.6g}")

print("\nCorrelaciones de Spearman entre series de medianas:")
for i in range(len(df_medianas.columns)):
    for j in range(i + 1, len(df_medianas.columns)):
        c1 = df_medianas.columns[i]
        c2 = df_medianas.columns[j]

        pares = df_medianas[[c1, c2]].dropna()
        if len(pares) < 2:
            print(f"{c1} vs {c2}: insuficientes datos")
            continue

        rho, p = spearmanr(pares[c1], pares[c2])
        print(f"{c1} vs {c2}: rho = {rho:.6f}, p = {p:.6g}")
