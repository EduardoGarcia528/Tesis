import pandas as pd
import numpy as np
import re
import ast
import unicodedata
import copy

COMPOSER_METADATA = {
    # Renacimiento
    "dufay": (1397, "Renacimiento"),
    "morales": (1500, "Renacimiento"),
    "josquin_des_prez": (1450, "Renacimiento"),
    "palestrina": (1525, "Renacimiento"),
    "lasso": (1532, "Renacimiento"),
    "byrd": (1540, "Renacimiento"),
    "victoria": (1548, "Renacimiento"),
    "gabrieli_a": (1533, "Renacimiento"),
    "gabrieli_g": (1557, "Renacimiento"),
    "dowland": (1563, "Renacimiento"),
    "gesualdo": (1566, "Renacimiento"),

    # Barroco
    "monteverdi": (1567, "Barroco"),
    "frescobaldi": (1583, "Barroco"),
    "scheidt": (1587, "Barroco"),
    "froberger": (1616, "Barroco"),
    "anglebert": (1629, "Barroco"),
    "lully": (1632, "Barroco"),
    "buxtehude": (1637, "Barroco"),
    "pachelbel": (1653, "Barroco"),
    "couperin_f": (1668, "Barroco"),
    "couperin": (1668, "Barroco"),
    "albinoni": (1671, "Barroco"),
    "vivaldi": (1678, "Barroco"),
    "telemann": (1681, "Barroco"),
    "dandrieu": (1682, "Barroco"),
    "rameau": (1683, "Barroco"),
    "bach_js": (1685, "Barroco"),
    "handel": (1685, "Barroco"),
    "scarlatti": (1685, "Barroco"),
    "zipoli": (1688, "Barroco"),

    # Clasicismo
    "haydn": (1732, "Clasicismo"),
    "albrechtsberger": (1736, "Clasicismo"),
    "clementi": (1752, "Clasicismo"),
    "mozart": (1756, "Clasicismo"),
    "beethoven": (1770, "Clasicismo"),

    # Romanticismo
    "cramer": (1771, "Romanticismo"),
    "paganini": (1782, "Romanticismo"),
    "schubert": (1797, "Romanticismo"),
    "berlioz": (1803, "Romanticismo"),
    "mendelssohn": (1809, "Romanticismo"),
    "chopin": (1810, "Romanticismo"),
    "schumann": (1810, "Romanticismo"),
    "liszt": (1811, "Romanticismo"),
    "alkan": (1813, "Romanticismo"),
    "franck": (1822, "Romanticismo"),
    "bruckner": (1824, "Romanticismo"),
    "gottschalk": (1829, "Romanticismo"),
    "brahms": (1833, "Romanticismo"),
    "saint_saens": (1835, "Romanticismo"),
    "guilmant": (1837, "Romanticismo"),
    "bizet": (1838, "Romanticismo"),

    # Tardorromanticismo / Nacionalismo
    "mussorgsky": (1839, "Tardorromanticismo_Nacionalismo"),
    "tchaikovsky": (1840, "Tardorromanticismo_Nacionalismo"),
    "dvorak": (1841, "Tardorromanticismo_Nacionalismo"),
    "grieg": (1843, "Tardorromanticismo_Nacionalismo"),
    "faure": (1845, "Tardorromanticismo_Nacionalismo"),
    "janacek": (1854, "Tardorromanticismo_Nacionalismo"),
    "albeniz": (1860, "Tardorromanticismo_Nacionalismo"),
    "mahler": (1860, "Tardorromanticismo_Nacionalismo"),
    "debussy": (1862, "Tardorromanticismo_Nacionalismo"),
    "busoni": (1866, "Tardorromanticismo_Nacionalismo"),
    "satie": (1866, "Tardorromanticismo_Nacionalismo"),
    "joplin": (1868, "Tardorromanticismo_Nacionalismo"),
    "godowsky": (1870, "Tardorromanticismo_Nacionalismo"),
    "scriabin": (1872, "Tardorromanticismo_Nacionalismo"),
    "rachmaninoff": (1873, "Tardorromanticismo_Nacionalismo"),
    "reger": (1873, "Tardorromanticismo_Nacionalismo"),

    # Modernismo / siglo XX
    "schoenberg": (1874, "Modernismo_Siglo_XX"),
    "ravel": (1875, "Modernismo_Siglo_XX"),
    "karg": (1877, "Modernismo_Siglo_XX"),
    "respighi": (1879, "Modernismo_Siglo_XX"),
    "medtner": (1880, "Modernismo_Siglo_XX"),
    "bartok": (1881, "Modernismo_Siglo_XX"),
    "stravinsky": (1882, "Modernismo_Siglo_XX"),
    "prokofiev": (1891, "Modernismo_Siglo_XX"),
    "hindemith": (1895, "Modernismo_Siglo_XX"),
    "gershwin": (1898, "Modernismo_Siglo_XX"),
    "shostakovich": (1906, "Modernismo_Siglo_XX"),
    "messiaen": (1908, "Modernismo_Siglo_XX"),
}

ERA_ORDER = {
    "Renacimiento": 0,
    "Barroco": 1,
    "Clasicismo": 2,
    "Romanticismo": 3,
    "Tardorromanticismo_Nacionalismo": 4,
    "Modernismo_Siglo_XX": 5,
    "Desconocida": 99,
}

def normalize_label(s):
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def capitalize_label(s):
    """
    Convierte bach_js -> Bach_js.
    Mantiene formato simple para que sea parecido a tu función original.
    """
    return str(s).capitalize()


def parse_sequence_cell(x, dtype=float):
    """
    Convierte celdas tipo:
        Int64[60, 62, 64]
        Float64[0.0, 0.5]
        [60, 62, 64]
        60, 62, 64
    en np.array.
    """
    if isinstance(x, np.ndarray):
        return x.astype(dtype)

    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=dtype)

    if pd.isna(x):
        return np.array([], dtype=dtype)

    s = str(x).strip()

    if s in ["", "[]", "Int64[]", "Float64[]"]:
        return np.array([], dtype=dtype)

    # Int64[1, 2, 3] -> [1, 2, 3]
    s2 = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*\[(.*)\]$", r"[\1]", s)

    try:
        arr = ast.literal_eval(s2)
        return np.asarray(arr, dtype=dtype)
    except Exception:
        # Respaldo robusto: extrae números
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        return np.asarray(nums, dtype=dtype)


def classify_composer(label):
    """
    Clasifica cada etiqueta original en:
        source_composer
        contributor
        relation
        work_tag
        composer_clean
        composer_attribution
        is_collaboration
    """

    label = normalize_label(label)

    alias = {
        # Josquin / des Prez
        "desprez": "josquin_des_prez",
        "josquin": "josquin_des_prez",
        "pres_des": "josquin_des_prez",

        # Palestrina
        "palestina": "palestrina",
        "palestrina": "palestrina",

        # Dowland
        "downland": "dowland",
        "dowland": "dowland",

        # Handel / Haendel
        "haendel": "handel",
        "handel": "handel",

        # Mendelssohn
        "mendellsohn": "mendelssohn",
        "mendelssohn": "mendelssohn",

        # Mussorgsky
        "moussorgski": "mussorgsky",
        "mussorgskij": "mussorgsky",
        "mussorgsky": "mussorgsky",

        # Tchaikovsky
        "tchaikovsky": "tchaikovsky",
        "tchaikowski": "tchaikovsky",
        "tchaikowsky": "tchaikovsky",
        "tchajkowski": "tchaikovsky",
        "tchajkowskij": "tchaikovsky",
        "tchajkowsky": "tchaikovsky",

        # Rachmaninoff
        "rachmaninoff": "rachmaninoff",
        "rachmaninov": "rachmaninoff",

        # Schoenberg
        "schoenberg": "schoenberg",
        "schonberg": "schoenberg",

        # Saint-Saëns
        "saint": "saint_saens",
        "saint_saens": "saint_saens",

        #Scarlatti
        "scarlatti": "scarlatti",
        "scarlatti_d": "scarlatti",

    }

    collab_map = {
        "bach_karageanes": ("bach_js", "karageanes", "arrangement_or_edition"),
        "bach_liszt": ("bach_js", "liszt", "arrangement"),
        "bach_mozart": ("bach_js", "mozart", "arrangement"),
        "bach_oguri": ("bach_js", "oguri", "arrangement_or_edition"),

        "handel_oguri": ("handel", "oguri", "arrangement_or_edition"),

        "mozart_czerny": ("mozart", "czerny", "arrangement"),
        "mozart_oguri": ("mozart", "oguri", "arrangement_or_edition"),

        "beethoven_laviano": ("beethoven", "laviano", "arrangement_or_edition"),
        "beethoven_liszt": ("beethoven", "liszt", "arrangement"),
        "beethoven_saint_saens": ("beethoven", "saint_saens", "arrangement"),
        "beethoven_simonetto": ("beethoven", "simonetto", "arrangement_or_edition"),

        "schubert_liszt": ("schubert", "liszt", "arrangement"),
        "mendellsohn_liszt": ("mendelssohn", "liszt", "arrangement"),
        "schumann_karageanes": ("schumann", "karageanes", "arrangement_or_edition"),
        "brahms_karageanes": ("brahms", "karageanes", "arrangement_or_edition"),

        "tchaikovsky_grainger": ("tchaikovsky", "grainger", "arrangement"),
        "tchaikovsky_rachmaninov": ("tchaikovsky", "rachmaninoff", "arrangement"),

        "dvorak_hilkemeijer": ("dvorak", "hilkemeijer", "arrangement_or_edition"),

        "albeniz_godowsky": ("albeniz", "godowsky", "arrangement"),
        "stravinsky_karageanes": ("stravinsky", "karageanes", "arrangement_or_edition"),

        "debussy_ravel": ("debussy", "ravel", "arrangement"),
    }

    work_map = {
        "palestrina_missa": ("palestrina", "missa"),
        "palestrina_o": ("palestrina", "o"),
        "palestrina_sicut": ("palestrina", "sicut"),
        "palestrina_surge": ("palestrina", "surge"),

        "vivaldi_credo_01_credo": ("vivaldi", "credo_01"),
        "vivaldi_credo_02_et": ("vivaldi", "credo_02"),
        "vivaldi_credo_04_et": ("vivaldi", "credo_04"),

        "albeniz_m": ("albeniz", "m"),
    }

    if label in collab_map:
        source, contributor, relation = collab_map[label]
        source = alias.get(source, source)
        contributor = alias.get(contributor, contributor)

        return pd.Series({
            "source_composer": source,
            "contributor": contributor,
            "relation": relation,
            "work_tag": np.nan,
            "composer_clean": source,
            "composer_attribution": f"{source}__{contributor}",
            "is_collaboration": True,
        })

    if label in work_map:
        source, work_tag = work_map[label]
        source = alias.get(source, source)

        return pd.Series({
            "source_composer": source,
            "contributor": np.nan,
            "relation": "work_subset",
            "work_tag": work_tag,
            "composer_clean": source,
            "composer_attribution": source,
            "is_collaboration": False,
        })

    if label == "trad_hymn":
        return pd.Series({
            "source_composer": "traditional",
            "contributor": np.nan,
            "relation": "traditional",
            "work_tag": "hymn",
            "composer_clean": "traditional",
            "composer_attribution": "traditional",
            "is_collaboration": False,
        })

    clean = alias.get(label, label)

    return pd.Series({
        "source_composer": clean,
        "contributor": np.nan,
        "relation": "original",
        "work_tag": np.nan,
        "composer_clean": clean,
        "composer_attribution": clean,
        "is_collaboration": False,
    })


def extraer_dataset_musica_originales(
    csv_path="melodies_found.csv",
    sequence_col="pitches",
    methods=("hybrid", "complexity"),
    min_len=800,
    min_piezas=30,
    incluir_work_subset=True,
    incluir_traditional=False,
    dtype=float,
    verbose=True,
):
    """
    Construye diccionarios tipo composers y datos_composers desde melodies_found.csv.

    Parámetros
    ----------
    csv_path : str
        Ruta del archivo melodies_found.csv.

    sequence_col : str
        Columna usada como serie. Por defecto: "pitches".
        También podrías usar "durations" u "onsets".

    methods : tuple
        Versiones que existen por pieza. Por defecto: ("hybrid", "complexity").

    min_len : int
        Longitud mínima de la serie para conservar una pieza.
        En tu función anterior usabas len(serie)//2 < 400.
        Si ahora usamos solo pitches, el equivalente natural es len(pitches) < 800.

    min_piezas : int
        Número mínimo de piezas por compositor después de depurar por longitud.

    incluir_work_subset : bool
        Si True, conserva etiquetas como palestrina_missa o vivaldi_credo como obras propias
        del compositor base.

    incluir_traditional : bool
        Si False, excluye trad_hymn porque no es compositor individual.

    dtype : type
        Tipo numérico de salida.

    Retorna
    -------
    composers : dict
        composers[method][composer]["Serie_i"] = np.array

    datos_composers : dict
        datos_composers[method][composer]["# Piezas"], ["Indice"], etc.

    df_main : pd.DataFrame
        DataFrame limpio y filtrado.
    """

    df = pd.read_csv(csv_path)

    required_cols = {"filename", "composer", sequence_col, "method"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Guardar etiqueta cruda
    df["composer_raw"] = df["composer"].copy()
    df["composer_norm"] = df["composer_raw"].apply(normalize_label)

    # Clasificar compositor
    clean_cols = df["composer_norm"].apply(classify_composer)
    df = pd.concat([df, clean_cols], axis=1)

    # Filtro: quedarse con obras propias originales
    relations_keep = ["original"]

    if incluir_work_subset:
        relations_keep.append("work_subset")

    if incluir_traditional:
        relations_keep.append("traditional")

    df_main = df[df["relation"].isin(relations_keep)].copy()

    # Excluir colaboraciones/arreglos explícitamente
    df_main = df_main[~df_main["is_collaboration"]].copy()

    # Excluir traditional si se pidió
    if not incluir_traditional:
        df_main = df_main[df_main["composer_clean"] != "traditional"].copy()

    # Filtrar métodos
    df_main = df_main[df_main["method"].isin(methods)].copy()

    # Parsear secuencias
    df_main["_sequence"] = df_main[sequence_col].apply(lambda x: parse_sequence_cell(x, dtype=dtype))
    df_main["_length"] = df_main["_sequence"].apply(len)

    # Depuración por longitud mínima
    df_main = df_main[df_main["_length"] >= min_len].copy()

    # Orden estable
    df_main = df_main.sort_values(["method", "composer_clean", "filename"]).reset_index(drop=True)

    composers = {}
    datos_composers = {}

    for method in methods:
        df_m = df_main[df_main["method"] == method].copy()

        composers[method] = {}
        datos_composers[method] = {}

        # Primero filtramos compositores con suficientes piezas
        counts = df_m.groupby("composer_clean")["filename"].nunique()
        valid_composers = counts[counts >= min_piezas].index.tolist()

        df_m = df_m[df_m["composer_clean"].isin(valid_composers)].copy()

        # Construir diccionarios
        for indice, composer_clean in enumerate(sorted(valid_composers)):
            df_c = df_m[df_m["composer_clean"] == composer_clean].copy()
            df_c = df_c.sort_values("filename").reset_index(drop=True)

            composer_key = capitalize_label(composer_clean)

            composers[method][composer_key] = {}

            for i, row in df_c.iterrows():
                serie_key = f"Serie_{i + 1}"
                composers[method][composer_key][serie_key] = row["_sequence"]

            datos_composers[method][composer_key] = {
                "# Piezas": len(df_c),
                "Indice": indice,
                "Composer_clean": composer_clean,
                "Method": method,
                "Filenames": df_c["filename"].tolist(),
                "Longitudes": df_c["_length"].tolist(),
            }

    if verbose:
        print("Resumen del conjunto depurado")
        print("--------------------------------")
        print("Columna usada como serie:", sequence_col)
        print("Longitud mínima:", min_len)
        print("Mínimo de piezas por compositor:", min_piezas)
        print()

        for method in methods:
            n_comp = len(composers[method])
            n_piezas = sum(datos_composers[method][c]["# Piezas"] for c in datos_composers[method])

            print(f"Method: {method}")
            print(" # de compositores restantes:", n_comp)
            print(" # de piezas restantes:", n_piezas)
            print()

    return composers, datos_composers, df_main

def ordenar_diccionarios_por_epoca(composers, datos_composers):
    composers_ord = {}
    datos_ord = {}

    for method in composers.keys():
        keys = list(composers[method].keys())

        def sort_key(composer_key):
            clean = composer_key.lower()
            birth_year, era = COMPOSER_METADATA.get(clean, (9999, "Desconocida"))
            return (ERA_ORDER.get(era, 99), birth_year, clean)

        ordered_keys = sorted(keys, key=sort_key)

        composers_ord[method] = {}
        datos_ord[method] = {}

        for new_index, composer_key in enumerate(ordered_keys):
            clean = composer_key.lower()
            birth_year, era = COMPOSER_METADATA.get(clean, (np.nan, "Desconocida"))

            composers_ord[method][composer_key] = composers[method][composer_key]

            datos_ord[method][composer_key] = datos_composers[method][composer_key].copy()
            datos_ord[method][composer_key]["Indice"] = new_index
            datos_ord[method][composer_key]["Birth_year"] = birth_year
            datos_ord[method][composer_key]["Era"] = era
            datos_ord[method][composer_key]["Era_order"] = ERA_ORDER.get(era, 99)

    return composers_ord, datos_ord

composers, datos_composers, df_originales = extraer_dataset_musica_originales(
    csv_path="melodies_found.csv",
    sequence_col="pitches",
    min_len=500,
    min_piezas=30,
    incluir_work_subset=True,
    incluir_traditional=False,
)

composers, datos_composers = ordenar_diccionarios_por_epoca(
    composers,
    datos_composers
)

print(composers["hybrid"].keys())
print(composers["complexity"].keys())

# print(datos_composers["hybrid"]["Bach_js"]["Birth_year"])

# print(datos_composers["hybrid"]["Scarlatti"])
# print(datos_composers["hybrid"]["Scarlatti_d"])

import os
import pickle
carpeta_salida = r"data"
# Guardar composers
with open(os.path.join(carpeta_salida, "composers_complex.pkl"), "wb") as f:
    pickle.dump(composers["complexity"], f, protocol=pickle.HIGHEST_PROTOCOL)

# Guardar datos_composers
with open(os.path.join(carpeta_salida, "datos_composers_complex.pkl"), "wb") as f:
    pickle.dump(datos_composers["complexity"], f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(carpeta_salida, "composers_hybrid.pkl"), "wb") as f:
    pickle.dump(composers["hybrid"], f, protocol=pickle.HIGHEST_PROTOCOL)

# Guardar datos_composers
with open(os.path.join(carpeta_salida, "datos_composers_hybrid.pkl"), "wb") as f:
    pickle.dump(datos_composers["hybrid"], f, protocol=pickle.HIGHEST_PROTOCOL)