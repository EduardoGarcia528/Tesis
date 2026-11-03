# -*- coding: utf-8 -*-
"""
cargar_serie_corpus.py

Utilidad mínima para cargar cualquier melodía del corpus
guardado como:

    data/composers_hybrid.pkl
    data/composers_complexity.pkl

Uso:
    x = obtener_serie("hybrid", "Bach", 3)

o:
    x = obtener_serie("complexity", "Mozart", "Serie_12")
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mi_libreria as ml


DATA_DIR = "data"


def cargar_corpus(method="hybrid", data_dir=DATA_DIR):
    """
    Carga el diccionario de melodías.

    Parameters
    ----------
    method : {"hybrid", "complexity"}
        Versión del corpus.
    data_dir : str
        Carpeta donde están los archivos .pkl.

    Returns
    -------
    composers : dict
        Estructura:
            composers[compositor][serie] -> array 1D
    """
    method = method.lower()

    if method not in ("hybrid", "complexity"):
        raise ValueError("method debe ser 'hybrid' o 'complexity'.")

    path = os.path.join(data_dir, f"composers_{method}.pkl")

    with open(path, "rb") as f:
        composers = pickle.load(f)

    return composers


def obtener_serie(method, compositor, serie, data_dir=DATA_DIR):
    """
    Devuelve una melodía como numpy array 1D.

    Parameters
    ----------
    method : {"hybrid", "complexity"}
    compositor : str
        Ejemplo: "Bach", "Mozart", "Debussy".
        No distingue mayúsculas/minúsculas.
    serie : int o str
        Ejemplos:
            3
            "Serie_3"

    Returns
    -------
    x : np.ndarray
        Serie melódica 1D.
    """
    composers = cargar_corpus(method, data_dir=data_dir)

    # Buscar compositor sin importar mayúsculas/minúsculas
    mapa_compositores = {c.lower(): c for c in composers.keys()}

    key_comp = compositor.lower()
    if key_comp not in mapa_compositores:
        disponibles = ", ".join(sorted(composers.keys()))
        raise KeyError(
            f"Compositor '{compositor}' no encontrado.\n"
            f"Disponibles: {disponibles}"
        )

    compositor_real = mapa_compositores[key_comp]

    # Permitir serie=3 o serie="Serie_3"
    if isinstance(serie, (int, np.integer)):
        serie = f"Serie_{int(serie)}"

    if serie not in composers[compositor_real]:
        disponibles = ", ".join(composers[compositor_real].keys())
        raise KeyError(
            f"Serie '{serie}' no encontrada para {compositor_real}.\n"
            f"Disponibles: {disponibles}"
        )

    return np.asarray(
        composers[compositor_real][serie],
        dtype=float
    )



def histograma_corpus(
    method="hybrid",
    data_dir="data",
    bins=30,
    xlabel="Valor",
    title=None
):
    # -------------------------
    # Cargar corpus
    # -------------------------
    path = os.path.join(
        data_dir,
        f"composers_{method}.pkl"
    )

    with open(path, "rb") as f:
        composers = pickle.load(f)

    # -------------------------
    # Calcular medida
    # -------------------------
    resultados = []

    # for compositor in composers:
    #     print(compositor)

    #     for serie in composers[compositor]:

    #         x = np.asarray(
    #             composers[compositor][serie],
    #             dtype=float
    #         )

    #         try:
    #             valor = ml.indice_no_linealidad_mdfa(x)[0]

    #             if np.isfinite(valor):
    #                 resultados.append(
    #                     (compositor, serie, valor)
    #                 )

    #         except Exception as e:
    #             print(
    #                 f"Error en {compositor}, {serie}: {e}"
    #             )

    # valores = np.array(
    #     [r[2] for r in resultados],
    #     dtype=float
    # )
    valores = np.load(f"xi_index_{method}.npy")
    # -------------------------
    # Histograma
    # -------------------------
    plt.figure(figsize=(7, 5))

    plt.hist(
        valores,
        bins=bins,
        edgecolor="black"
    )
    # np.save(f"xi_index_{method}.npy",valores)

    plt.xlabel(xlabel)
    plt.ylabel("Número de melodías")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()

    print("Melodías analizadas:", len(valores))

    return valores, resultados


if __name__ == "__main__":

    # Ejemplo:
    # x = obtener_serie(
    #     method="hybrid",
    #     compositor="Bach",
    #     serie=1
    # )

    histograma_corpus(
        # method="hybrid",
        method="complexity",
        bins=60,
        xlabel="Índice de no linealidad (MFDFA)",
        title="Distribución del índice de no linealidad en el corpus"
    )
