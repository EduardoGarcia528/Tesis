# -*- coding: utf-8 -*-
"""
Rehace las figuras de Delta RMSE usando puntos individuales en vez de boxplots.

Cada punto corresponde a una voz de un movimiento del corpus.

Se generan cuatro figuras:
    1. Delta RMSE por compositor
    2. Delta RMSE por periodo/epoca
    3. Delta RMSE por voz/instrumento
    4. Delta RMSE vs ano de composicion de la pieza

Definicion:
    Delta RMSE = RMSE_exp - RMSE_powerlaw

Por tanto, el signo de Delta RMSE indica cual de los dos ajustes tuvo menor
RMSE, pero las figuras no incluyen textos interpretativos adicionales.

Requiere:
    numpy
    pandas
    matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path("figuras_C1null_shuffle_K0")

INPUT_CSV = (
    BASE_DIR
    / "resumen_seleccion_modelo_Cd_observado_y_shuffle_exponencial.csv"
)

# CSV del corpus al que se agrego la columna composition_year.
# Se busca en esta ruta y, si no existe, por nombre en el directorio actual.
CORPUS_CSV = Path(
    "quartet_subcorpus_all_movements_with_composition_year.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "distribucion_exponencial_vs_powerlaw"
    / "figuras_delta_rmse_puntos"
)

PNG_DPI = 240
RNG_SEED = 12345

# Amplitud del desplazamiento vertical aleatorio de los puntos.
JITTER = 0.16
POINT_SIZE = 22
POINT_ALPHA = 0.55

VOICE_ORDER = ["vln I", "vln II", "viola", "cello"]

# Si queda vacio, se conserva el orden de aparicion en el CSV.
# Ejemplo:
# PERIOD_ORDER = ["Baroque", "Classical", "Romantic", "Modern"]
PERIOD_ORDER: list[str] = []

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


# ============================================================
# CARGA DE DATOS
# ============================================================


def locate_csv(path: Path) -> Path:
    """Busca primero la ruta configurada y luego el mismo archivo en cwd."""
    candidates = [path, Path(path.name)]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No se encontro el CSV. Se intento:\n"
        + "\n".join(f"  - {p.resolve()}" for p in candidates)
    )


def _clean_string_key(series: pd.Series) -> pd.Series:
    """Normaliza claves de texto antes de unir los dos CSV."""
    return series.fillna("").astype(str).str.strip()


def add_composition_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega composition_year desde el CSV actualizado del corpus.

    La union intenta, en este orden, claves cada vez menos especificas.
    En el corpus actual, work_id es la opcion preferida porque identifica
    univocamente la obra y el ano es el mismo para todos sus movimientos.
    """
    if "composition_year" in df.columns:
        out = df.copy()
        out["composition_year"] = pd.to_numeric(
            out["composition_year"], errors="coerce"
        )
        return out

    corpus_path = locate_csv(CORPUS_CSV)
    corpus = pd.read_csv(corpus_path)

    if "composition_year" not in corpus.columns:
        raise KeyError(
            f"El CSV del corpus {corpus_path} no contiene 'composition_year'."
        )

    # Claves candidatas. Se usa la primera que exista en ambos CSV.
    candidate_keys = [
        ["work_id"],
        ["filename"],
        ["composer", "work_title"],
        ["composer", "catalogue"],
    ]

    for keys in candidate_keys:
        if not all(k in df.columns and k in corpus.columns for k in keys):
            continue

        left = df.copy()
        right = corpus[keys + ["composition_year"]].copy()

        # Normalizar solo las claves de texto.
        for key in keys:
            if (
                pd.api.types.is_object_dtype(left[key])
                or pd.api.types.is_string_dtype(left[key])
                or pd.api.types.is_object_dtype(right[key])
                or pd.api.types.is_string_dtype(right[key])
            ):
                left[key] = _clean_string_key(left[key])
                right[key] = _clean_string_key(right[key])

        right["composition_year"] = pd.to_numeric(
            right["composition_year"], errors="coerce"
        )

        # Comprobar que una misma clave no apunte a anos distintos.
        n_years = (
            right.dropna(subset=["composition_year"])
            .groupby(keys, dropna=False)["composition_year"]
            .nunique()
        )
        ambiguous = n_years[n_years > 1]
        if not ambiguous.empty:
            # Esta clave no es suficientemente especifica; probar la siguiente.
            continue

        mapping = right.drop_duplicates(subset=keys, keep="first")
        merged = left.merge(
            mapping,
            on=keys,
            how="left",
            validate="many_to_one",
        )

        n_matched = merged["composition_year"].notna().sum()
        if n_matched > 0:
            n_missing = merged["composition_year"].isna().sum()
            print(
                "Ano de composicion agregado usando "
                f"{'+'.join(keys)}: {n_matched} filas asociadas, "
                f"{n_missing} sin ano."
            )
            return merged

    raise KeyError(
        "No se encontro una clave comun suficiente para asociar los anos de "
        "composicion. Se intentaron: work_id, filename, composer+work_title y "
        "composer+catalogue."
    )


def load_data() -> pd.DataFrame:
    """Carga el CSV, calcula Delta RMSE y agrega el ano de composicion."""
    path = locate_csv(INPUT_CSV)
    df = pd.read_csv(path)

    required = {
        "composer",
        "period",
        "voice",
        "rmse_log_exponential_observed",
        "rmse_log_powerlaw_observed",
    }

    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas necesarias: {sorted(missing)}")

    df = df.copy()

    df["delta_rmse"] = (
        pd.to_numeric(df["rmse_log_exponential_observed"], errors="coerce")
        - pd.to_numeric(df["rmse_log_powerlaw_observed"], errors="coerce")
    )

    for col in ["composer", "period", "voice"]:
        df[col] = df[col].fillna("Desconocido").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Desconocido"

    df = df[np.isfinite(df["delta_rmse"])].copy()

    if df.empty:
        raise ValueError("No quedaron valores finitos de Delta RMSE.")

    df = add_composition_year(df)
    return df


# ============================================================
# ORDEN DE CATEGORIAS
# ============================================================


def ordered_unique(series: pd.Series) -> list[str]:
    """Valores unicos preservando el orden de aparicion."""
    return list(dict.fromkeys(series.astype(str).tolist()))


def composer_order(df: pd.DataFrame) -> list[str]:
    """Ordena compositores por la mediana de Delta RMSE, de menor a mayor."""
    medians = (
        df.groupby("composer", sort=False)["delta_rmse"]
        .median()
        .sort_values(ascending=True)
    )
    return medians.index.tolist()


def period_order(df: pd.DataFrame) -> list[str]:
    present = ordered_unique(df["period"])

    if not PERIOD_ORDER:
        return present

    preferred = [p for p in PERIOD_ORDER if p in present]
    extras = [p for p in present if p not in preferred]
    return preferred + extras


def voice_order(df: pd.DataFrame) -> list[str]:
    present = ordered_unique(df["voice"])
    preferred = [v for v in VOICE_ORDER if v in present]
    extras = [v for v in present if v not in preferred]
    return preferred + extras


# ============================================================
# GRAFICA DE PUNTOS
# ============================================================


def plot_delta_rmse_points(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    title: str,
    filename: str,
    rng: np.random.Generator,
) -> None:
    """Grafica un punto por voz-movimiento con jitter vertical."""

    groups = [g for g in groups if g in set(df[group_col])]
    if not groups:
        return

    fig_height = max(4.8, 0.45 * len(groups) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))

    for y_index, group in enumerate(groups):
        values = df.loc[df[group_col] == group, "delta_rmse"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        y = y_index + rng.uniform(-JITTER, JITTER, size=len(values))

        ax.scatter(
            values,
            y,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
        )

    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="0.35")

    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel(r"$\Delta RMSE = RMSE_{exp}-RMSE_{power}$")
    ax.set_title(title)

    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_delta_rmse_vs_year(df: pd.DataFrame) -> None:
    """Grafica Delta RMSE contra el ano de composicion de la obra."""
    data = df.copy()
    data["composition_year"] = pd.to_numeric(
        data["composition_year"], errors="coerce"
    )

    mask = (
        np.isfinite(data["composition_year"])
        & np.isfinite(data["delta_rmse"])
    )
    data = data.loc[mask].copy()

    if data.empty:
        print(
            "Advertencia: no hay anos de composicion finitos; "
            "se omite la figura Delta RMSE vs ano."
        )
        return

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    # Una serie de puntos por voz. Cada punto sigue correspondiendo a
    # una voz de un movimiento; no se promedian movimientos ni obras.
    for voice in voice_order(data):
        sub = data.loc[data["voice"] == voice]
        ax.scatter(
            sub["composition_year"],
            sub["delta_rmse"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
            label=voice,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")

    ax.set_xlabel("Ano de composicion")
    ax.set_ylabel(r"$\Delta RMSE = RMSE_{exp}-RMSE_{power}$")
    ax.set_title(r"$\Delta RMSE$ vs ano de composicion de la pieza")

    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(title="Voz", frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "delta_rmse_vs_ano_composicion.png",
        dpi=PNG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    rng = np.random.default_rng(RNG_SEED)

    plot_delta_rmse_points(
        df=df,
        group_col="composer",
        groups=composer_order(df),
        title="Ventaja relativa del ajuste por compositor",
        filename="delta_rmse_puntos_por_compositor.png",
        rng=rng,
    )

    plot_delta_rmse_points(
        df=df,
        group_col="period",
        groups=period_order(df),
        title="Ventaja relativa del ajuste por período/época",
        filename="delta_rmse_puntos_por_periodo.png",
        rng=rng,
    )

    plot_delta_rmse_points(
        df=df,
        group_col="voice",
        groups=voice_order(df),
        title="Ventaja relativa del ajuste por voz/instrumento",
        filename="delta_rmse_puntos_por_voz.png",
        rng=rng,
    )

    plot_delta_rmse_vs_year(df)

    print("Proceso terminado.")
    print(f"CSV utilizado: {locate_csv(INPUT_CSV).resolve()}")
    print(f"CSV de anos: {locate_csv(CORPUS_CSV).resolve()}")
    print(f"Figuras guardadas en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
