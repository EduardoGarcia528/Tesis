import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import mi_libreria as ml


RC_TEORICO = 3.569945672


# ============================================================
# 1. MAPEO LOGISTICO Y GENERACION DE SERIES
# ============================================================

@njit
def generar_serie_determinista(r, longitud_serie, iter_descartar, x0=0.6):
    """
    Genera la órbita determinista del mapeo logístico.
    Se conserva la misma regla de clipping del código original:
    clip antes de aplicar el mapa.
    """
    x = x0

    for _ in range(iter_descartar):
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        x = r * x * (1.0 - x)

    serie = np.empty(longitud_serie, dtype=np.float64)

    for i in range(longitud_serie):
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        x = r * x * (1.0 - x)
        serie[i] = x

    return serie


@njit
def generar_serie_iterativa(
    r,
    longitud_serie,
    iter_descartar,
    ruido_transitorio,
    ruido_serie,
    x0=0.6
):
    """
    Ruido dinámico/iterativo:
        x_{n+1} = f_r(x_n) + eta_n

    Se conserva la convención del código original:
    el clipping ocurre antes de aplicar el mapa.
    """
    x = x0

    for i in range(iter_descartar):
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0

        x = r * x * (1.0 - x)
        x += ruido_transitorio[i]

    serie = np.empty(longitud_serie, dtype=np.float64)

    for i in range(longitud_serie):
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0

        x = r * x * (1.0 - x)
        x += ruido_serie[i]
        serie[i] = x

    return serie


def ruido_uniforme_rng(rng, size, var_ruido):
    """
    Conserva EXACTAMENTE la parametrización histórica del código original:

        eta ~ U(-sqrt(var_ruido), +sqrt(var_ruido))

    Por tanto, la varianza matemática efectiva es:
        Var(eta) = var_ruido / 3

    No se corrige aquí para no cambiar simultáneamente la intensidad
    de ruido de los experimentos previos.
    """
    a = np.sqrt(var_ruido)
    return rng.uniform(-a, a, size=size)


def rng_para_celda(seed, realizacion, indice_r):
    """
    RNG reproducible por celda J_k(r_q).
    Esto permite reanudar un cálculo sin cambiar las realizaciones
    que faltan por calcular.
    """
    ss = np.random.SeedSequence([int(seed), int(realizacion), int(indice_r)])
    return np.random.default_rng(ss)


# ============================================================
# 2. GENERACION Y GUARDADO DE TODAS LAS REALIZACIONES J_k(r)
# ============================================================

def guardar_checkpoint(
    archivo,
    r_vals,
    J_all,
    x_bif,
    parametros
):
    np.savez(
        archivo,
        r_vals=r_vals,
        J_all=J_all,
        x_bif=x_bif,
        **parametros
    )


def generar_J_realizaciones(
    r_min=3.5696,
    r_max=3.5701,
    resolucion_r=300,
    longitud_serie=20_000,
    iter_descartar=1000,
    var_ruido=1e-7,
    tipo_ruido="aditivo",
    n_realizaciones=500,
    seed=12345,
    archivo_salida="J_transicion_realizaciones.npz",
    reanudar=True,
    checkpoint_cada_r=1,
    n_bifurcacion=1000,
    verbose=True
):
    """
    Calcula y guarda TODAS las realizaciones:

        J_all[k, q] = J_k(r_q)

    Importante
    ----------
    - r_c NO se inserta en la malla de r.
    - Para ruido aditivo, la órbita determinista se genera una sola vez
      por cada r y después se añaden realizaciones independientes de ruido.
    - Se guarda un checkpoint para poder reanudar.
    - x_bif contiene sólo los últimos n_bifurcacion puntos de la primera
      realización, para no guardar millones de puntos innecesariamente.
    """

    if tipo_ruido not in ("no", "aditivo", "iterativo"):
        raise ValueError("tipo_ruido debe ser 'no', 'aditivo' o 'iterativo'.")

    r_vals = np.linspace(r_min, r_max, resolucion_r, dtype=float)

    if r_vals[0] == 3.0:
        r_vals = r_vals[1:]

    nr = len(r_vals)
    n_bif = min(int(n_bifurcacion), int(longitud_serie))

    parametros = {
        "r_min": np.array(r_min),
        "r_max": np.array(r_max),
        "resolucion_r": np.array(resolucion_r),
        "longitud_serie": np.array(longitud_serie),
        "iter_descartar": np.array(iter_descartar),
        "var_ruido": np.array(var_ruido),
        "tipo_ruido": np.array(tipo_ruido),
        "n_realizaciones": np.array(n_realizaciones),
        "seed": np.array(seed),
        "n_bifurcacion": np.array(n_bif)
    }

    J_all = np.full((n_realizaciones, nr), np.nan, dtype=float)
    x_bif = np.full((nr, n_bif), np.nan, dtype=float)

    # --------------------------------------------------------
    # Reanudar si existe un checkpoint compatible
    # --------------------------------------------------------
    if reanudar and os.path.exists(archivo_salida):
        old = np.load(archivo_salida, allow_pickle=False)

        r_old = old["r_vals"]
        J_old = old["J_all"]

        if J_old.shape != J_all.shape:
            raise ValueError(
                "El checkpoint existe, pero J_all tiene una forma distinta.\n"
                f"Checkpoint: {J_old.shape}; configuración actual: {J_all.shape}"
            )

        if not np.allclose(r_old, r_vals, rtol=0.0, atol=1e-15):
            raise ValueError(
                "El checkpoint existe, pero la malla de r no coincide "
                "con la configuración actual."
            )

        J_all[:] = J_old

        if "x_bif" in old.files and old["x_bif"].shape == x_bif.shape:
            x_bif[:] = old["x_bif"]

        if verbose:
            hechos = np.sum(np.isfinite(J_all))
            total = J_all.size
            print(f"Checkpoint cargado: {hechos}/{total} valores de J ya calculados.")

    # --------------------------------------------------------
    # Cálculo
    # --------------------------------------------------------
    for q, r in enumerate(r_vals):

        # Si toda la columna ya existe, no recalcular.
        if np.all(np.isfinite(J_all[:, q])):
            if verbose:
                print(f"r {q+1:4d}/{nr}: {r:.10f}  [ya calculado]")
            continue

        if verbose:
            hechos_col = np.sum(np.isfinite(J_all[:, q]))
            print(
                f"r {q+1:4d}/{nr}: {r:.10f}  "
                f"({hechos_col}/{n_realizaciones} realizaciones previas)"
            )

        # Para ruido aditivo/no, la dinámica subyacente es la misma
        # para todas las realizaciones.
        if tipo_ruido in ("aditivo", "no"):
            serie_base = generar_serie_determinista(
                r=r,
                longitud_serie=longitud_serie,
                iter_descartar=iter_descartar,
                x0=0.6
            )

        for k in range(n_realizaciones):

            if np.isfinite(J_all[k, q]):
                continue

            rng = rng_para_celda(seed, k, q)

            if tipo_ruido == "aditivo":
                # Ruido observacional:
                # y_n = x_n + eta_n
                serie = serie_base + ruido_uniforme_rng(
                    rng, longitud_serie, var_ruido
                )

            elif tipo_ruido == "iterativo":
                ruido_trans = ruido_uniforme_rng(
                    rng, iter_descartar, var_ruido
                )
                ruido_ser = ruido_uniforme_rng(
                    rng, longitud_serie, var_ruido
                )

                serie = generar_serie_iterativa(
                    r=r,
                    longitud_serie=longitud_serie,
                    iter_descartar=iter_descartar,
                    ruido_transitorio=ruido_trans,
                    ruido_serie=ruido_ser,
                    x0=0.6
                )

            else:
                serie = serie_base.copy()

            J_all[k, q] = ml.indice_J(serie)

            # Una sola realización representativa para bifurcación.
            if k == 0 and np.any(~np.isfinite(x_bif[q])):
                x_bif[q, :] = serie[-n_bif:]

        if (q + 1) % checkpoint_cada_r == 0:
            guardar_checkpoint(
                archivo=archivo_salida,
                r_vals=r_vals,
                J_all=J_all,
                x_bif=x_bif,
                parametros=parametros
            )

    # Guardado final
    guardar_checkpoint(
        archivo=archivo_salida,
        r_vals=r_vals,
        J_all=J_all,
        x_bif=x_bif,
        parametros=parametros
    )

    if verbose:
        print(f"\nDatos completos guardados en: {archivo_salida}")
        print("Forma de J_all:", J_all.shape)

    return r_vals, J_all, x_bif

# ============================================================
# 3. GRAFICA J(r) CERCA DEL PUNTO CRITICO
# ============================================================

def graficar_J_cerca_rc(
    archivo_datos,
    rc_teorico=RC_TEORICO,
    guardar_figura=True,
    mostrar_figura=True,
    archivo_figura="J_cerca_rc.png"
):
    """
    Carga las realizaciones J_k(r) y grafica únicamente la curva media
    J(r) en la ventana local alrededor del punto crítico.

    Se dibuja una línea vertical en el valor teórico r_c.
    No se realiza ningún ajuste ni prueba estadística.
    """
    data = np.load(archivo_datos, allow_pickle=False)

    r_vals = np.asarray(data["r_vals"], dtype=float)
    J_all = np.asarray(data["J_all"], dtype=float)

    if np.any(~np.isfinite(J_all)):
        faltantes = np.sum(~np.isfinite(J_all))
        raise ValueError(
            f"J_all todavía contiene {faltantes} valores no calculados. "
            "Completa primero la generación."
        )

    # Promedio sobre realizaciones para cada valor de r.
    J_mean = np.mean(J_all, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        r_vals,
        J_mean,
        marker=".",
        markersize=3,
        linewidth=0.9,
        label=r"$\overline{J}(r)$"
    )

    ax.axvline(
        rc_teorico,
        linestyle="--",
        linewidth=1.3,
        label=rf"$r_c={rc_teorico:.9f}$"
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$J$")
    ax.set_xlim(r_vals.min(), r_vals.max())
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if guardar_figura:
        fig.savefig(archivo_figura, dpi=300, bbox_inches="tight")

    if mostrar_figura:
        plt.show()
    else:
        plt.close(fig)

    return r_vals, J_mean


# ============================================================
# 4. EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    archivo = "J_transicion_realizaciones.npz"

    # --------------------------------------------------------
    # A) GENERACION DE LAS REALIZACIONES
    # --------------------------------------------------------
    r_vals, J_all, x_bif = generar_J_realizaciones(
        r_min=3.5699,
        r_max=3.5700,
        resolucion_r=300,
        longitud_serie=20_000,
        iter_descartar=1000,
        var_ruido=1e-7,
        tipo_ruido="aditivo",
        n_realizaciones=500,
        seed=12345,
        archivo_salida=archivo,
        reanudar=True,
        checkpoint_cada_r=1,
        n_bifurcacion=1000,
        verbose=True
    )

    # --------------------------------------------------------
    # B) UNICA SALIDA: J(r) CERCA DE r_c
    # --------------------------------------------------------
    graficar_J_cerca_rc(
        archivo_datos=archivo,
        rc_teorico=RC_TEORICO,
        guardar_figura=True,
        mostrar_figura=True,
        archivo_figura="J_cerca_rc.png"
    )
