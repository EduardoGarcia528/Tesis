import numpy as np
import matplotlib.pyplot as plt


def logistic_bifurcation(r_values, n_transient=1500, n_keep=120, x0=0.5):
    """
    Genera datos del diagrama de bifurcación del mapa logístico:
        x_{n+1} = r x_n (1 - x_n)
    """
    r_values = np.asarray(r_values, dtype=float)
    x = np.full_like(r_values, x0, dtype=float)

    for _ in range(n_transient):
        x = r_values * x * (1.0 - x)

    xs = []
    for _ in range(n_keep):
        x = r_values * x * (1.0 - x)
        xs.append(x.copy())

    xx = np.array(xs)
    rr = np.tile(r_values, n_keep)
    return rr, xx.ravel()


def extraer_xy_desde_archivo(archivo):
    """
    Carga archivo .npy y construye el eje r compatible con tu caso.
    """

    r_full = np.sort(np.concatenate((
        np.linspace(3.5695, 3.5702, 300),
        np.array([3.569945672])
    )))

    data = np.load(archivo)

    if data.ndim == 2:
        if data.shape[0] == 2:
            y = data[1]
            x_archivo = data[0]
        elif data.shape[1] == 2:
            x_archivo = data[:, 0]
            y = data[:, 1]
        else:
            raise ValueError(
                f"No se pudo interpretar el archivo con shape {data.shape}. "
                "Se esperaba (2, N), (N, 2) o (N,)."
            )
    elif data.ndim == 1:
        y = data
        x_archivo = None
    else:
        raise ValueError(f"Formato no soportado: shape={data.shape}")

    if x_archivo is not None:
        x = np.asarray(x_archivo, dtype=float)

    else:
        if len(y) == len(r_full):
            x = r_full.copy()
        elif len(y) == 300:
            x = np.linspace(3.5695, 3.5702, 300)
            print("Aviso: el archivo tiene 300 puntos; se usa r = linspace(3.5695, 3.5702, 300)")
        else:
            raise ValueError(
                f"La longitud de y es {len(y)}, pero el eje r esperado tiene 301 "
                "o 300 puntos. No coinciden."
            )

    y = np.asarray(y, dtype=float)

    orden = np.argsort(x)
    x = x[orden]
    y = y[orden]

    return x, y


def ajuste_lineal_estable(x, y):
    """
    Ajuste lineal y = m x + b usando centrado numérico para evitar
    problemas de precisión cuando x varía poco alrededor de 3.57.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x0 = np.mean(x)
    X = np.column_stack((x - x0, np.ones_like(x)))

    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    m = coef[0]
    c = coef[1]

    b = c - m * x0

    y_pred = m * x + b
    rss = np.sum((y - y_pred)**2)

    return m, b, rss, y_pred


def ajuste_doble_lineal_automatico(
    x,
    y,
    min_puntos=20,
    margen=0.05,
    exigir_interseccion_en_dominio=True,
    penalizar_interseccion_lejana=True
):
    """
    Ajuste automático de dos rectas.

    Parámetros
    ----------
    x, y : arrays
        Datos de la curva J(r).

    min_puntos : int
        Número mínimo de puntos en cada región.

    margen : float
        Fracción de puntos que se ignora en los extremos para evitar
        cortes degenerados.

    exigir_interseccion_en_dominio : bool
        Si True, exige que la intersección caiga dentro del dominio de x.

    penalizar_interseccion_lejana : bool
        Si True, penaliza soluciones donde la intersección esté muy lejos
        del punto de corte.

    Retorna
    -------
    resultado : dict
        Diccionario con los parámetros óptimos del ajuste.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)

    if n != len(y):
        raise ValueError("x e y deben tener la misma longitud.")

    if n < 2 * min_puntos:
        raise ValueError("No hay suficientes puntos para hacer dos ajustes lineales.")

    i_min = max(min_puntos, int(margen * n))
    i_max = min(n - min_puntos, int((1.0 - margen) * n))

    mejores = []

    for i in range(i_min, i_max):

        x_izq = x[:i]
        y_izq = y[:i]

        x_der = x[i:]
        y_der = y[i:]

        m1, b1, rss1, _ = ajuste_lineal_estable(x_izq, y_izq)
        m2, b2, rss2, _ = ajuste_lineal_estable(x_der, y_der)

        if np.isclose(m1, m2):
            continue

        x_int = (b2 - b1) / (m1 - m2)
        y_int = m1 * x_int + b1

        if exigir_interseccion_en_dominio:
            if not (x.min() <= x_int <= x.max()):
                continue

        rss_total = rss1 + rss2

        # Criterio tipo BIC: RSS + penalización por número de parámetros.
        # Hay 4 parámetros: m1, b1, m2, b2.
        k_param = 4
        bic = n * np.log(rss_total / n) + k_param * np.log(n)

        score = bic

        if penalizar_interseccion_lejana:
            x_corte = x[i]
            escala = x.max() - x.min()
            penalizacion = ((x_int - x_corte) / escala)**2
            score = score + penalizacion

        mejores.append({
            "i_corte": i,
            "x_corte": x[i],
            "x_int": x_int,
            "y_int": y_int,
            "m1": m1,
            "b1": b1,
            "m2": m2,
            "b2": b2,
            "rss1": rss1,
            "rss2": rss2,
            "rss_total": rss_total,
            "bic": bic,
            "score": score,
            "n_izq": len(x_izq),
            "n_der": len(x_der),
        })

    if len(mejores) == 0:
        raise RuntimeError(
            "No se encontró un ajuste válido. Prueba reducir min_puntos, "
            "desactivar exigir_interseccion_en_dominio, o revisar la forma de J(r)."
        )

    mejor = min(mejores, key=lambda d: d["score"])

    return mejor, mejores


def bootstrap_interseccion(
    x,
    y,
    n_boot=500,
    min_puntos=20,
    margen=0.05,
    semilla=123
):
    """
    Bootstrap simple para estimar incertidumbre de la intersección.
    Re-muestrea residuos alrededor del ajuste doble óptimo.
    """

    rng = np.random.default_rng(semilla)

    mejor, _ = ajuste_doble_lineal_automatico(
        x,
        y,
        min_puntos=min_puntos,
        margen=margen
    )

    i = mejor["i_corte"]

    m1, b1 = mejor["m1"], mejor["b1"]
    m2, b2 = mejor["m2"], mejor["b2"]

    y_model = np.empty_like(y)
    y_model[:i] = m1 * x[:i] + b1
    y_model[i:] = m2 * x[i:] + b2

    residuos = y - y_model
    xints = []

    for _ in range(n_boot):
        residuos_boot = rng.choice(residuos, size=len(residuos), replace=True)
        y_boot = y_model + residuos_boot

        try:
            mejor_boot, _ = ajuste_doble_lineal_automatico(
                x,
                y_boot,
                min_puntos=min_puntos,
                margen=margen
            )
            xints.append(mejor_boot["x_int"])
        except Exception:
            pass

    xints = np.array(xints)

    if len(xints) == 0:
        return None

    return {
        "xints": xints,
        "media": np.mean(xints),
        "std": np.std(xints, ddof=1),
        "q025": np.quantile(xints, 0.025),
        "q975": np.quantile(xints, 0.975),
        "n_validos": len(xints)
    }


def ajuste_doble_automatico_con_bifurcacion(
    archivo='J_transicion4.npy',
    min_puntos=20,
    margen=0.05,
    n_transient=4000,
    n_keep=150,
    usar_bootstrap=True,
    n_boot=500
):
    """
    Versión automática del ajuste doble.

    Ya no se introducen intervalos manuales.
    El código busca el mejor punto de separación entre dos regiones lineales.
    """

    x, y = extraer_xy_desde_archivo(archivo)

    # x = np.delete(x, [243, 244, 245, 246, 247, 248, 249])
    # y = np.delete(y, [243, 244, 245, 246, 247, 248, 249])

    mejor, todos = ajuste_doble_lineal_automatico(
        x,
        y,
        min_puntos=min_puntos,
        margen=margen,
        exigir_interseccion_en_dominio=False,
        penalizar_interseccion_lejana=False
    )

    x_int = mejor["x_int"]
    y_int = mejor["y_int"]

    m1, b1 = mejor["m1"], mejor["b1"]
    m2, b2 = mejor["m2"], mejor["b2"]

    if usar_bootstrap:
        boot = bootstrap_interseccion(
            x,
            y,
            n_boot=n_boot,
            min_puntos=min_puntos,
            margen=margen
        )
    else:
        boot = None

    rr, xx = logistic_bifurcation(
        x,
        n_transient=n_transient,
        n_keep=n_keep
    )

    xfit = np.linspace(x.min(), x.max(), 800)
    yfit1 = m1 * xfit + b1
    yfit2 = m2 * xfit + b2

    i = mejor["i_corte"]

    # print(np.where(y < 0.45)[0])

    fig, axs = plt.subplots(
        2, 1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2]}
    )

    ax0, ax1 = axs

    # Panel superior: bifurcación
    ax0.plot(rr, xx, ',', alpha=0.5)
    ax0.axvline(x_int, ls=':', lw=1.8)
    ax0.axvline(mejor["x_corte"], ls='--', lw=1.2, alpha=0.7)
    ax0.set_ylabel(r'$x_n$')
    ax0.set_title('Detección automática del punto crítico')

    # Panel inferior: J y ajustes
    ax1.plot([], [], 'o-', ms=3, lw=1, label=r'$J(r)$')
    ax1.axvline(3.569945672, ls='-.', lw=1.5, label=r'$r_\infty$')

    ax1.plot(
        xfit,
        yfit1,
        '--',
        lw=2,
        label='Ajuste lineal izquierdo'
    )

    ax1.plot(
        xfit,
        yfit2,
        '--',
        lw=2,
        label='Ajuste lineal derecho'
    )

    ax1.plot(
        x_int,
        y_int,
        '*',
        ms=14,
        label=fr'Intersección: $r = {x_int:.9f}$'
    )

    ax1.axvline(x_int, ls=':', lw=1.8)
    # ax1.axvline(mejor["x_corte"], ls='--', lw=1.2, alpha=0.7)

    ax1.scatter(
        x[:i],
        y[:i],
        s=20,
        alpha=0.7,
    )

    ax1.scatter(
        x[i:],
        y[i:],
        s=20,
        alpha=0.7,
    )

    ax1.set_xlabel(r'$r$')
    ax1.set_ylabel(r'$J$', rotation=0)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # Figura de diagnóstico: score contra posición del corte
    x_cortes = np.array([d["x_corte"] for d in todos])
    scores = np.array([d["score"] for d in todos])
    rss = np.array([d["rss_total"] for d in todos])

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x_cortes, scores, 'o-', ms=3, lw=1)
    ax.axvline(mejor["x_corte"], ls='--', lw=1.5, label='Corte óptimo')
    ax.axvline(x_int, ls=':', lw=1.5, label='Intersección')
    ax.axvline(3.569945672, ls='-.', lw=1.5, label=r'$r_\infty$')
    ax.set_xlabel(r'Corte candidato $r_c$')
    ax.set_ylabel('Score')
    ax.set_title('Diagnóstico de selección automática del corte')
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("RESULTADO DEL AJUSTE DOBLE AUTOMÁTICO")
    print("="*80)

    print(f"Número de puntos totales      = {len(x)}")
    print(f"Puntos región izquierda       = {mejor['n_izq']}")
    print(f"Puntos región derecha         = {mejor['n_der']}")

    print("\n--- Corte elegido por minimización ---")
    print(f"r_corte = {mejor['x_corte']:.12f}")

    print("\n--- Recta izquierda ---")
    print(f"J(r) = {m1:.12e} r + {b1:.12e}")

    print("\n--- Recta derecha ---")
    print(f"J(r) = {m2:.12e} r + {b2:.12e}")

    print("\n--- Intersección ---")
    print(f"r_interseccion = {x_int:.12f}")
    print(f"J_interseccion = {y_int:.12e}")

    print("\n--- Calidad del ajuste ---")
    print(f"RSS izquierda = {mejor['rss1']:.12e}")
    print(f"RSS derecha   = {mejor['rss2']:.12e}")
    print(f"RSS total     = {mejor['rss_total']:.12e}")
    print(f"BIC           = {mejor['bic']:.12e}")
    print(f"Score         = {mejor['score']:.12e}")

    if boot is not None:
        print("\n--- Bootstrap de la intersección ---")
        print(f"Bootstrap válidos = {boot['n_validos']} / {n_boot}")
        print(f"Media             = {boot['media']:.12f}")
        print(f"Desv. estándar    = {boot['std']:.12e}")
        print(f"IC 95%            = [{boot['q025']:.12f}, {boot['q975']:.12f}]")

    print("="*80)

    return mejor, todos, boot


mejor, todos, boot = ajuste_doble_automatico_con_bifurcacion(
    archivo='J_transicion.npy',
    min_puntos=50, 
    margen=0.1,
    n_transient=600,
    n_keep=120,
    usar_bootstrap=True,
    n_boot=500
)