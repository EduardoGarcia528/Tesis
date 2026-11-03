"""
Funciones para construir trayectorias de fase invariantes (o robustas) frente
al término lineal producido por una traslación temporal de Fourier.

Se implementan tres formulaciones:

1) TORO DE BIFASES (T^2)
   P_k = (psi_{k,r1}, psi_{k,r2}),
   psi_{k,r} = arg[X_k X_r X^*_{k+r}].

   Los vectores se calculan con la diferencia angular mínima en cada
   coordenada del toro y los ángulos alpha se obtienen entre tangentes
   consecutivas.

2) EMBEDDING EUCLÍDEO DE BIFASES (R^4)
   Gamma_k = (cos psi_{k,r1}, sin psi_{k,r1},
              cos psi_{k,r2}, sin psi_{k,r2}).

   Evita por construcción los problemas de wrap/unwrapping. La trayectoria
   vive sobre S^1 x S^1 embebido en R^4.

3) EMBEDDING R^4 PONDERADO POR BICOHERENCIA
   Z_{k,r} = b(k,r) exp(i beta_{k,r}),
   Gamma_k = (Re Z_{k,r1}, Im Z_{k,r1},
              Re Z_{k,r2}, Im Z_{k,r2}).

   beta es la bifase del bispectrum estimado por segmentos y b es la
   bicoherencia normalizada. Esta tercera formulación requiere varias
   realizaciones o, como aproximación para una sola serie, segmentación.

Notas conceptuales
------------------
Para una traslación circular x[n] -> x[(n-tau) mod N], cada coeficiente de
Fourier adquiere una fase lineal exp(-i omega_k tau). En el producto de cierre

    X_k X_r X^*_{k+r}

la fase de traslación se cancela porque omega_{k+r} = omega_k + omega_r
(mod 2*pi). Por eso las bifases son invariantes frente a una traslación
circular ideal de la serie completa.

En las formulaciones 1 y 2 se usa una sola FFT de la serie completa. En la
formulación 3, al estimar bicoherencia segmentando una sola realización,
la segmentación y la ventana introducen efectos de borde: una traslación
circular GLOBAL de la serie no tiene por qué dejar exactamente igual la
estimación segmentada, aunque la combinación trifásica sí sea invariante en
cada realización ideal.

El archivo contiene sólo funciones; no produce gráficas ni imprime resultados.
"""

import numpy as np


# ============================================================================
# Utilidades básicas
# ============================================================================

def wrap_pi(a):
    """Envuelve ángulos al intervalo [-pi, pi)."""
    a = np.asarray(a)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _validar_serie(x, min_len=8):
    """Convierte x a float64 y valida que sea una serie 1D finita."""
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x debe ser una serie unidimensional.")
    if len(x) < min_len:
        raise ValueError(f"x debe contener al menos {min_len} muestras.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contiene NaN o valores infinitos.")

    return x


def _max_bin_positivo_rfft(n):
    """
    Mayor bin positivo usado para fases.

    Se excluye DC (k=0). Si n es par también se excluye Nyquist, porque su
    coeficiente de rFFT es real y su fase queda restringida a 0/pi.
    """
    n_rfft = n // 2 + 1

    if n % 2 == 0:
        return n_rfft - 2
    return n_rfft - 1


def _validar_offsets(r1, r2, max_bin):
    """Valida los offsets enteros r1 y r2 usados en las tríadas (k,r,k+r)."""
    for nombre, r in (("r1", r1), ("r2", r2)):
        if not isinstance(r, (int, np.integer)):
            raise TypeError(f"{nombre} debe ser un entero.")
        if r <= 0:
            raise ValueError(f"{nombre} debe ser > 0; k=0 (DC) no se usa.")
        if r >= max_bin:
            raise ValueError(
                f"{nombre}={r} es demasiado grande para el espectro disponible "
                f"(máximo bin positivo utilizable: {max_bin})."
            )

    if r1 == r2:
        raise ValueError("r1 y r2 deben ser distintos para evitar coordenadas duplicadas.")


def _k_comunes(r1, r2, max_bin):
    """Bins k para los que k+r1 y k+r2 permanecen en el rango utilizable."""
    k_max = min(max_bin - r1, max_bin - r2)

    if k_max < 1:
        return np.empty(0, dtype=np.int64)

    return np.arange(1, k_max + 1, dtype=np.int64)


def _umbral_amplitud(X, max_bin, amplitude_floor):
    """Convierte un umbral relativo de amplitud en un umbral absoluto."""
    if amplitude_floor is None:
        return 0.0

    amplitude_floor = float(amplitude_floor)
    if amplitude_floor < 0.0:
        raise ValueError("amplitude_floor debe ser >= 0 o None.")

    amps = np.abs(X[1:max_bin + 1])
    if amps.size == 0:
        return 0.0

    escala = np.max(amps)
    return amplitude_floor * escala


def _mascara_triada_amplitud(X, k, r, threshold):
    """Máscara para tríadas cuyas tres amplitudes exceden threshold."""
    return (
        (np.abs(X[k]) > threshold)
        & (np.abs(X[r]) > threshold)
        & (np.abs(X[k + r]) > threshold)
    )


def _producto_cierre(X, k, r):
    """Producto de cierre C_{k,r} = X_k X_r X^*_{k+r}."""
    return X[k] * X[r] * np.conj(X[k + r])


def bifase_desde_fft(X, k, r):
    """
    Calcula psi_{k,r} = arg[X_k X_r X^*_{k+r}] para uno o varios k.

    Usar el producto complejo directamente evita hacer unwrap de las fases
    individuales.
    """
    k = np.asarray(k, dtype=np.int64)
    return np.angle(_producto_cierre(X, k, r))


# ============================================================================
# Geometría discreta: sólo conectar bins verdaderamente consecutivos
# ============================================================================

def vectores_toro_consecutivos(puntos, k_indices):
    """
    Tangentes geodésicas de una trayectoria en T^d.

    Sólo conecta P_k con P_{k+1} si los índices de frecuencia son realmente
    consecutivos. Esto evita saltar artificialmente huecos producidos por un
    filtro de baja amplitud.

    Retorna
    -------
    vectores : ndarray, shape (M,d)
    k_vectores : ndarray, shape (M,)
        k_vectores[j] es el k inicial del segmento P_k -> P_{k+1}.
    """
    puntos = np.asarray(puntos, dtype=np.float64)
    k_indices = np.asarray(k_indices, dtype=np.int64)

    if puntos.ndim != 2:
        raise ValueError("puntos debe tener shape (n_puntos, dimension).")
    if len(puntos) != len(k_indices):
        raise ValueError("puntos y k_indices deben tener la misma longitud.")
    if len(puntos) < 2:
        return np.empty((0, puntos.shape[1])), np.empty(0, dtype=np.int64)

    consecutivos = np.diff(k_indices) == 1
    diferencias = wrap_pi(puntos[1:] - puntos[:-1])

    return diferencias[consecutivos], k_indices[:-1][consecutivos]


def vectores_euclideos_consecutivos(puntos, k_indices):
    """
    Tangentes euclídeas P_{k+1}-P_k, sin conectar huecos en k.

    Retorna vectores y el bin inicial de cada vector.
    """
    puntos = np.asarray(puntos, dtype=np.float64)
    k_indices = np.asarray(k_indices, dtype=np.int64)

    if puntos.ndim != 2:
        raise ValueError("puntos debe tener shape (n_puntos, dimension).")
    if len(puntos) != len(k_indices):
        raise ValueError("puntos y k_indices deben tener la misma longitud.")
    if len(puntos) < 2:
        return np.empty((0, puntos.shape[1])), np.empty(0, dtype=np.int64)

    consecutivos = np.diff(k_indices) == 1
    diferencias = puntos[1:] - puntos[:-1]

    return diferencias[consecutivos], k_indices[:-1][consecutivos]


def angulos_2d_consecutivos(vectores, k_vectores, orientados=False, eps=1e-15):
    """
    Ángulos entre tangentes 2D consecutivas.

    Si orientados=False (recomendado para curvatura), alpha pertenece a [0,pi].
    Si orientados=True, usa atan2(cross,dot) y devuelve alpha en [0,2*pi).

    Sólo compara dos vectores si corresponden a los segmentos
        P_k -> P_{k+1}
        P_{k+1} -> P_{k+2}.
    """
    V = np.asarray(vectores, dtype=np.float64)
    kv = np.asarray(k_vectores, dtype=np.int64)

    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("vectores debe tener shape (n,2).")
    if len(V) != len(kv):
        raise ValueError("vectores y k_vectores deben tener la misma longitud.")
    if len(V) < 2:
        return np.empty(0), np.empty(0, dtype=np.int64)

    adyacentes = np.diff(kv) == 1
    v1 = V[:-1]
    v2 = V[1:]

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    validos = adyacentes & (n1 > eps) & (n2 > eps)

    if not np.any(validos):
        return np.empty(0), np.empty(0, dtype=np.int64)

    v1 = v1[validos]
    v2 = v2[validos]
    k_alpha = kv[:-1][validos] + 1

    dot = np.sum(v1 * v2, axis=1)

    if orientados:
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        alpha = np.arctan2(cross, dot)
        alpha = np.mod(alpha, 2.0 * np.pi)
    else:
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        cosang = np.clip(dot / denom, -1.0, 1.0)
        alpha = np.arccos(cosang)

    return alpha, k_alpha


def angulos_nd_consecutivos(vectores, k_vectores, eps=1e-15):
    """
    Ángulo principal entre tangentes consecutivas en R^d.

    En dimensión >2 no existe un signo de giro único sin introducir estructura
    adicional; por ello alpha se devuelve naturalmente en [0,pi].
    """
    V = np.asarray(vectores, dtype=np.float64)
    kv = np.asarray(k_vectores, dtype=np.int64)

    if V.ndim != 2:
        raise ValueError("vectores debe tener shape (n,dimension).")
    if len(V) != len(kv):
        raise ValueError("vectores y k_vectores deben tener la misma longitud.")
    if len(V) < 2:
        return np.empty(0), np.empty(0, dtype=np.int64)

    adyacentes = np.diff(kv) == 1
    v1 = V[:-1]
    v2 = V[1:]

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    validos = adyacentes & (n1 > eps) & (n2 > eps)

    if not np.any(validos):
        return np.empty(0), np.empty(0, dtype=np.int64)

    v1 = v1[validos]
    v2 = v2[validos]
    n1 = n1[validos]
    n2 = n2[validos]

    dot = np.sum(v1 * v2, axis=1)
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    alpha = np.arccos(cosang)
    k_alpha = kv[:-1][validos] + 1

    return alpha, k_alpha


# ============================================================================
# PROPUESTA 1: trayectoria en el toro de dos bifases
# ============================================================================

def trayectoria_toro_bifases(x, r1=1, r2=2, amplitude_floor=1e-12,
                              restar_media=True):
    """
    Construye
        P_k = (psi_{k,r1}, psi_{k,r2}) in T^2,
    con
        psi_{k,r} = arg[X_k X_r X^*_{k+r}].

    Parameters
    ----------
    x : array-like
        Serie temporal real.
    r1, r2 : int
        Offsets/bin de frecuencia fijos que definen las dos familias de
        tríadas (k,r1,k+r1) y (k,r2,k+r2).
    amplitude_floor : float or None
        Umbral relativo respecto a la máxima amplitud espectral positiva.
        Una tríada se descarta si cualquiera de sus coeficientes cae debajo
        del umbral. Usar 0 o None para no filtrar por amplitud.
    restar_media : bool
        Si True, resta la media antes de la FFT.

    Returns
    -------
    puntos : ndarray (M,2)
    k_validos : ndarray (M,)
    info : dict
    """
    x = _validar_serie(x)

    if restar_media:
        x = x - np.mean(x)

    X = np.fft.rfft(x)
    max_bin = _max_bin_positivo_rfft(len(x))
    _validar_offsets(r1, r2, max_bin)

    k = _k_comunes(r1, r2, max_bin)
    threshold = _umbral_amplitud(X, max_bin, amplitude_floor)

    mask1 = _mascara_triada_amplitud(X, k, r1, threshold)
    mask2 = _mascara_triada_amplitud(X, k, r2, threshold)
    mask = mask1 & mask2

    k_validos = k[mask]
    psi1 = bifase_desde_fft(X, k_validos, r1)
    psi2 = bifase_desde_fft(X, k_validos, r2)

    puntos = np.column_stack((psi1, psi2))

    info = {
        "r1": int(r1),
        "r2": int(r2),
        "N": len(x),
        "max_bin": int(max_bin),
        "amplitude_threshold": float(threshold),
        "n_k_candidatos": len(k),
        "n_k_validos": len(k_validos),
    }

    return puntos, k_validos, info


def angulos_alpha_toro_bifases(x, r1=1, r2=2, amplitude_floor=1e-12,
                                restar_media=True, orientados=True):
    """
    Calcula alpha para la propuesta 1.

    Retorna
    -------
    alpha : ndarray
        [0,pi] si orientados=False; [0,2*pi) si orientados=True.
    k_alpha : ndarray
        Bin central k asociado a cada giro.
    detalles : dict
        Incluye puntos, k, vectores e información de construcción.
    """
    puntos, k, info = trayectoria_toro_bifases(
        x, r1=r1, r2=r2,
        amplitude_floor=amplitude_floor,
        restar_media=restar_media,
    )

    vectores, k_vectores = vectores_toro_consecutivos(puntos, k)
    alpha, k_alpha = angulos_2d_consecutivos(
        vectores, k_vectores, orientados=orientados
    )

    detalles = {
        **info,
        "puntos": puntos,
        "k_puntos": k,
        "vectores": vectores,
        "k_vectores": k_vectores,
    }

    return alpha, k_alpha, detalles


# ============================================================================
# PROPUESTA 2: embedding de las bifases en R^4
# ============================================================================

def trayectoria_embedding_bifases(x, r1=1, r2=2, amplitude_floor=1e-12,
                                   restar_media=True):
    """
    Construye la trayectoria

        Gamma_k = (cos psi1, sin psi1, cos psi2, sin psi2) in R^4.

    Esta representación evita discontinuidades artificiales en +/-pi y es una
    inmersión natural de T^2 = S^1 x S^1 en R^4.
    """
    puntos_toro, k, info = trayectoria_toro_bifases(
        x, r1=r1, r2=r2,
        amplitude_floor=amplitude_floor,
        restar_media=restar_media,
    )

    if len(puntos_toro) == 0:
        gamma = np.empty((0, 4), dtype=np.float64)
    else:
        psi1 = puntos_toro[:, 0]
        psi2 = puntos_toro[:, 1]
        gamma = np.column_stack((
            np.cos(psi1), np.sin(psi1),
            np.cos(psi2), np.sin(psi2),
        ))

    info = {
        **info,
        "puntos_toro": puntos_toro,
    }

    return gamma, k, info


def angulos_alpha_embedding_bifases(x, r1=1, r2=2, amplitude_floor=1e-12,
                                     restar_media=True):
    """Calcula alpha in [0,pi] para la propuesta 2 en R^4."""
    gamma, k, info = trayectoria_embedding_bifases(
        x, r1=r1, r2=r2,
        amplitude_floor=amplitude_floor,
        restar_media=restar_media,
    )

    vectores, k_vectores = vectores_euclideos_consecutivos(gamma, k)
    alpha, k_alpha = angulos_nd_consecutivos(vectores, k_vectores)

    detalles = {
        **info,
        "puntos": gamma,
        "k_puntos": k,
        "vectores": vectores,
        "k_vectores": k_vectores,
    }

    return alpha, k_alpha, detalles


# ============================================================================
# PROPUESTA 3: embedding R^4 ponderado por bicoherencia
# ============================================================================

def segmentar_serie(x, nperseg=None, noverlap=None, window="hann",
                    restar_media_segmento=True):
    """
    Divide una serie en segmentos solapados y aplica una ventana.

    Parameters
    ----------
    x : array-like
    nperseg : int or None
        Si None, elige automáticamente min(256, len(x)//2), con mínimo 8.
    noverlap : int or None
        Si None, usa 50 % de solapamiento.
    window : {'hann', 'rect'} or array-like
    restar_media_segmento : bool

    Returns
    -------
    segmentos : ndarray, shape (n_segmentos,nperseg)
    inicios : ndarray
    ventana : ndarray
    """
    x = _validar_serie(x)

    if nperseg is None:
        nperseg = min(256, max(8, len(x) // 2))

    nperseg = int(nperseg)
    if nperseg < 8:
        raise ValueError("nperseg debe ser >= 8.")
    if nperseg > len(x):
        raise ValueError("nperseg no puede exceder la longitud de x.")

    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)

    if not (0 <= noverlap < nperseg):
        raise ValueError("noverlap debe satisfacer 0 <= noverlap < nperseg.")

    paso = nperseg - noverlap
    inicios = np.arange(0, len(x) - nperseg + 1, paso, dtype=np.int64)

    if len(inicios) < 2:
        raise ValueError(
            "Se requieren al menos 2 segmentos para estimar bicoherencia. "
            "Reduce nperseg o noverlap, o usa una serie más larga."
        )

    if isinstance(window, str):
        wname = window.lower()
        if wname in ("hann", "hanning"):
            ventana = np.hanning(nperseg)
        elif wname in ("rect", "boxcar", "rectangular"):
            ventana = np.ones(nperseg, dtype=np.float64)
        else:
            raise ValueError("window debe ser 'hann', 'rect' o un array de longitud nperseg.")
    else:
        ventana = np.asarray(window, dtype=np.float64)
        if ventana.ndim != 1 or len(ventana) != nperseg:
            raise ValueError("La ventana proporcionada debe tener longitud nperseg.")

    segmentos = np.empty((len(inicios), nperseg), dtype=np.float64)

    for j, inicio in enumerate(inicios):
        seg = x[inicio:inicio + nperseg].copy()
        if restar_media_segmento:
            seg -= np.mean(seg)
        segmentos[j] = seg * ventana

    return segmentos, inicios, ventana


def espectros_segmentos(x, nperseg=None, noverlap=None, window="hann",
                        restar_media_segmento=True):
    """Calcula rFFT para todos los segmentos de una serie."""
    segmentos, inicios, ventana = segmentar_serie(
        x,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        restar_media_segmento=restar_media_segmento,
    )

    Xseg = np.fft.rfft(segmentos, axis=1)
    return Xseg, inicios, ventana


def bicoherencia_fila(Xseg, k, r, eps=1e-30):
    """
    Estima bispectrum complejo, bifase y bicoherencia para (k,r,k+r).

    Se usa la normalización

        b = | <X_k X_r X^*_{k+r}> |
            / sqrt( <|X_k X_r|^2> <|X_{k+r}|^2> )

    que satisface 0 <= b <= 1 salvo pequeños errores numéricos.

    Parameters
    ----------
    Xseg : complex ndarray, shape (n_segmentos,n_freq)
    k : int or array-like
    r : int

    Returns
    -------
    B : complex ndarray
        Bispectrum promedio.
    beta : ndarray
        Bifase = angle(B).
    b : ndarray
        Bicoherencia no-cuadrada en [0,1].
    """
    Xseg = np.asarray(Xseg)
    if Xseg.ndim != 2:
        raise ValueError("Xseg debe tener shape (n_segmentos,n_freq).")

    k = np.atleast_1d(np.asarray(k, dtype=np.int64))
    r = int(r)

    if np.any(k < 0) or r < 0 or np.any(k + r >= Xseg.shape[1]):
        raise ValueError("Alguna tríada (k,r,k+r) cae fuera de Xseg.")

    Xk = Xseg[:, k]
    Xr = Xseg[:, r][:, None]
    Xsum = Xseg[:, k + r]

    # Cada columna corresponde a un k diferente.
    prod = Xk * Xr * np.conj(Xsum)
    B = np.mean(prod, axis=0)

    den1 = np.mean(np.abs(Xk * Xr) ** 2, axis=0)
    den2 = np.mean(np.abs(Xsum) ** 2, axis=0)
    denom = np.sqrt(den1 * den2)

    b = np.zeros_like(denom, dtype=np.float64)
    validos = denom > eps
    b[validos] = np.abs(B[validos]) / denom[validos]
    b = np.clip(b, 0.0, 1.0)

    beta = np.angle(B)
    return B, beta, b


def trayectoria_bicoherencia(x, r1=1, r2=2, nperseg=None, noverlap=None,
                             window="hann", min_bicoherence=0.0,
                             restar_media_segmento=True, eps=1e-30):
    """
    Construye la trayectoria ponderada por bicoherencia

        Z_{k,r} = b(k,r) exp(i beta_{k,r})

        Gamma_k = (Re Z_{k,r1}, Im Z_{k,r1},
                   Re Z_{k,r2}, Im Z_{k,r2}) in R^4.

    IMPORTANTE: aquí r1, r2 y k son bins de la FFT de cada segmento, por lo
    que su frecuencia física depende de nperseg (y de fs si después se desea
    convertir bins a frecuencia).
    """
    x = _validar_serie(x)

    Xseg, inicios, ventana = espectros_segmentos(
        x,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        restar_media_segmento=restar_media_segmento,
    )

    nperseg_real = Xseg.shape[1] * 2 - 2 if (len(ventana) % 2 == 0) else len(ventana)
    # La longitud real de la FFT es exactamente la longitud de la ventana.
    nperseg_real = len(ventana)
    max_bin = _max_bin_positivo_rfft(nperseg_real)
    _validar_offsets(r1, r2, max_bin)
    k = _k_comunes(r1, r2, max_bin)

    B1, beta1, b1 = bicoherencia_fila(Xseg, k, r1, eps=eps)
    B2, beta2, b2 = bicoherencia_fila(Xseg, k, r2, eps=eps)

    min_bicoherence = float(min_bicoherence)
    if not (0.0 <= min_bicoherence <= 1.0):
        raise ValueError("min_bicoherence debe estar entre 0 y 1.")

    # Requerimos que ambas coordenadas superen el umbral para que P_k esté
    # completamente definido en el embedding de cuatro dimensiones.
    mask = (
        np.isfinite(beta1) & np.isfinite(beta2)
        & np.isfinite(b1) & np.isfinite(b2)
        & (b1 >= min_bicoherence)
        & (b2 >= min_bicoherence)
    )

    k_validos = k[mask]
    beta1 = beta1[mask]
    beta2 = beta2[mask]
    b1 = b1[mask]
    b2 = b2[mask]
    B1 = B1[mask]
    B2 = B2[mask]

    Z1 = b1 * np.exp(1j * beta1)
    Z2 = b2 * np.exp(1j * beta2)

    gamma = np.column_stack((
        np.real(Z1), np.imag(Z1),
        np.real(Z2), np.imag(Z2),
    ))

    info = {
        "r1": int(r1),
        "r2": int(r2),
        "N": len(x),
        "nperseg": int(nperseg_real),
        "noverlap": int(nperseg_real // 2 if noverlap is None else noverlap),
        "n_segmentos": int(Xseg.shape[0]),
        "max_bin": int(max_bin),
        "min_bicoherence": min_bicoherence,
        "beta1": beta1,
        "beta2": beta2,
        "bicoherence1": b1,
        "bicoherence2": b2,
        "bispectrum1": B1,
        "bispectrum2": B2,
        "segment_starts": inicios,
        "window": ventana,
    }

    return gamma, k_validos, info


def angulos_alpha_bicoherencia(x, r1=1, r2=2, nperseg=None, noverlap=None,
                               window="hann", min_bicoherence=0.0,
                               restar_media_segmento=True, eps=1e-30):
    """Calcula alpha in [0,pi] para la propuesta 3 ponderada por bicoherencia."""
    gamma, k, info = trayectoria_bicoherencia(
        x,
        r1=r1,
        r2=r2,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        min_bicoherence=min_bicoherence,
        restar_media_segmento=restar_media_segmento,
        eps=eps,
    )

    vectores, k_vectores = vectores_euclideos_consecutivos(gamma, k)
    alpha, k_alpha = angulos_nd_consecutivos(vectores, k_vectores)

    detalles = {
        **info,
        "puntos": gamma,
        "k_puntos": k,
        "vectores": vectores,
        "k_vectores": k_vectores,
    }

    return alpha, k_alpha, detalles


# ============================================================================
# Wrapper para comparar las tres formulaciones
# ============================================================================

def tres_formulaciones_alpha(x, r1=1, r2=2, amplitude_floor=1e-12,
                             nperseg=None, noverlap=None, window="hann",
                             min_bicoherence=0.0, orientados_toro=False):
    """
    Ejecuta las tres formulaciones sobre la misma serie.

    Nota: las propuestas 1 y 2 usan la FFT de longitud N, mientras que la
    propuesta 3 usa la FFT de longitud nperseg. Por tanto, un mismo valor
    numérico de k o r no representa la misma frecuencia física si
    nperseg != N.

    Retorna un diccionario con alpha, k_alpha y detalles de cada método.
    """
    a1, k1, d1 = angulos_alpha_toro_bifases(
        x,
        r1=r1,
        r2=r2,
        amplitude_floor=amplitude_floor,
        orientados=orientados_toro,
    )

    a2, k2, d2 = angulos_alpha_embedding_bifases(
        x,
        r1=r1,
        r2=r2,
        amplitude_floor=amplitude_floor,
    )

    a3, k3, d3 = angulos_alpha_bicoherencia(
        x,
        r1=r1,
        r2=r2,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        min_bicoherence=min_bicoherence,
    )

    return {
        "toro_bifases": {
            "alpha": a1,
            "k_alpha": k1,
            "detalles": d1,
        },
        "embedding_bifases_R4": {
            "alpha": a2,
            "k_alpha": k2,
            "detalles": d2,
        },
        "embedding_bicoherencia_R4": {
            "alpha": a3,
            "k_alpha": k3,
            "detalles": d3,
        },
    }

def circular_varianza(angulos):
    """
    Calcula la varianza circular de un conjunto de ángulos.

    Parameters
    ----------
    angulos : array-like
        Ángulos en radianes.

    Returns
    -------
    varianza_circular : float
        Varianza circular, en el rango [0, 1].
    """
    angulos = np.asarray(angulos)
    if angulos.ndim != 1:
        raise ValueError("angulos debe ser un array unidimensional.")

    n = len(angulos)
    if n == 0:
        return np.nan

    # Convertir ángulos a coordenadas complejas en el círculo unitario
    z = np.exp(1j * angulos)
    R = np.abs(np.mean(z))  # Magnitud del vector promedio

    # Varianza circular
    varianza_circular = 1 - R
    return varianza_circular