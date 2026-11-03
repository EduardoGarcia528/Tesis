import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.special import gammaln,loggamma
from scipy.integrate import trapezoid
from scipy.optimize import least_squares

# ============================================================
# BRF GENERADOR
# ============================================================

@njit
def BRF(N, A, a, b, DGBD=False, normalizar=False):
    """
    Genera una secuencia de valores ordenados por rango mediante
    la función BRF (Beta Rank Function) o su forma DGBD.

    La forma BRF se define como:

        x_r = A * (N + 1)**(a - b)
              * (N + 1 - r)**b / r**a

    mientras que, cuando `DGBD=True`, se utiliza directamente:

        x_r = A * (N + 1 - r)**b / r**a

    con:

        u = r / (N + 1)

    donde `r` representa el rango de cada elemento.

    Parameters
    ----------
    N : int
        Número de elementos que tendrá la secuencia.

    A : float
        Parámetro de escala o amplitud de la distribución.

    a : float
        Exponente asociado al término de rango `r`. Controla
        principalmente el comportamiento de la cola de rangos bajos.

    b : float
        Exponente asociado al término `(N + 1 - r)`. Controla
        principalmente el comportamiento de la cola de rangos altos.

    DGBD : bool, default=False
        Determina la forma de la función utilizada.

        - `False`: genera la forma BRF, incluyendo el factor de
          normalización `(N + 1)**(a - b)`.
        - `True`: genera la forma DGBD sin dicho factor.

    normalizar : bool, default=False
        Si es `True`, normaliza la secuencia dividiendo todos sus
        elementos entre la suma total:

            x_i <- x_i / sum(x)

        Por tanto, la suma de los elementos resultantes será igual
        a 1.

    Returns
    -------
    numpy.ndarray
        Arreglo de longitud `N` que contiene los valores generados,
        ordenados de acuerdo con su rango.

    Notes
    -----
    La secuencia se construye para los rangos:

        r = 1, 2, ..., N

    por lo que el valor correspondiente al rango `r` se almacena en
    `x[r - 1]`.

    El factor `(N + 1)**(a - b)` diferencia la parametrización BRF
    de la forma DGBD utilizada cuando `DGBD=True`.

    La función está compilada con Numba mediante `@njit`, lo que
    permite acelerar su ejecución cuando se utiliza repetidamente,
    por ejemplo, durante procesos de ajuste o simulaciones.

    Examples
    --------
    Generar una secuencia BRF:

        x = BRF(700, 56.0, 0.7, 0.5)

    Generar la forma DGBD:

        x = BRF(700, 56.0, 0.7, 0.5, DGBD=True)

    Generar una secuencia normalizada:

        x = BRF(700, 56.0, 0.7, 0.5, normalizar=True)

    """
    x = np.empty(N)

    pref = (N + 1) ** (a - b)

    for i in range(N):
        r = i + 1

        if not DGBD:
            x[i] = (
                A
                * pref
                * ((N + 1 - r) ** b)
                / (r ** a)
            )
        else:
            x[i] = (
                A
                * ((N + 1 - r) ** b)
                / (r ** a)
            )

    if normalizar:
        x /= np.sum(x)

    return x


def fit_rank(dat, usar_BRF=True, normalizar=False):
    """
    Ajusta una distribución BRF o DGBD a un conjunto de datos
    ordenados por rango.

    El ajuste puede realizarse sobre los datos en su escala original
    (`normalizar=False`) o sobre una versión normalizada de los datos
    (`normalizar=True`).

    También puede elegirse entre dos parametrizaciones:

        BRF:
            y_r = A (N + 1)^(a-b)
                  (N + 1 - r)^b / r^a

        DGBD:
            y_r = A
                  (N + 1 - r)^b / r^a

    donde:

        r = 1, 2, ..., N

    es el rango y `N` es el número total de elementos.

    Parameters
    ----------
    dat : array-like
        Datos de entrada. Los valores deben ser positivos para poder
        realizar el ajuste en espacio logarítmico.

    usar_BRF : bool, default=True
        Selecciona la parametrización utilizada.

        - `True`: utiliza la parametrización BRF:
          
              y_r = A (N + 1)^(a-b)
                    (N + 1-r)^b / r^a

        - `False`: utiliza la parametrización DGBD:

              y_r = A (N + 1-r)^b / r^a

    normalizar : bool, default=False
        Determina si el ajuste se realiza sobre datos normalizados.

        - `False`: conserva la escala original de los datos y ajusta
          los tres parámetros `(A, a, b)`.

        - `True`: normaliza los datos de modo que:

              sum(y_r) = 1

          En este caso la amplitud `A` deja de ser un parámetro
          independiente, por lo que únicamente se ajustan `(a, b)`.

    Returns
    -------
    numpy.ndarray
        Parámetros ajustados.

        Si `normalizar=False`:

            [A, a, b]

        Si `normalizar=True`:

            [a, b]

    Notes
    -----
    El ajuste se realiza en espacio logarítmico. La inicialización
    de los parámetros se obtiene mediante una regresión lineal de:

        log(y_r) = log(A) + b log(N+1-r) - a log(r)

    para la parametrización DGBD.

    Para la parametrización BRF se utiliza la forma equivalente:

        log(y_r) = log(A)
                   + (a-b) log(N+1)
                   + b log(N+1-r)
                   - a log(r)

    Cuando `normalizar=True`, la constante de escala se elimina
    mediante la normalización del modelo:

        p_r = f_r / sum(f_r)

    y el ajuste se realiza minimizando:

        log(p_r) - log(y_r)

    mediante `scipy.optimize.least_squares`.

    La restricción:

        a < 1

    garantiza la convergencia de la suma asociada al término
    r^(-a) en el límite N -> infinito.

    La restricción:

        b > -1

    evita una divergencia en el extremo r -> N de la forma
    (N+1-r)^b.

    Examples
    --------
    Ajuste BRF sin normalizar:

        A, a, b = fit_rank(data, usar_BRF=True, normalizar=False)

    Ajuste DGBD sin normalizar:

        A, a, b = fit_rank(data, usar_BRF=False, normalizar=False)

    Ajuste BRF normalizado:

        a, b = fit_rank(data, usar_BRF=True, normalizar=True)

    Ajuste DGBD normalizado:

        a, b = fit_rank(data, usar_BRF=False, normalizar=True)

    See Also
    --------
    BRF : Genera datos mediante las parametrizaciones BRF o DGBD.
    """

    # ==========================================================
    # Preparación de los datos
    # ==========================================================

    dat = np.asarray(dat, dtype=float)

    # Orden descendente por rango
    y = np.sort(dat)[::-1]

    # Número original de elementos
    N = len(y)

    # Rangos
    r = np.arange(1, N + 1)

    # Mantener únicamente valores positivos
    mask = y > 0

    y = y[mask]
    r = r[mask]

    # Verificación básica
    if len(y) < 3:
        raise ValueError(
            "Se necesitan al menos 3 valores positivos para realizar "
            "el ajuste."
        )

    # ==========================================================
    # Normalización de los datos
    # ==========================================================

    if normalizar:
        y = y / np.sum(y)

    # ==========================================================
    # Inicialización lineal
    # ==========================================================

    if usar_BRF:

        # ------------------------------------------------------
        # BRF
        #
        # f(r) =
        # (1-u)^b / u^a
        #
        # con:
        # u = r/(N+1)
        # ------------------------------------------------------

        u = r / (N + 1)

        X1 = np.log(1 - u)
        X2 = np.log(u)

    else:

        # ------------------------------------------------------
        # DGBD
        #
        # f(r) =
        # (N+1-r)^b / r^a
        # ------------------------------------------------------

        X1 = np.log(N + 1 - r)
        X2 = np.log(r)

    Y = np.log(y)

    M = np.column_stack([
        np.ones_like(X1),
        X1,
        -X2
    ])

    coef, *_ = np.linalg.lstsq(
        M,
        Y,
        rcond=None
    )

    # ==========================================================
    # Semillas iniciales
    # ==========================================================

    if normalizar:

        # La constante de escala no es necesaria
        b0 = coef[1]
        a0 = coef[2]

    else:

        # Se conserva la escala original
        log_A0 = coef[0]

        # La regresión lineal con BRF contiene el término:
        #
        # log(A) + (a-b)log(N+1)
        #
        # Por ello hay que corregir la amplitud.

        if usar_BRF:
            A0 = np.exp(
                log_A0 - (a0 if False else 0)
            )
        else:
            A0 = np.exp(log_A0)

        b0 = coef[1]
        a0 = coef[2]

        if usar_BRF:
            A0 = np.exp(
                log_A0 - (a0 - b0) * np.log(N + 1)
            )

    # ==========================================================
    # Ajuste normalizado
    # ==========================================================

    if normalizar:

        # ------------------------------------------------------
        # Proyectar la semilla a la región normalizable
        # ------------------------------------------------------

        a0 = min(a0, 0.99)
        b0 = max(b0, -0.99)

        # ------------------------------------------------------
        # Modelo sin amplitud
        # ------------------------------------------------------

        def model(p):

            a, b = p

            if usar_BRF:

                u = r / (N + 1)

                pred = (
                    (1 - u) ** b
                    /
                    u ** a
                )

            else:

                pred = (
                    (N + 1 - r) ** b
                    /
                    r ** a
                )

            # Normalización del modelo
            pred /= np.sum(pred)

            return np.log(pred) - np.log(y)

        sol = least_squares(
            model,
            x0=(a0, b0),
            bounds=(
                [-np.inf, -0.999999],
                [0.999999, np.inf]
            ),
            method="trf"
        )

        a_fit, b_fit = sol.x

        return a_fit, b_fit

    # ==========================================================
    # Ajuste sin normalizar
    # ==========================================================

    else:

        def model(p):

            A, a, b = p

            if usar_BRF:

                pref = (N + 1) ** (a - b)

                pred = (
                    A
                    * pref
                    * (N + 1 - r) ** b
                    / r ** a
                )

            else:

                pred = (
                    A
                    * (N + 1 - r) ** b
                    / r ** a
                )

            return np.log(pred) - np.log(y)

        sol = least_squares(
            model,
            x0=(A0, a0, b0),
            bounds=(
                [0, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf]
            ),
            method="trf"
        )

        A_fit, a_fit, b_fit = sol.x

        return A_fit, a_fit, b_fit



def cumulantes_teoricos(N,a,b,A=None,usar_BRF=True,normalizar=False):

    """
    Calcula los cuatro primeros cumulantes teóricos asociados
    a una distribución BRF o DGBD.

    Los cumulantes corresponden a la variable logarítmica:

        z = log(x)

    y se definen como:

        kappa_n = cumulante_n(z)

    La función permite calcular los cumulantes tanto para la
    distribución en su escala original como para su versión
    normalizada.

    Modelos
    -------
    BRF
        La función de rango utilizada es:

            x_r = A (N + 1)^(a-b)
                  (N + 1-r)^b / r^a

        o, equivalentemente,

            x_r = A (1-u)^b / u^a

        con:

            u = r / (N+1)

    DGBD
        La función de rango utilizada es:

            x_r = A (N + 1-r)^b / r^a

    Parameters
    ----------
    N : int
        Número de elementos de la distribución.

    a : float
        Exponente asociado al término de rango `r`.

    b : float
        Exponente asociado al término `(N+1-r)`.

    A : float or None, default=None
        Parámetro de escala.

        Es necesario cuando `normalizar=False`, ya que en ese
        caso el primer cumulante depende explícitamente de `A`.

        Cuando `normalizar=True`, `A` no es necesario porque la
        normalización elimina la escala absoluta.

    usar_BRF : bool, default=True
        Selecciona la parametrización utilizada.

        - `True`: BRF.
        - `False`: DGBD.

    normalizar : bool, default=False
        Determina si se calculan los cumulantes de la distribución
        en su escala original o de la distribución normalizada.

        - `False`: utiliza la escala definida por `A`.
        - `True`: normaliza la distribución de modo que:

              sum(x_r) = 1

    Returns
    -------
    numpy.ndarray
        Arreglo:

            [kappa_1, kappa_2, kappa_3, kappa_4]

        correspondiente a los cuatro primeros cumulantes de
        `log(x)`.

    Notes
    -----
    Los cumulantes de segundo, tercer y cuarto orden son:

        kappa_2 =
            (a-b)^2 + (pi^2/3)ab

        kappa_3 =
            2 zeta(3)(a^3-b^3)
            + 2(zeta(3)-1)(b-a)^3

        kappa_4 =
            (pi^4/15)
            [a^4+b^4-(b-a)^4]
            + 6(b-a)^4

    Estas expresiones son comunes a las parametrizaciones BRF y
    DGBD.

    El primer cumulante, `kappa_1`, depende de la parametrización
    y de si la distribución está normalizada.

    Para la distribución sin normalizar:

        BRF:

            kappa_1 = log(A) + a - b

        DGBD:

            kappa_1 = log(A)
                      + ((b-a)/N) log(Gamma(N+1))

    donde se utiliza:

        log(Gamma(N+1)) = gammaln(N+1)

    Para la distribución normalizada:

        BRF:

            kappa_1 = -log(Z) + a - b

        DGBD:

            kappa_1 =
                ((b-a)/N) log(Gamma(N+1))
                - log(Z)

    con `Z` definido como la suma de los términos de la
    distribución antes de normalizar.

    Examples
    --------
    Cumulantes BRF sin normalizar:

        k = cumulantes_teoricos(
            N=700,
            A=56.0,
            a=0.7,
            b=0.5,
            usar_BRF=True,
            normalizar=False
        )

    Cumulantes BRF normalizados:

        k = cumulantes_teoricos(
            N=700,
            a=0.7,
            b=0.5,
            usar_BRF=True,
            normalizar=True
        )

    Cumulantes DGBD sin normalizar:

        k = cumulantes_teoricos(
            N=700,
            A=56.0,
            a=0.7,
            b=0.5,
            usar_BRF=False,
            normalizar=False
        )

    Cumulantes DGBD normalizados:

        k = cumulantes_teoricos(
            N=700,
            a=0.7,
            b=0.5,
            usar_BRF=False,
            normalizar=True
        )
    """

    # ==========================================================
    # Constante de Apéry
    # ==========================================================

    zeta3 = 1.202056903159594

    # ==========================================================
    # Cumulante de primer orden
    # ==========================================================

    if normalizar:

        r = np.arange(1, N + 1)

        if usar_BRF:

            # --------------------------------------------------
            # BRF normalizada
            # --------------------------------------------------

            u = r / (N + 1)

            f = (
                (1 - u) ** b
                /
                u ** a
            )

            Z = np.sum(f)

            k1 = -np.log(Z) + a - b

        else:

            # --------------------------------------------------
            # DGBD normalizada
            # --------------------------------------------------

            f = (
                (N + 1 - r) ** b
                /
                r ** a
            )

            Z = np.sum(f)

            k1 = (
                ((b - a) / N)
                * gammaln(N + 1)
                - np.log(Z)
            )

    else:

        # ------------------------------------------------------
        # Distribución sin normalizar
        # ------------------------------------------------------

        if A is None:
            raise ValueError(
                "A debe especificarse cuando normalizar=False."
            )

        if usar_BRF:

            # BRF
            k1 = np.log(A) + a - b

        else:

            # DGBD
            k1 = (
                np.log(A)
                + ((b - a) / N)
                * gammaln(N + 1)
            )

    # ==========================================================
    # Cumulante de segundo orden
    # ==========================================================

    k2 = (
        (a - b) ** 2
        + (np.pi ** 2 / 3) * a * b
    )

    # ==========================================================
    # Cumulante de tercer orden
    # ==========================================================

    k3 = (
        2 * zeta3 * (a ** 3 - b ** 3)
        + 2 * (zeta3 - 1) * (b - a) ** 3
    )

    # ==========================================================
    # Cumulante de cuarto orden
    # ==========================================================

    k4 = (
        (np.pi ** 4 / 15)
        * (
            a ** 4
            + b ** 4
            - (b - a) ** 4
        )
        + 6 * (b - a) ** 4
    )

    # ==========================================================
    # Vector de cumulantes
    # ==========================================================

    return np.array([k1,k2,k3,k4])




def Beta_entropy(
    serie,
    A=None,
    a=None,
    b=None,
    normalize=False
):
    """
    Calcula la entropía diferencial de Shannon de la distribución
    BRF asociada a una serie de datos.

    La función obtiene los parámetros de la distribución BRF,
    construye la densidad de probabilidad mediante la función
    generadora de cumulantes (CGF) y su inversión numérica mediante
    FFT, y finalmente calcula la entropía diferencial de Shannon.

    La entropía se define como:

        H(X) = -∫ f_X(x) log_2[f_X(x)] dx

    donde `f_X(x)` es la densidad de probabilidad continua de la
    variable `X`.

    El procedimiento completo es:

        datos
          ↓
        ajuste BRF
          ↓
        parámetros (A, a, b)
          ↓
        CGF de z = log(x)
          ↓
        función característica
          ↓
        inversión de Fourier (FFT)
          ↓
        PDF de x
          ↓
        entropía diferencial de Shannon

    Parameters
    ----------
    serie : array-like
        Serie de datos utilizada para obtener la distribución.

        Si `A`, `a` y `b` no se proporcionan, los parámetros se
        estiman automáticamente mediante `fit_rank()`.

    A : float or None, default=None
        Parámetro de escala de la distribución BRF.

        Si se proporcionan `A`, `a` y `b`, estos valores se utilizan
        directamente y no se realiza un nuevo ajuste.

    a : float or None, default=None
        Exponente asociado al término de rango `r`.

    b : float or None, default=None
        Exponente asociado al término `(N + 1 - r)`.

    normalize : bool, default=False
        Determina si la entropía diferencial se normaliza.

        Si `False`, devuelve directamente:

            H(X) = -∫ f_X(x) log_2[f_X(x)] dx

        en bits.

        Si `True`, divide la entropía entre la entropía máxima de
        una distribución uniforme definida sobre el intervalo
        cubierto por la PDF:

            H_max = log_2(x_max - x_min)

        obteniendo:

            H_norm = H / H_max

    Returns
    -------
    float
        Entropía diferencial de Shannon en bits.

        Si `normalize=True`, devuelve la entropía normalizada
        respecto a la entropía máxima correspondiente al intervalo
        considerado.

    Notes
    -----
    La distribución BRF utilizada es:

        x_r =
            A (N + 1)^(a-b)
            (N + 1-r)^b / r^a

    que puede escribirse equivalentemente como:

        x_r =
            A (1-u)^b / u^a

    con:

        u = r / (N + 1)

    Para obtener la PDF se trabaja con la variable logarítmica:

        z = log(x)

    cuya función generadora de cumulantes es:

        K_z(t) =
            t log(A)
            + log Γ(1-a t)
            + log Γ(1+b t)
            - log Γ(2+(b-a)t)

    donde Γ representa la función Gamma.

    La función característica de `z` se obtiene mediante:

        φ_z(ω) = exp[K_z(iω)]

    y la densidad de `z` se recupera mediante inversión numérica
    de Fourier utilizando una FFT.

    Posteriormente se realiza el cambio de variable:

        x = exp(z)

    por lo que:

        f_X(x) = f_Z(log(x)) / x

    La entropía se calcula mediante integración numérica de:

        H(X) =
            -∫ f_X(x) log_2[f_X(x)] dx

    utilizando la regla del trapecio.

    Antes de realizar la integración se eliminan los puntos donde
    la PDF es cero, negativa o no finita, evitando así problemas
    numéricos asociados con `log2`.

    La malla utilizada para la inversión de Fourier se determina
    dinámicamente a partir de los dos primeros cumulantes teóricos
    de `log(x)`:

        μ =
            log(A) + a - b

        σ² =
            (a-b)² + (π²/3)ab

    La malla se extiende aproximadamente 10 desviaciones estándar
    alrededor de la media y utiliza 60 puntos por desviación
    estándar.

    Examples
    --------
    Calcular automáticamente la entropía:

        H = Beta_shannon_pdf(serie)

    Utilizar parámetros BRF conocidos:

        H = Beta_shannon_pdf(
            serie,
            A=56.0,
            a=0.7,
            b=0.5
        )

    Calcular la entropía normalizada:

        H = Beta_shannon_pdf(
            serie,
            normalize=True
        )

    Utilizar parámetros conocidos y normalizar la entropía:

        H = Beta_shannon_pdf(
            serie,
            A=56.0,
            a=0.7,
            b=0.5,
            normalize=True
        )

    See Also
    --------
    fit_rank : Ajusta los parámetros A, a y b de la distribución.
    BRF : Genera una secuencia mediante la función BRF.
    cumulantes_teoricos : Calcula los cumulantes teóricos de log(x).
    """

    # ==========================================================
    # Función auxiliar: construcción de la PDF mediante el CGF
    # ==========================================================

    def pdf_from_cgf_dinamico(A, a, b, N):
        """
        Construye numéricamente la PDF de x a partir del CGF
        de z = log(x).

        Parameters
        ----------
        A, a, b : float
            Parámetros de la distribución BRF.

        N : int
            Número de elementos de la serie.

        Returns
        -------
        x : numpy.ndarray
            Malla correspondiente a la variable original x.

        pdf_x : numpy.ndarray
            Densidad de probabilidad correspondiente a x.
        """

        # ------------------------------------------------------
        # Selección dinámica de la malla de Fourier
        # ------------------------------------------------------

        def elegir_malla(
            A,
            a,
            b,
            N,
            nsigmas=10,
            puntos_sigma=60
        ):
            """
            Determina la extensión y resolución de la malla de
            Fourier a partir de los cumulantes de log(x).
            """

            # Primer cumulante:
            #
            # μ = log(A) + a - b

            mu = np.log(A) + a - b

            # Segundo cumulante:
            #
            # σ² = (a-b)² + (π²/3)ab

            sigma2 = (
                (a - b) ** 2
                + (np.pi ** 2 / 3) * a * b
            )

            sigma = np.sqrt(sigma2)

            # Extensión de la malla en el espacio logarítmico

            ymax = (
                abs(mu)
                + nsigmas * sigma
            )

            # Resolución espacial

            dy = sigma / puntos_sigma

            # Frecuencia máxima

            wmax = np.pi / dy

            # Número inicial de puntos

            M = int(
                np.ceil(
                    2 * ymax / dy
                )
            )

            # Utilizar una potencia de dos favorece la eficiencia
            # computacional de la FFT

            M = 2 ** int(
                np.ceil(
                    np.log2(M)
                )
            )

            return wmax, M

        # ------------------------------------------------------
        # Función generadora de cumulantes
        # ------------------------------------------------------

        def Kz(t, A, a, b):
            """
            Función generadora de cumulantes de z = log(x).

            K_z(t) =
                t log(A)
                + log Γ(1-at)
                + log Γ(1+bt)
                - log Γ(2+(b-a)t)
            """

            return (
                t * np.log(A)
                + loggamma(1 - a * t)
                + loggamma(1 + b * t)
                - loggamma(2 + (b - a) * t)
            )

        # ------------------------------------------------------
        # Construcción de la malla
        # ------------------------------------------------------

        wmax, M = elegir_malla(
            A,
            a,
            b,
            N
        )

        dw = 2 * wmax / M

        w = (
            np.arange(
                -M // 2,
                M // 2
            )
            * dw
        )

        # ------------------------------------------------------
        # Función característica
        # ------------------------------------------------------

        phi = np.exp(
            Kz(
                1j * w,
                A,
                a,
                b
            )
        )

        # ------------------------------------------------------
        # Malla en el espacio logarítmico
        # ------------------------------------------------------

        dx = 2 * np.pi / (M * dw)

        y = (
            np.arange(
                -M // 2,
                M // 2
            )
            * dx
        )

        # ------------------------------------------------------
        # Inversión de Fourier
        # ------------------------------------------------------

        pdf_y = (
            np.real(
                np.fft.fftshift(
                    np.fft.ifft(
                        np.fft.ifftshift(
                            np.conj(phi)
                        )
                    )
                )
            )
            / dx
        )

        # Eliminar pequeños valores negativos producidos por
        # errores numéricos de la inversión de Fourier

        pdf_y[pdf_y < 1e-12] = 0

        # ------------------------------------------------------
        # Transformación z -> x
        # ------------------------------------------------------

        x = np.exp(y)

        # Cambio de variable:
        #
        # f_X(x) = f_Z(log(x)) / x

        pdf_x = pdf_y / x

        return x, pdf_x

    # ==========================================================
    # Obtención de los parámetros BRF
    # ==========================================================

    if (
        A is None
        or a is None
        or b is None
    ):

        # Si no se proporcionan los parámetros completos,
        # se realiza el ajuste automáticamente.

        A, a, b = fit_rank(
            serie,
            usar_BRF=True,
            normalizar=False
        )

    # ==========================================================
    # Tamaño de la serie
    # ==========================================================

    N = len(serie)

    # ==========================================================
    # Construcción de la PDF
    # ==========================================================

    x, pdf_x = pdf_from_cgf_dinamico(
        A,
        a,
        b,
        N
    )

    x = np.asarray(
        x,
        dtype=np.float64
    )

    pdf_x = np.asarray(
        pdf_x,
        dtype=np.float64
    )

    # ==========================================================
    # Filtrado de valores no válidos
    # ==========================================================

    # La entropía solamente puede evaluarse donde:
    #
    #     f_X(x) > 0
    #
    # Además, se eliminan valores no finitos.

    mask = (
        (pdf_x > 0)
        & np.isfinite(pdf_x)
        & np.isfinite(x)
    )

    x_f = x[mask]
    pdf_f = pdf_x[mask]

    # Si no existen puntos válidos, no puede calcularse
    # la entropía.

    if x_f.size == 0:
        return 0.0

    # ==========================================================
    # Integrando de Shannon
    # ==========================================================

    # H(X) =
    #     -∫ f_X(x) log2[f_X(x)] dx

    integrando = (
        pdf_f
        * np.log2(pdf_f)
    )

    # ==========================================================
    # Integración numérica
    # ==========================================================

    H = -trapezoid(
        integrando,
        x_f
    )

    # ==========================================================
    # Normalización de la entropía
    # ==========================================================

    if normalize:

        # Entropía máxima de una distribución uniforme
        # sobre [xmin, xmax]:
        #
        # H_max = log2(xmax - xmin)

        rango = (
            x_f.max()
            - x_f.min()
        )

        if rango > 0:

            H_max = np.log2(rango)

            if H_max != 0:
                H = H / H_max

            else:
                H = 0.0

    # ==========================================================
    # Resultado
    # ==========================================================

    return H