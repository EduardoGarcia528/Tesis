import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import mi_libreria as ml


# ============================================================
# CONFIGURACION
# ============================================================

# Parametros del par de Rossler usado habitualmente para estudiar
# sincronizacion de fase.
A = 0.15
B = 0.2
C = 10.0
DELTA_OMEGA = 0.015
OMEGA1 = 1.0 + DELTA_OMEGA
OMEGA2 = 1.0 - DELTA_OMEGA

# Referencia reportada para la transicion de sincronizacion de fase.
# NO se usa para ajustar ni localizar numericamente la transicion;
# solo se dibuja como referencia visual.
EPS_SYNC_REF = 0.028

# Barrido local alrededor de la transicion.
EPS_MIN = 0.015
EPS_MAX = 0.045
N_EPS = 61

# Integracion.
DT = 0.1
T_TRANSITORIO = 1000.0
T_ALINEAMIENTO_LYAP = 200.0
T_ANALISIS = 20000.0
SAMPLE_EVERY = 5       # dt efectivo de las series = DT * SAMPLE_EVERY = 0.1
RENORM_EVERY = 10      # renormalizacion del vector tangente cada 10 pasos

# Condicion inicial [x1,y1,z1,x2,y2,z2]
Y0 = np.array([1.0, 0.0, 0.0, -1.0, 0.5, 0.0], dtype=np.float64)

# Ruido observacional.
# sigma_i = RUIDO_RELATIVO * std(x_i limpio), para cada epsilon.
# RUIDO_RELATIVO = 0 desactiva el ruido.
M_RUIDO = 100
RUIDO_RELATIVO = 0.01
TIPO_RUIDO = "gaussiano"   # "gaussiano" o "uniforme"
SEED_RUIDO = 2026

# Epsilon representativos para inspeccionar las series y la diferencia de fase.
EPS_REPRESENTATIVOS = (0.020, 0.028, 0.035)

# Serie que se muestra en la figura-resumen junto a R_phi y lambda_max.
SERIE_RESUMEN = "x1"       # "x1", "x2", "suma", "diferencia"

# Salidas.
PREFIJO = "rossler_sync_J"
GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = True


# Orden fijo de las cuatro series analizadas.
NOMBRES_SERIES = ("x1", "x2", "suma", "diferencia")
ETIQUETAS_SERIES = (
    r"$x_1(t)$",
    r"$x_2(t)$",
    r"$x_1(t)+x_2(t)$",
    r"$x_1(t)-x_2(t)$",
    # r"Bivariante $x_1(t),x_2(t)$",
)


# ============================================================
# 1. SISTEMA DE DOS ROSSLER ACOPLADOS
# ============================================================

@njit
def rhs_estado(y, eps, w1, w2, a, b, c):
    """
    y = [x1,y1,z1,x2,y2,z2]

    dx1 = -w1*y1 - z1 + eps*(x2-x1)
    dy1 =  w1*x1 + a*y1
    dz1 =  b + z1*(x1-c)

    dx2 = -w2*y2 - z2 + eps*(x1-x2)
    dy2 =  w2*x2 + a*y2
    dz2 =  b + z2*(x2-c)
    """
    out = np.empty(6, dtype=np.float64)

    x1, y1, z1, x2, y2, z2 = y

    out[0] = -w1 * y1 - z1 + eps * (x2 - x1)
    out[1] =  w1 * x1 + a * y1
    out[2] =  b + z1 * (x1 - c)

    out[3] = -w2 * y2 - z2 + eps * (x1 - x2)
    out[4] =  w2 * x2 + a * y2
    out[5] =  b + z2 * (x2 - c)

    return out


@njit
def rk4_estado(y, dt, eps, w1, w2, a, b, c):
    k1 = rhs_estado(y, eps, w1, w2, a, b, c)
    k2 = rhs_estado(y + 0.5 * dt * k1, eps, w1, w2, a, b, c)
    k3 = rhs_estado(y + 0.5 * dt * k2, eps, w1, w2, a, b, c)
    k4 = rhs_estado(y + dt * k3, eps, w1, w2, a, b, c)

    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ============================================================
# 2. SISTEMA VARIACIONAL PARA lambda_max
# ============================================================

@njit
def rhs_aumentado(u, eps, w1, w2, a, b, c):
    """
    Integra simultaneamente el estado y un vector tangente:

        d(delta)/dt = DF(y) delta

    u[:6]  = estado
    u[6:]  = vector tangente
    """
    out = np.empty(12, dtype=np.float64)

    x1, y1, z1, x2, y2, z2 = u[:6]
    v0, v1, v2, v3, v4, v5 = u[6:]

    # Estado
    out[0] = -w1 * y1 - z1 + eps * (x2 - x1)
    out[1] =  w1 * x1 + a * y1
    out[2] =  b + z1 * (x1 - c)

    out[3] = -w2 * y2 - z2 + eps * (x1 - x2)
    out[4] =  w2 * x2 + a * y2
    out[5] =  b + z2 * (x2 - c)

    # Jacobiano por vector, sin construir la matriz 6x6.
    out[6]  = -eps*v0 - w1*v1 - v2 + eps*v3
    out[7]  =  w1*v0 + a*v1
    out[8]  =  z1*v0 + (x1-c)*v2

    out[9]  =  eps*v0 - eps*v3 - w2*v4 - v5
    out[10] =  w2*v3 + a*v4
    out[11] =  z2*v3 + (x2-c)*v5

    return out


@njit
def rk4_aumentado(u, dt, eps, w1, w2, a, b, c):
    k1 = rhs_aumentado(u, eps, w1, w2, a, b, c)
    k2 = rhs_aumentado(u + 0.5 * dt * k1, eps, w1, w2, a, b, c)
    k3 = rhs_aumentado(u + 0.5 * dt * k2, eps, w1, w2, a, b, c)
    k4 = rhs_aumentado(u + dt * k3, eps, w1, w2, a, b, c)

    return u + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit
def integrar_rossler_y_lyapunov(
    eps,
    y0,
    dt,
    t_transitorio,
    t_alineamiento,
    t_analisis,
    sample_every,
    renorm_every,
    w1,
    w2,
    a,
    b,
    c,
):
    """
    1) Elimina transitorio de la orbita.
    2) Alinea el vector tangente sin acumular lambda.
    3) Integra durante t_analisis:
       - guarda la trayectoria muestreada,
       - calcula el maximo exponente de Lyapunov con Benettin.

    Retorna
    -------
    traj : array (N,6)
        [x1,y1,z1,x2,y2,z2] muestreados.
    lambda_max : float
    """
    y = y0.copy()

    n_trans = int(np.round(t_transitorio / dt))
    n_align = int(np.round(t_alineamiento / dt))
    n_ana = int(np.round(t_analisis / dt))

    # Transitorio del estado.
    for _ in range(n_trans):
        y = rk4_estado(y, dt, eps, w1, w2, a, b, c)

    # Vector tangente inicial determinista.
    v = np.array([1.0, 0.37, -0.21, 0.49, -0.31, 0.17], dtype=np.float64)
    v /= np.sqrt(np.sum(v*v))

    u = np.empty(12, dtype=np.float64)
    u[:6] = y
    u[6:] = v

    # Alineamiento del vector tangente con la direccion mas inestable.
    for i in range(n_align):
        u = rk4_aumentado(u, dt, eps, w1, w2, a, b, c)

        if (i + 1) % renorm_every == 0:
            norm_v = np.sqrt(np.sum(u[6:] * u[6:]))
            u[6:] /= norm_v

    n_samples = n_ana // sample_every
    traj = np.empty((n_samples, 6), dtype=np.float64)

    suma_log = 0.0
    tiempo_lyap = 0.0
    j_sample = 0

    for i in range(n_ana):
        u = rk4_aumentado(u, dt, eps, w1, w2, a, b, c)

        if (i + 1) % renorm_every == 0:
            norm_v = np.sqrt(np.sum(u[6:] * u[6:]))
            suma_log += np.log(norm_v)
            tiempo_lyap += renorm_every * dt
            u[6:] /= norm_v

        if (i + 1) % sample_every == 0:
            traj[j_sample, :] = u[:6]
            j_sample += 1

    lambda_max = suma_log / tiempo_lyap
    return traj, lambda_max


# ============================================================
# 3. SINCRONIZACION DE FASE
# ============================================================

def metricas_fase(traj, dt_muestreo):
    """
    Fase geometrica del Rossler:
        phi_i = unwrap(atan2(y_i, x_i))

    R_phi = | < exp(i (phi1-phi2)) > |

    Tambien calcula el desajuste de frecuencias medias y el rango de la
    diferencia de fase desenrollada.
    """
    x1 = traj[:, 0]
    y1 = traj[:, 1]
    x2 = traj[:, 3]
    y2 = traj[:, 4]

    phi1 = np.unwrap(np.arctan2(y1, x1))
    phi2 = np.unwrap(np.arctan2(y2, x2))
    delta_phi = phi1 - phi2

    R_phi = np.abs(np.mean(np.exp(1j * delta_phi)))

    duracion = (len(phi1) - 1) * dt_muestreo
    omega_media_1 = (phi1[-1] - phi1[0]) / duracion
    omega_media_2 = (phi2[-1] - phi2[0]) / duracion
    delta_omega_media = np.abs(omega_media_1 - omega_media_2)

    rango_delta_phi = np.ptp(delta_phi)

    return {
        "phi1": phi1,
        "phi2": phi2,
        "delta_phi": delta_phi,
        "R_phi": R_phi,
        "omega_media_1": omega_media_1,
        "omega_media_2": omega_media_2,
        "delta_omega_media": delta_omega_media,
        "rango_delta_phi": rango_delta_phi,
    }


# ============================================================
# 4. INDICE J: LIMPIO Y CON RUIDO OBSERVACIONAL
# ============================================================

def cuatro_series(x1, x2):
    return (
        x1,
        x2,
        x1 + x2,
        x1 - x2,
    )


def calcular_J_limpio(x1, x2):
    series = cuatro_series(x1, x2)
    
    result = np.array([ml.circular_varianza(ml.angulos_alpha_embedding_bifases(s)[0]) for s in series], dtype=float)
    # result[-1] = ml.indice_J(x1,x2,tau=1)
    return result


def ruido_observacional(rng, n, sigma, tipo):
    if sigma == 0.0:
        return np.zeros(n, dtype=float)

    if tipo == "gaussiano":
        return rng.normal(0.0, sigma, size=n)

    if tipo == "uniforme":
        # Uniforme con desviacion estandar exactamente sigma.
        a = np.sqrt(3.0) * sigma
        return rng.uniform(-a, a, size=n)

    raise ValueError("TIPO_RUIDO debe ser 'gaussiano' o 'uniforme'.")


def calcular_J_con_ruido(
    x1,
    x2,
    M,
    ruido_relativo,
    tipo_ruido,
    seed,
):
    """
    Ruido OBSERVACIONAL, no dinamico.

    Para cada realizacion m:
        x1_obs = x1 + eta1_m
        x2_obs = x2 + eta2_m

    eta1 y eta2 son independientes entre si y entre realizaciones.

    Intensidad:
        std(eta1) = ruido_relativo * std(x1)
        std(eta2) = ruido_relativo * std(x2)

    Luego se construyen suma y diferencia a partir de los dos canales
    observados, de modo que el experimento sea consistente con dos medidas
    independientes del sistema.
    """
    if M < 1:
        raise ValueError("M debe ser >= 1.")

    rng = np.random.default_rng(seed)

    sigma1 = ruido_relativo * np.std(x1, ddof=1)
    sigma2 = ruido_relativo * np.std(x2, ddof=1)

    J_all = np.empty((M, 4), dtype=float)

    for m in range(M):
        eta1 = ruido_observacional(rng, len(x1), sigma1, tipo_ruido)
        eta2 = ruido_observacional(rng, len(x2), sigma2, tipo_ruido)

        x1_obs = x1 + eta1
        x2_obs = x2 + eta2

        series_obs = cuatro_series(x1_obs, x2_obs)

        for j, serie in enumerate(series_obs):
            # J_all[m, j] = ml.indice_J(serie)
            J_all[m, j] = ml.circular_varianza(ml.angulos_alpha_embedding_bifases(serie)[0])
        # J_all[m, -1] = ml.indice_J(x1_obs, x2_obs, tau=1)

    return {
        "J_all": J_all,
        "J_mean": np.mean(J_all, axis=0),
        "J_std": np.std(J_all, axis=0, ddof=1) if M > 1 else np.zeros(4),
        "sigma1": sigma1,
        "sigma2": sigma2,
    }


# ============================================================
# 5. BARRIDO COMPLETO EN epsilon
# ============================================================

def ejecutar_barrido(
    eps_vals,
    y0=Y0,
    dt=DT,
    t_transitorio=T_TRANSITORIO,
    t_alineamiento=T_ALINEAMIENTO_LYAP,
    t_analisis=T_ANALISIS,
    sample_every=SAMPLE_EVERY,
    renorm_every=RENORM_EVERY,
    M_ruido=M_RUIDO,
    ruido_relativo=RUIDO_RELATIVO,
    tipo_ruido=TIPO_RUIDO,
    seed_ruido=SEED_RUIDO,
    eps_representativos=EPS_REPRESENTATIVOS,
):
    eps_vals = np.asarray(eps_vals, dtype=float)
    n_eps = len(eps_vals)

    J_clean = np.empty((n_eps, 4), dtype=float)
    J_noise_all = np.empty((n_eps, M_ruido, 4), dtype=float)
    J_noise_mean = np.empty((n_eps, 4), dtype=float)
    J_noise_std = np.empty((n_eps, 4), dtype=float)

    R_phi = np.empty(n_eps, dtype=float)
    delta_omega = np.empty(n_eps, dtype=float)
    rango_delta_phi = np.empty(n_eps, dtype=float)
    lambda_max = np.empty(n_eps, dtype=float)

    dt_muestreo = dt * sample_every

    # Guardar trayectorias para los epsilon representativos mas cercanos.
    idx_rep = sorted(set(int(np.argmin(np.abs(eps_vals - e))) for e in eps_representativos))
    reps = {}

    for q, eps in enumerate(eps_vals):
        print(f"epsilon {q+1:3d}/{n_eps}: {eps:.6f}")

        traj, lam = integrar_rossler_y_lyapunov(
            eps=eps,
            y0=y0,
            dt=dt,
            t_transitorio=t_transitorio,
            t_alineamiento=t_alineamiento,
            t_analisis=t_analisis,
            sample_every=sample_every,
            renorm_every=renorm_every,
            w1=OMEGA1,
            w2=OMEGA2,
            a=A,
            b=B,
            c=C,
        )

        x1 = traj[:, 0]
        x2 = traj[:, 3]

        fase = metricas_fase(traj, dt_muestreo)

        R_phi[q] = fase["R_phi"]
        delta_omega[q] = fase["delta_omega_media"]
        rango_delta_phi[q] = fase["rango_delta_phi"]
        lambda_max[q] = lam

        J_clean[q, :] = calcular_J_limpio(x1, x2)

        # Semilla distinta y reproducible para cada epsilon.
        seed_q = int(np.random.SeedSequence([seed_ruido, q]).generate_state(1)[0])
        ruido = calcular_J_con_ruido(
            x1=x1,
            x2=x2,
            M=M_ruido,
            ruido_relativo=ruido_relativo,
            tipo_ruido=tipo_ruido,
            seed=seed_q,
        )

        J_noise_all[q, :, :] = ruido["J_all"]
        J_noise_mean[q, :] = ruido["J_mean"]
        J_noise_std[q, :] = ruido["J_std"]

        if q in idx_rep:
            reps[q] = {
                "epsilon": eps,
                "traj": traj.copy(),
                "delta_phi": fase["delta_phi"].copy(),
            }

    return {
        "eps_vals": eps_vals,
        "J_clean": J_clean,
        "J_noise_all": J_noise_all,
        "J_noise_mean": J_noise_mean,
        "J_noise_std": J_noise_std,
        "R_phi": R_phi,
        "delta_omega": delta_omega,
        "rango_delta_phi": rango_delta_phi,
        "lambda_max": lambda_max,
        "dt_muestreo": dt_muestreo,
        "representativos": reps,
    }


# ============================================================
# 6. GRAFICAS
# ============================================================

def terminar_figura(fig, nombre):
    fig.tight_layout()

    if GUARDAR_FIGURAS:
        fig.savefig(nombre, dpi=300, bbox_inches="tight")

    if MOSTRAR_FIGURAS:
        plt.show()
    else:
        plt.close(fig)


def graficar_J_limpio(res):
    eps = res["eps_vals"]
    J = res["J_clean"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()

    for j, ax in enumerate(axes):
        ax.plot(eps, J[:, j], marker=".", markersize=3, linewidth=0.9)
        ax.axvline(EPS_SYNC_REF, linestyle="--", linewidth=1.0,
                   label=rf"$\epsilon_c\approx{EPS_SYNC_REF}$")
        ax.set_ylabel("J")
        ax.set_title(ETIQUETAS_SERIES[j])
        ax.grid(alpha=0.25)
        ax.legend()

    axes[2].set_xlabel(r"Acoplamiento $\epsilon$")
    axes[3].set_xlabel(r"Acoplamiento $\epsilon$")

    terminar_figura(fig, PREFIJO + "_J_limpio.png")


def graficar_sincronizacion(res):
    eps = res["eps_vals"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(eps, res["R_phi"], marker=".", linewidth=0.9)
    axes[0].axvline(EPS_SYNC_REF, linestyle="--")
    axes[0].set_ylabel(r"$R_\phi$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].plot(eps, res["delta_omega"], marker=".", linewidth=0.9)
    axes[1].axvline(EPS_SYNC_REF, linestyle="--")
    axes[1].set_ylabel(r"$|\bar\omega_1-\bar\omega_2|$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(eps, res["rango_delta_phi"], marker=".", linewidth=0.9)
    axes[2].axvline(EPS_SYNC_REF, linestyle="--")
    axes[2].set_ylabel(r"rango de $\Delta\phi$")
    axes[2].set_xlabel(r"Acoplamiento $\epsilon$")
    axes[2].grid(alpha=0.25)

    terminar_figura(fig, PREFIJO + "_sincronizacion.png")


def graficar_lyapunov(res):
    eps = res["eps_vals"]
    lam = res["lambda_max"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(eps, lam, marker=".", markersize=4, linewidth=0.9)
    ax.axhline(0.0, linestyle=":", label=r"$\lambda_{\max}=0$")
    ax.axvline(EPS_SYNC_REF, linestyle="--",
               label=rf"$\epsilon_c\approx{EPS_SYNC_REF}$")
    ax.set_xlabel(r"Acoplamiento $\epsilon$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.grid(alpha=0.25)
    ax.legend()

    terminar_figura(fig, PREFIJO + "_lambda_max.png")


def graficar_J_ruido(res):
    eps = res["eps_vals"]
    J0 = res["J_clean"]
    Jm = res["J_noise_mean"]
    Js = res["J_noise_std"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()

    for j, ax in enumerate(axes):
        ax.plot(eps, J0[:, j], linestyle="--", linewidth=1.0,
                label="sin ruido")
        ax.plot(eps, Jm[:, j], linewidth=1.0,
                label=rf"ruido: media de $M={M_RUIDO}$")
        ax.fill_between(
            eps,
            Jm[:, j] - Js[:, j],
            Jm[:, j] + Js[:, j],
            alpha=0.2,
            label=r"$\pm 1\sigma$ entre realizaciones",
        )
        ax.axvline(EPS_SYNC_REF, linestyle=":")
        ax.set_ylabel("J")
        ax.set_title(ETIQUETAS_SERIES[j])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[2].set_xlabel(r"Acoplamiento $\epsilon$")
    axes[3].set_xlabel(r"Acoplamiento $\epsilon$")

    terminar_figura(fig, PREFIJO + "_J_ruido.png")


def graficar_representativos(res, ventana_t=300.0):
    reps = res["representativos"]
    if not reps:
        return

    dt_s = res["dt_muestreo"]
    claves = sorted(reps.keys())

    fig, axes = plt.subplots(len(claves), 2, figsize=(12, 3.3*len(claves)))
    if len(claves) == 1:
        axes = np.array([axes])

    n_ventana = max(10, int(round(ventana_t / dt_s)))

    for fila, q in enumerate(claves):
        rep = reps[q]
        traj = rep["traj"]
        dphi = rep["delta_phi"]
        eps = rep["epsilon"]

        n = min(n_ventana, len(traj))
        t = np.arange(n) * dt_s

        x1 = traj[-n:, 0]
        x2 = traj[-n:, 3]
        d = dphi[-n:]

        axes[fila, 0].plot(t, x1, linewidth=0.8, label=r"$x_1$")
        axes[fila, 0].plot(t, x2, linewidth=0.8, label=r"$x_2$")
        axes[fila, 0].set_ylabel(rf"$\epsilon={eps:.3f}$")
        axes[fila, 0].grid(alpha=0.25)
        axes[fila, 0].legend()

        axes[fila, 1].plot(t, d, linewidth=0.8)
        axes[fila, 1].grid(alpha=0.25)
        axes[fila, 1].set_ylabel(r"$\phi_1-\phi_2$")

    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    axes[0, 0].set_title("Series temporales")
    axes[0, 1].set_title("Diferencia de fase desenrollada")

    terminar_figura(fig, PREFIJO + "_representativos.png")


def graficar_resumen(res, serie=SERIE_RESUMEN):
    if serie not in NOMBRES_SERIES:
        raise ValueError(f"serie debe ser una de {NOMBRES_SERIES}")

    j = NOMBRES_SERIES.index(serie)
    eps = res["eps_vals"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(eps, res["J_clean"][:, j], marker=".", linewidth=0.9)
    axes[0].axvline(EPS_SYNC_REF, linestyle="--")
    axes[0].set_ylabel("J")
    axes[0].set_title("J para " + ETIQUETAS_SERIES[j])
    axes[0].grid(alpha=0.25)

    axes[1].plot(eps, res["R_phi"], marker=".", linewidth=0.9)
    axes[1].axvline(EPS_SYNC_REF, linestyle="--")
    axes[1].set_ylabel(r"$R_\phi$")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)

    axes[2].plot(eps, res["lambda_max"], marker=".", linewidth=0.9)
    axes[2].axhline(0.0, linestyle=":")
    axes[2].axvline(EPS_SYNC_REF, linestyle="--")
    axes[2].set_ylabel(r"$\lambda_{\max}$")
    axes[2].set_xlabel(r"Acoplamiento $\epsilon$")
    axes[2].grid(alpha=0.25)

    terminar_figura(fig, PREFIJO + "_resumen.png")


# ============================================================
# 7. GUARDADO DE DATOS
# ============================================================

def guardar_resultados(res):
    np.savez(
        PREFIJO + "_resultados.npz",
        eps_vals=res["eps_vals"],
        J_clean=res["J_clean"],
        J_noise_all=res["J_noise_all"],
        J_noise_mean=res["J_noise_mean"],
        J_noise_std=res["J_noise_std"],
        R_phi=res["R_phi"],
        delta_omega=res["delta_omega"],
        rango_delta_phi=res["rango_delta_phi"],
        lambda_max=res["lambda_max"],
        nombres_series=np.array(NOMBRES_SERIES),
        omega1=np.array(OMEGA1),
        omega2=np.array(OMEGA2),
        a=np.array(A),
        b=np.array(B),
        c=np.array(C),
        eps_sync_ref=np.array(EPS_SYNC_REF),
        dt=np.array(DT),
        dt_muestreo=np.array(res["dt_muestreo"]),
        t_transitorio=np.array(T_TRANSITORIO),
        t_alineamiento_lyap=np.array(T_ALINEAMIENTO_LYAP),
        t_analisis=np.array(T_ANALISIS),
        M_ruido=np.array(M_RUIDO),
        ruido_relativo=np.array(RUIDO_RELATIVO),
        tipo_ruido=np.array(TIPO_RUIDO),
        seed_ruido=np.array(SEED_RUIDO),
    )

    # CSV largo por epsilon, util para inspeccion y para compartir resultados.
    columnas = [
        res["eps_vals"],
        res["R_phi"],
        res["delta_omega"],
        res["rango_delta_phi"],
        res["lambda_max"],
    ]
    header = [
        "epsilon",
        "R_phi",
        "delta_omega_media",
        "rango_delta_phi",
        "lambda_max",
    ]

    for j, nombre in enumerate(NOMBRES_SERIES):
        columnas.extend([
            res["J_clean"][:, j],
            res["J_noise_mean"][:, j],
            res["J_noise_std"][:, j],
        ])
        header.extend([
            f"J_{nombre}_clean",
            f"J_{nombre}_noise_mean",
            f"J_{nombre}_noise_std",
        ])

    tabla = np.column_stack(columnas)
    np.savetxt(
        PREFIJO + "_resultados.csv",
        tabla,
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    # Trayectorias representativas.
    reps = res["representativos"]
    if reps:
        claves = sorted(reps.keys())
        eps_rep = np.array([reps[q]["epsilon"] for q in claves])
        traj_rep = np.stack([reps[q]["traj"] for q in claves], axis=0)
        dphi_rep = np.stack([reps[q]["delta_phi"] for q in claves], axis=0)

        np.savez(
            PREFIJO + "_representativos.npz",
            eps_rep=eps_rep,
            traj=traj_rep,
            delta_phi=dphi_rep,
            dt_muestreo=np.array(res["dt_muestreo"]),
        )


# ============================================================
# 8. EJECUCION
# ============================================================

if __name__ == "__main__":

    eps_vals = np.linspace(EPS_MIN, EPS_MAX, N_EPS)

    print("=" * 70)
    print("DOS ROSSLER ACOPLADOS: SINCRONIZACION DE FASE E INDICE J")
    print("=" * 70)
    print(f"omega1 = {OMEGA1:.6f}")
    print(f"omega2 = {OMEGA2:.6f}")
    print(f"epsilon en [{EPS_MIN}, {EPS_MAX}] con N={N_EPS}")
    print(f"dt = {DT}; dt_muestreo = {DT*SAMPLE_EVERY}")
    print(f"T_transitorio = {T_TRANSITORIO}")
    print(f"T_analisis = {T_ANALISIS}")
    print(f"M_ruido = {M_RUIDO}")
    print(f"ruido relativo = {RUIDO_RELATIVO}")
    print(f"tipo de ruido = {TIPO_RUIDO}")
    print("=" * 70)

    resultados = ejecutar_barrido(eps_vals)

    guardar_resultados(resultados)

    graficar_J_limpio(resultados)
    graficar_sincronizacion(resultados)
    graficar_lyapunov(resultados)
    graficar_J_ruido(resultados)
    graficar_representativos(resultados)
    graficar_resumen(resultados, SERIE_RESUMEN)

    print("\nArchivos guardados:")
    print("  ", PREFIJO + "_resultados.npz")
    print("  ", PREFIJO + "_resultados.csv")
    print("  ", PREFIJO + "_representativos.npz")
    print("  figuras PNG con el mismo prefijo")
