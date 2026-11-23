# -*- coding: utf-8 -*-
"""
SINCRONIZACION GENERALIZADA: MAPAS LOGISTICOS + INDICE J
=======================================================
Modelo conductor-respuesta:
    x[n+1] = 4*x[n]*(1-x[n])
    y[n+1] = (1-eps)*3.8*y[n]*(1-y[n]) + eps*x[n+1]

Objetivos:
  (1) Repetir las simulaciones (condiciones iniciales independientes) y
      obtener dispersion e IC bootstrap del 95 % para J(y), J(x,y) y lambda_c.
  (4) Contrastar J con nulos shuffle e IAAFT, usando
      ml.iaaft(y, ns, tol_pc=..., verbose=False, maxiter=...).

NULO UNIVARIANTE: J(y_surr) frente a J(y).
NULO BIVARIANTE: J(x, y_surr) frente a J(x,y).
  x se conserva intacta, y se sustituye por una serie independiente
  respecto a x; este nulo rompe el acoplamiento X-Y. El IAAFT mantiene,
  aproximadamente, el espectro y distribucion de y.

Importante: la estadistica principal es una diferencia *emparejada*
    Delta = J_real - media_surr(J)
calculada por realizacion y epsilon. Se informa IC bootstrap de Delta
puntual y simultaneo (max-t) para controlar la exploracion de 71 epsilons.

Dependencias:
    numpy, pandas, matplotlib, mi_libreria

Uso:
    python experimento_J_logisticos_repeticiones_IAAFT.py

Para una prueba de funcionamiento rapida, activar MODO_PRUEBA_RAPIDA=True.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mi_libreria as ml

# ============================ CONFIGURACION =============================
MODO_PRUEBA_RAPIDA = False

R_X = 4.0
R_Y = 3.8
EPS_MIN = 0.0
EPS_MAX = 0.85
N_EPS = 71
N_TRANSITORIO = 12_000
N_DATOS = 20_192
TAU_J = 1
SEMILLA = 2026

N_REPETICIONES = 30     # 20-50 condiciones iniciales diferentes
N_SURROGADOS = 2       # por tipo, por repeticion, por epsilon
                        # se acumulan 30*2=60 surrogados de cada tipo/epsilon
N_BOOTSTRAP = 2000     # remuestreos de las repeticiones, NO de puntos temporales

TOL_IAAFT = 0.1
MAXITER_IAAFT = 1000    # limite finito, para evitar ejecuciones indefinidas

SALIDA = Path('resultados_J_logisticos_robustez')
MOSTRAR_FIGURAS = True

if MODO_PRUEBA_RAPIDA:
    N_EPS = 9
    N_TRANSITORIO = 1000
    N_DATOS = 1024
    N_REPETICIONES = 4
    N_SURROGADOS = 1
    N_BOOTSTRAP = 300
    MAXITER_IAAFT = 100
    SALIDA = Path('resultados_J_logisticos_PRUEBA')


# ============================ UTILIDADES ================================
def indice_j(x, y=None):
    """Adaptador para la firma del usuario, sin modificar mi_libreria."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if y is None:
        valor = ml.indice_J(x, tau=TAU_J)
    else:
        y = np.ascontiguousarray(y, dtype=np.float64)
        valor = ml.indice_J(x, y, tau=TAU_J)
    valor = float(np.asarray(valor).item())
    if not np.isfinite(valor):
        raise ValueError('ml.indice_J devolvio un valor no finito.')
    return valor


def simular_una_repeticion(eps, semilla):
    """Un conductor comun por epsilon, con respuestas inicialmente distintas."""
    rng = np.random.default_rng(semilla)
    ne = len(eps)
    factor = (1.0 - eps) * R_Y
    x = float(rng.uniform(0.1, 0.9))
    y = rng.uniform(0.1, 0.9, size=ne)
    z = rng.uniform(0.1, 0.9, size=ne)   # auxiliar, misma entrada que y

    x_guardado = np.empty(N_DATOS, dtype=np.float64)
    y_guardado = np.empty((ne, N_DATOS), dtype=np.float64)
    suma_log = np.zeros(ne, dtype=np.float64)
    suma_error2 = np.zeros(ne, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    for n in range(N_TRANSITORIO + N_DATOS):
        x_sig = R_X * x * (1.0 - x)
        if n >= N_TRANSITORIO:
            suma_log += np.log(np.maximum(np.abs(factor * (1.0 - 2.0*y)), tiny))
        y_sig = factor * y * (1.0 - y) + eps * x_sig
        z_sig = factor * z * (1.0 - z) + eps * x_sig
        if n >= N_TRANSITORIO:
            k = n - N_TRANSITORIO
            x_guardado[k] = x_sig
            y_guardado[:, k] = y_sig
            suma_error2 += (y_sig - z_sig)**2
        x, y, z = x_sig, y_sig, z_sig

    return x_guardado, y_guardado, suma_log / N_DATOS, np.sqrt(suma_error2 / N_DATOS)


def surrogados_iaaft(y, ns, semilla):
    """Una llamada a ml.iaaft(y, ns): se reutiliza en J_uni y J_bi."""
    # Algunas versiones de iaaft usan np.random global; esto ayuda a repetir
    # sus resultados. Si usa default_rng interno sin semilla, no se garantiza.
    np.random.seed(int(semilla) % (2**32 - 1))
    s = ml.iaaft(
        np.ascontiguousarray(y, dtype=np.float64), ns=ns,
        tol_pc=TOL_IAAFT, verbose=False, maxiter=MAXITER_IAAFT,
        sorttype='quicksort'
    )
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (ns, len(y)):
        raise ValueError(f'ml.iaaft devolvio forma {s.shape}; esperada {(ns, len(y))}.')
    if not np.all(np.isfinite(s)):
        raise ValueError('ml.iaaft devolvio algun NaN o infinito.')
    return s


def estadisticos_bootstrap(matriz, rng, n_boot=N_BOOTSTRAP,
                           simultaneo=False):
    """
    matriz: (repeticiones, epsilons).
    IC sobre la MEDIA entre replicas, con remuestreo de filas completas.
    El intervalo simultaneo max-t utiliza las mismas filas remuestreadas
    en todas las epsilons y controla el efecto de buscar picos en un barrido.
    """
    a = np.asarray(matriz, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError('La entrada al bootstrap debe ser una matriz finita R x E.')
    r, ne = a.shape
    media = np.mean(a, axis=0)
    desv = np.std(a, axis=0, ddof=1) if r > 1 else np.zeros(ne)
    if r < 2:
        return {'media': media, 'std': desv, 'ic_inf': media,
                'ic_sup': media, 'sim_inf': media, 'sim_sup': media}

    ids = rng.integers(0, r, size=(n_boot, r))
    medias_boot = np.mean(a[ids, :], axis=1)  # (B,E)
    ic_inf, ic_sup = np.quantile(medias_boot, [0.025, 0.975], axis=0)
    sim_inf, sim_sup = ic_inf, ic_sup
    if simultaneo:
        # Bootstrap t max. Evita seleccionar umbrales post hoc mirando 71 tests.
        se = np.std(medias_boot, axis=0, ddof=1)
        no_cero = se > 1e-14
        if np.any(no_cero):
            max_t = np.max(
                np.abs((medias_boot[:, no_cero] - media[no_cero]) /
                       se[None, no_cero]), axis=1)
            c = np.quantile(max_t, 0.95)
            sim_inf = media - c*se
            sim_sup = media + c*se
        else:
            sim_inf, sim_sup = media.copy(), media.copy()
    return {'media': media, 'std': desv, 'ic_inf': ic_inf, 'ic_sup': ic_sup,
            'sim_inf': sim_inf, 'sim_sup': sim_sup}


def cruces_cero(eps, valores):
    res = []
    for i in range(len(eps)-1):
        a, b = valores[i:i+2]
        if a == 0:
            res.append(float(eps[i]))
        elif a*b < 0:
            res.append(float(eps[i] - a*(eps[i+1]-eps[i])/(b-a)))
    return res


def plot_linea_ci(ax, eps, d, etiqueta, color, ls='-'):
    ax.plot(eps, d['media'], ls=ls, lw=1.6, color=color, label=etiqueta)
    ax.fill_between(eps, d['ic_inf'], d['ic_sup'], color=color, alpha=.12)


def marcar_umbrales(ax, umbrales):
    for t in umbrales:
        ax.axvline(t, color='0.5', ls=':', lw=1)
    ax.grid(alpha=.20)


def main():
    t0 = time.monotonic()
    SALIDA.mkdir(parents=True, exist_ok=True)
    eps = np.linspace(EPS_MIN, EPS_MAX, N_EPS)
    R, E, S = N_REPETICIONES, len(eps), N_SURROGADOS
    if R < 2 or S < 1 or N_BOOTSTRAP < 100:
        raise ValueError('Usar al menos dos repeticiones, un sustituto y 100 bootstrap.')

    print(f'Configuracion: {R} repeticiones, {E} epsilons, '
          f'{S} surrogados de cada tipo por realizacion.')
    print(f'Se generaran {R*E*S:,} sustitutos IAAFT y otros tantos shuffle.')

    real_y  = np.empty((R, E))
    real_xy = np.empty((R, E))
    real_x  = np.empty(R)
    lambda_c = np.empty((R, E))
    err_aux  = np.empty((R, E))
    error_psd_iaaft = np.empty((R, E))

    # Por cada replica/epsilon, se conserva la MEDIA de los S nulos,
    # que luego se compara de manera emparejada con J_real.
    nulo_y = {t: np.empty((R,E)) for t in ('shuffle','IAAFT')}
    nulo_xy = {t: np.empty((R,E)) for t in ('shuffle','IAAFT')}
    detalle_sur = []
    rng_perm = np.random.default_rng(SEMILLA + 999)

    for r in range(R):
        x, ys, lam, err = simular_una_repeticion(eps, SEMILLA + 101*r)
        lambda_c[r] = lam
        err_aux[r] = err
        real_x[r] = indice_j(x)

        for i, e in enumerate(eps):
            y = ys[i]
            real_y[r,i] = indice_j(y)
            real_xy[r,i] = indice_j(x, y)

            iy_shu = np.empty(S)
            ixy_shu = np.empty(S)
            for k in range(S):
                y_s = rng_perm.permutation(y)
                iy_shu[k] = indice_j(y_s)
                ixy_shu[k] = indice_j(x, y_s)

            # Semilla separada por replica/epsilon, sin correlacionar con shuffle.
            seed_iaaft = SEMILLA + 10_000 + r*E + i
            try:
                iaaft = surrogados_iaaft(y, S, seed_iaaft)
            except Exception as exc:
                raise RuntimeError(
                    f'Error en ml.iaaft para repeticion {r+1}, '
                    f'epsilon={e:.5f}. Revisar firma, convergencia o valores.'
                ) from exc
            iy_ia = np.empty(S)
            ixy_ia = np.empty(S)
            # Error relativo L2 del espectro de potencia, como control
            # independiente del criterio interno de parada de ml.iaaft.
            p_real = np.abs(np.fft.rfft(y-y.mean()))**2
            norma_p = max(np.linalg.norm(p_real), np.finfo(float).tiny)
            psd_rel = np.empty(S)
            for k in range(S):
                iy_ia[k] = indice_j(iaaft[k])
                ixy_ia[k] = indice_j(x, iaaft[k])
                p_sur = np.abs(np.fft.rfft(iaaft[k]-iaaft[k].mean()))**2
                psd_rel[k] = np.linalg.norm(p_sur-p_real)/norma_p
            error_psd_iaaft[r,i] = psd_rel.mean()

            nulo_y['shuffle'][r,i] = iy_shu.mean()
            nulo_xy['shuffle'][r,i] = ixy_shu.mean()
            nulo_y['IAAFT'][r,i] = iy_ia.mean()
            nulo_xy['IAAFT'][r,i] = ixy_ia.mean()

            for k in range(S):
                detalle_sur.append({
                    'repeticion': r+1, 'epsilon': float(e), 'sustituto': k+1,
                    'J_y_shuffle': iy_shu[k], 'J_xy_shuffle': ixy_shu[k],
                    'J_y_IAAFT': iy_ia[k], 'J_xy_IAAFT': ixy_ia[k],
                    'error_rel_PSD_IAAFT': psd_rel[k]
                })

        print(f'Repeticion {r+1:02d}/{R} terminada; '
              f'tiempo transcurrido {(time.monotonic()-t0)/60:.1f} min', flush=True)

    # ------------------ SALVAR DATOS DE CADA REPLICA ------------------
    filas = []
    for r in range(R):
        for i,e in enumerate(eps):
            f = dict(repeticion=r+1, epsilon=float(e),
                     lambda_c=lambda_c[r,i], error_auxiliar_RMS=err_aux[r,i],
                     error_rel_PSD_IAAFT=error_psd_iaaft[r,i],
                     J_x=real_x[r], J_y=real_y[r,i], J_xy=real_xy[r,i])
            for nombre in ('shuffle','IAAFT'):
                f[f'J_y_{nombre}'] = nulo_y[nombre][r,i]
                f[f'J_xy_{nombre}'] = nulo_xy[nombre][r,i]
                f[f'Delta_J_y_{nombre}'] = real_y[r,i]-nulo_y[nombre][r,i]
                f[f'Delta_J_xy_{nombre}'] = real_xy[r,i]-nulo_xy[nombre][r,i]
            filas.append(f)
    pd.DataFrame(filas).to_csv(SALIDA/'resultados_por_repeticion.csv', index=False)
    pd.DataFrame(detalle_sur).to_csv(SALIDA/'valores_individuales_surrogados.csv', index=False)

    # ------------------- ESTIMADORES E INTERVALOS ---------------------
    rng_boot = np.random.default_rng(SEMILLA+34567)
    metricas = {
        'lambda_c': lambda_c, 'error_auxiliar_RMS': err_aux,
        'error_rel_PSD_IAAFT': error_psd_iaaft,
        'J_y': real_y, 'J_xy': real_xy,
    }
    for nombre in ('shuffle','IAAFT'):
        metricas[f'J_y_{nombre}'] = nulo_y[nombre]
        metricas[f'J_xy_{nombre}'] = nulo_xy[nombre]
        metricas[f'Delta_J_y_{nombre}'] = real_y - nulo_y[nombre]
        metricas[f'Delta_J_xy_{nombre}'] = real_xy - nulo_xy[nombre]

    est = {}
    resumen = pd.DataFrame({'epsilon': eps})
    for k,a in metricas.items():
        es_delta = k.startswith('Delta_')
        est[k] = estadisticos_bootstrap(a, rng_boot, simultaneo=es_delta)
        d = est[k]
        for col in ('media','std','ic_inf','ic_sup'):
            resumen[f'{k}_{col}'] = d[col]
        if es_delta:
            resumen[f'{k}_sim_inf'] = d['sim_inf']
            resumen[f'{k}_sim_sup'] = d['sim_sup']
            resumen[f'{k}_excluye_cero_sim'] = (
                (d['sim_inf']>0) | (d['sim_sup']<0)
            )

    umbrales = cruces_cero(eps, est['lambda_c']['media'])
    resumen['estable_condicionalmente_media'] = est['lambda_c']['media'] < 0
    resumen.to_csv(SALIDA/'resumen_bootstrap_por_epsilon.csv', index=False)

    # ------------------- GRAFICO 1: CURVAS DE J -----------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             constrained_layout=True)
    col = {'real':'black', 'shuffle':'tab:orange', 'IAAFT':'tab:blue'}
    plot_linea_ci(axes[0],eps,est['J_y'],'J(y) real',col['real'])
    plot_linea_ci(axes[0],eps,est['J_y_shuffle'],'J(shuffle y)',col['shuffle'])
    plot_linea_ci(axes[0],eps,est['J_y_IAAFT'],'J(IAAFT y)',col['IAAFT'])
    axes[0].axhline(real_x.mean(),ls='--',color='0.65',lw=1,
                    label='Promedio J(x): control')
    axes[0].set_ylabel('J univariante')
    axes[0].legend(fontsize=9)

    plot_linea_ci(axes[1],eps,est['J_xy'],'J(x,y) real',col['real'])
    plot_linea_ci(axes[1],eps,est['J_xy_shuffle'],'J(x, shuffle y)',col['shuffle'])
    plot_linea_ci(axes[1],eps,est['J_xy_IAAFT'],'J(x, IAAFT y)',col['IAAFT'])
    axes[1].set_ylabel('J bivariante')
    axes[1].legend(fontsize=9)

    plot_linea_ci(axes[2],eps,est['lambda_c'],r'$\lambda_c$',col['real'])
    axes[2].axhline(0,color='0.5',lw=1)
    axes[2].set_ylabel(r'Lyapunov condicional $\lambda_c$')
    axes[2].set_xlabel(r'Acoplamiento $\epsilon$')
    axes[2].legend()
    for ax in axes:
        marcar_umbrales(ax, umbrales)
    fig.suptitle('Promedios entre realizaciones e IC bootstrap del 95 %')
    fig.savefig(SALIDA/'J_real_vs_nulos_IC95.png',dpi=180)

    # ---------------- GRAFICO 2: DIFERENCIAS PAREADAS ------------------
    fig2, axs = plt.subplots(2, 1, figsize=(11,8), sharex=True,
                             constrained_layout=True)
    for ax, clave in zip(axs, ('Delta_J_y_', 'Delta_J_xy_')):
        ax.axhline(0,color='black',lw=1)
        for nombre in ('shuffle','IAAFT'):
            d = est[clave+nombre]
            color = col[nombre]
            ax.plot(eps,d['media'],color=color,lw=1.6,
                    label=nombre)
            # Bandas simultaneas en todos los epsilons (mas conservadoras).
            ax.fill_between(eps,d['sim_inf'],d['sim_sup'],color=color,alpha=.16)
            significativo = ((d['sim_inf']>0)|(d['sim_sup']<0))
            if np.any(significativo):
                ax.scatter(eps[significativo],d['media'][significativo],
                           color=color,marker='o',s=17,zorder=3)
        ax.set_ylabel(r'$\Delta J$ univariante' if clave=='Delta_J_y_' else
                      r'$\Delta J$ bivariante')
        ax.legend(title='Media real - nulo',fontsize=9)
        marcar_umbrales(ax,umbrales)
    axs[-1].set_xlabel(r'Acoplamiento $\epsilon$')
    fig2.suptitle('Diferencias emparejadas: bandas bootstrap simultaneas 95 %\n'
                  'Puntos: el intervalo simultaneo excluye cero')
    fig2.savefig(SALIDA/'DeltaJ_nulos_bandas_simultaneas.png',dpi=180)

    # ------------- RECAPITULACION y METADATOS -------------------------
    resumen_txt = [
        f'R={R}; E={E}; NS={S}; N_DATOS={N_DATOS}; tau={TAU_J}',
        f'IAAFT: tol_pc={TOL_IAAFT}, maxiter={MAXITER_IAAFT}',
        f'Bootstrap={N_BOOTSTRAP} replicaciones de filas completas',
        f'Error relativo PSD IAAFT (promedio): {error_psd_iaaft.mean():.6g}',
        f'Error relativo PSD IAAFT (maximo de medias por replica/eps): {error_psd_iaaft.max():.6g}',
        'Cruces de lambda medio: '+(', '.join(f'{v:.6f}' for v in umbrales)
                                     if umbrales else 'Ninguno'),
        '',
        'Significado de los nulos:',
        '* Univariante: J(y) vs J(shuffle(y)) y J(IAAFT(y)).',
        '* Bivariante: J(x,y) vs J(x,shuffle(y)) y J(x,IAAFT(y)).',
        '* Las diferencias se forman dentro de cada realizacion y epsilon.',
        '* Las bandas simultaneas 95% son de la media de Delta en todos los eps.',
        '* El criterio primario de sincronizacion es lambda_c<0; el error auxiliar',
        '  puede colapsar numericamente a cero aun con lambda_c>0.',
        '* Se deben revisar regiones persistentes y reproducibilidad, no solo',
        '  puntos aislados de significacion en el barrido.',
    ]
    for k in ('Delta_J_y_shuffle','Delta_J_y_IAAFT',
              'Delta_J_xy_shuffle','Delta_J_xy_IAAFT'):
        d = est[k]
        n_sig = int(np.sum((d['sim_inf']>0) | (d['sim_sup']<0)))
        resumen_txt.append(f'{k}: {n_sig}/{E} puntos con IC simultaneo sin cero')
    (SALIDA/'LEEME_resultados.txt').write_text('\n'.join(resumen_txt)+'\n', encoding='utf-8')

    print('\n'.join(resumen_txt))
    print(f'\nArchivos guardados en: {SALIDA.resolve()}')
    print(f'Duracion total: {(time.monotonic()-t0)/60:.1f} min')
    if MOSTRAR_FIGURAS:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
