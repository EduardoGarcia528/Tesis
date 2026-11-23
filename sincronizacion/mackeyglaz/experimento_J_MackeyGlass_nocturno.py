# -*- coding: utf-8 -*-
"""Mackey-Glass: indice J y sincronizacion generalizada — experimento nocturno.

Modelo: el mismo del experimento anterior. Conductor de retardo 17,
respuesta de retardo 23 y replica auxiliar. Se calcula el mayor exponente
condicional integrando TODA la historia de la ecuacion variacional con retardo.

Fijos por decision experimental:
    DT = 0.2; MUESTREO_CADA = 20 (intervalo muestreado = 4 unidades).

Mejoras: muestreo concentrado cerca del umbral y en acoplamiento fuerte,
mas realizaciones y datos, nulos shuffle/IAAFT/desplazamiento circular,
semillas independientes reproducibles, metrica PSD para IAAFT, guardado
atomico por acoplamiento en SQLite y reanudacion automatica, bootstrap
emparejado con bandas simultaneas para 5 contrastes x todos los epsilons.

Uso:
    python experimento_J_MackeyGlass_nocturno.py --perfil prueba
    python experimento_J_MackeyGlass_nocturno.py --perfil noche
    python experimento_J_MackeyGlass_nocturno.py --perfil profundo
    python experimento_J_MackeyGlass_nocturno.py --perfil noche --solo-resumen

El script NO usa aproximaciones ficticias para J ni IAAFT: requiere
mi_libreria con indice_J(x, seriey=None, tau=1) e iaaft(x, ns, ...).
"""
from __future__ import annotations

import argparse
import json
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import mi_libreria as ml

# ======================== CONFIGURACION ========================
SEED = 20261121
BETA_X, GAMMA_X, N_X, RETARDO_X = 0.20, 0.10, 10, 17.0
BETA_Y, GAMMA_Y, N_Y, RETARDO_Y = 0.20, 0.10, 10, 23.0

DT = 0.2
MUESTREO_CADA = 20
RENORM_CADA = 10       # renormalizacion de historia tangente cada 2 unidades
TAU_J = 1
TOL_IAAFT = 0.1
MOSTRAR_FIGURAS = False

# CONFIGURACION DE LOS PERFILES: editable antes de iniciar un experimento.
PERFILES = {
    'prueba': dict(n_repeticiones=3, n_datos=512, n_transitorio=400.0,
                   n_shuffle=1, n_iaaft=1, n_shift=2,
                   maxiter_iaaft=70, n_bootstrap=500),
    'noche': dict(n_repeticiones=48, n_datos=40_384, n_transitorio=8_000.0,
                  n_shuffle=4, n_iaaft=4, n_shift=8,
                  maxiter_iaaft=2_000, n_bootstrap=10_000),
    'profundo': dict(n_repeticiones=60, n_datos=65_536, n_transitorio=10_000.0,
                     n_shuffle=6, n_iaaft=6, n_shift=12,
                     maxiter_iaaft=3_000, n_bootstrap=15_000),
}

# PSD relativa superior a este valor se informa como advertencia.
# No filtrar arbitrariamente realizaciones: pueden crear sesgo.
PSD_ADVERTENCIA = 0.001        # error L2 relativo (0.1 %)


def rejilla_epsilon(perfil):
    if perfil == 'prueba':
        return np.array([0.0, 0.05, 0.06, 0.066, 0.074, 0.10, 0.12])
    general = np.round(np.arange(0.0, 0.12000001, 0.004), 6)
    transicion = np.round(np.arange(0.050, 0.0800001, 0.001), 6)
    fuerte = np.round(np.arange(0.090, 0.1200001, 0.002), 6)
    return np.unique(np.concatenate([general, transicion, fuerte]))


# ============================ DDE / NUMBA ============================
@njit(cache=True)
def _f_mg(u, beta, n):
    return beta * u / (1.0 + u ** n)


@njit(cache=True)
def _df_mg(u, beta, n):
    un = u ** n
    return beta * (1.0 + (1.0 - n) * un) / ((1.0 + un) ** 2)


@njit(cache=True)
def integrar_mg_rk4(eps, hx, hy, hz, hv,
                    h, dx, dy, nsteps_trans, ndata, stride, renorm,
                    beta_x, gamma_x, n_x, beta_y, gamma_y, n_y):
    """
    Simula TODOS los acoplamientos con el mismo conductor por realizacion.
    hx: (L,), hy/hz/hv: (E,L), historia antigua -> reciente.
    Retardo efectivo: dx*h y dy*h; ambos son enteros.
    Para el variacional: v' = df_mg(y_tau)*v_tau - (gamma_y+eps)*v.
    Retorna (x[t], y[e,t], cond_Lyap[e], RMS_aux[e], RMS_xy[e]).
    """
    ne = len(eps)
    L = hx.size
    ptr = L - 1                         # indice con valor en t=0
    xout = np.empty(ndata)
    yout = np.empty((ne, ndata))
    err_aux_sum = np.zeros(ne)
    err_xy_sum = np.zeros(ne)
    ly_sum = np.zeros(ne)
    ly_count = 0
    nsteps = nsteps_trans + ndata*stride

    for tstep in range(nsteps):
        d0x = (ptr-dx) % L
        d1x = (ptr-dx+1) % L
        d0y = (ptr-dy) % L
        d1y = (ptr-dy+1) % L

        # RK4 para la historia conductora, sin acoplamiento.
        x0 = hx[ptr]
        fx0 = _f_mg(hx[d0x], beta_x, n_x)
        fxh = _f_mg((hx[d0x]+hx[d1x])*0.5, beta_x, n_x)
        fx1 = _f_mg(hx[d1x], beta_x, n_x)
        k1x = fx0 - gamma_x*x0
        x2 = x0 + 0.5*h*k1x
        k2x = fxh - gamma_x*x2
        x3 = x0 + 0.5*h*k2x
        k3x = fxh - gamma_x*x3
        x4 = x0 + h*k3x
        k4x = fx1 - gamma_x*x4
        xn = x0 + (h/6.0)*(k1x+2*k2x+2*k3x+k4x)

        newptr = (ptr+1) % L
        hx[newptr] = xn

        for e in range(ne):
            ep = eps[e]
            y0, z0, v0 = hy[e,ptr], hz[e,ptr], hv[e,ptr]
            yd0, yd1 = hy[e,d0y], hy[e,d1y]
            zd0, zd1 = hz[e,d0y], hz[e,d1y]
            vd0, vd1 = hv[e,d0y], hv[e,d1y]

            fy0 = _f_mg(yd0, beta_y, n_y)
            fyh = _f_mg(0.5*(yd0+yd1), beta_y, n_y)
            fy1 = _f_mg(yd1, beta_y, n_y)
            fz0 = _f_mg(zd0, beta_y, n_y)
            fzh = _f_mg(0.5*(zd0+zd1), beta_y, n_y)
            fz1 = _f_mg(zd1, beta_y, n_y)
            ay0 = _df_mg(yd0, beta_y, n_y)
            ayh = _df_mg(0.5*(yd0+yd1), beta_y, n_y)
            ay1 = _df_mg(yd1, beta_y, n_y)
            a = gamma_y+ep

            k1y = fy0-a*y0+ep*x0
            k1z = fz0-a*z0+ep*x0
            k1v = ay0*vd0-a*v0

            y2, z2, v2 = y0+0.5*h*k1y, z0+0.5*h*k1z, v0+0.5*h*k1v
            k2y = fyh-a*y2+ep*x2
            k2z = fzh-a*z2+ep*x2
            k2v = ayh*0.5*(vd0+vd1)-a*v2

            y3, z3, v3 = y0+0.5*h*k2y, z0+0.5*h*k2z, v0+0.5*h*k2v
            k3y = fyh-a*y3+ep*x3
            k3z = fzh-a*z3+ep*x3
            k3v = ayh*0.5*(vd0+vd1)-a*v3

            y4, z4, v4 = y0+h*k3y, z0+h*k3z, v0+h*k3v
            k4y = fy1-a*y4+ep*x4
            k4z = fz1-a*z4+ep*x4
            k4v = ay1*vd1-a*v4

            yn = y0+(h/6.)*(k1y+2*k2y+2*k3y+k4y)
            zn = z0+(h/6.)*(k1z+2*k2z+2*k3z+k4z)
            vn = v0+(h/6.)*(k1v+2*k2v+2*k3v+k4v)
            hy[e,newptr] = yn
            hz[e,newptr] = zn
            hv[e,newptr] = vn

        ptr = newptr
        # Evaluar la trayectoria solo en los instantes de muestreo, post-burn.
        if tstep >= nsteps_trans and ((tstep-nsteps_trans+1) % stride == 0):
            k = (tstep - nsteps_trans)//stride
            xout[k] = xn
            for e in range(ne):
                yval = hy[e,ptr]
                yout[e,k] = yval
                err_aux_sum[e] += (yval-hz[e,ptr])**2
                err_xy_sum[e] += (yval-xn)**2

        # La perturbacion incluye TODA su historia, no solo v(t).
        if ((tstep+1) % renorm == 0):
            for e in range(ne):
                accum = 0.0
                for j in range(L):
                    accum += hv[e,j]*hv[e,j]
                norm = np.sqrt(accum/L)
                if not (norm > 0 and np.isfinite(norm)):
                    raise ValueError('Historia tangente degenerada. Reducir DT.')
                if tstep+1 > nsteps_trans:
                    ly_sum[e] += np.log(norm)
                for j in range(L):
                    hv[e,j] /= norm
            if tstep+1 > nsteps_trans:
                ly_count += 1

    if ly_count == 0:
        raise ValueError('Transitorio y datos insuficientes para medir Lyapunov.')
    lam = ly_sum/(ly_count*renorm*h)  # 1 / unidad de tiempo
    return xout, yout, lam, np.sqrt(err_aux_sum/ndata), np.sqrt(err_xy_sum/ndata)


def condiciones_iniciales(eps, rng):
    """Historias suaves, diferentes entre conductor, respuesta y auxiliar."""
    ne = len(eps)
    dx = int(round(RETARDO_X/DT))
    dy = int(round(RETARDO_Y/DT))
    if dx < 2 or dy < 2 or abs(dx*DT-RETARDO_X)>1e-9 or abs(dy*DT-RETARDO_Y)>1e-9:
        raise ValueError('Los retardos deben ser multiplos exactos de DT y >= 2*DT.')
    L = max(dx,dy)+2
    t = DT*np.arange(-(L-1),1, dtype=float)
    hx = 1.10 + 0.035*np.sin(0.3*t+rng.uniform(0,6.28))
    hx += 0.015*np.cos(0.11*t+rng.uniform(0,6.28))
    hy = np.empty((ne,L),dtype=float)
    hz = np.empty((ne,L),dtype=float)
    hv = np.empty((ne,L),dtype=float)
    for e in range(ne):
        hy[e] = (1.04+0.07*np.sin(0.28*t+rng.uniform(0,6.28))+
                 0.02*np.cos(0.17*t+rng.uniform(0,6.28)))
        hz[e] = (1.04+0.07*np.sin(0.28*t+rng.uniform(0,6.28))+
                 0.02*np.cos(0.17*t+rng.uniform(0,6.28)))
        hv[e] = 1.0+0.07*np.sin(0.23*t + rng.uniform(0,6.28))
    return hx,hy,hz,hv,dx,dy


def simular_repeticion(eps, seed, n_trans, ndata):
    rng = np.random.default_rng(seed)
    hx,hy,hz,hv,dx,dy = condiciones_iniciales(eps, rng)
    trans_steps = int(round(n_trans/DT))
    if abs(trans_steps*DT-n_trans)>1e-8:
        raise ValueError('N_TRANSITORIO/DT debe ser un entero.')
    if trans_steps % RENORM_CADA != 0:
        raise ValueError('N_TRANSITORIO/DT debe ser divisible por RENORM_CADA.')
    return integrar_mg_rk4(eps,hx,hy,hz,hv,DT,dx,dy,trans_steps,ndata,
                          MUESTREO_CADA,RENORM_CADA,
                          BETA_X,GAMMA_X,N_X,BETA_Y,GAMMA_Y,N_Y)



# ====================== J, SURROGADOS Y METRICAS ======================

def j_value(x, y=None):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if y is None:
        j = ml.indice_J(x, tau=TAU_J)
    else:
        j = ml.indice_J(x, np.ascontiguousarray(y, dtype=np.float64), tau=TAU_J)
    j = float(np.asarray(j).item())
    if not np.isfinite(j):
        raise ValueError('ml.indice_J devolvio un escalar no finito.')
    return j


def create_iaaft(y, n, seed, maxiter):
    # Algunas implementaciones utilizan np.random global: el estado
    # se limita al proceso actual, que es secuencial.
    np.random.seed(int(seed) % (2**32 - 1))
    out = ml.iaaft(np.ascontiguousarray(y, dtype=float), ns=n,
                   tol_pc=TOL_IAAFT, verbose=False,
                   maxiter=maxiter, sorttype='quicksort')
    out = np.asarray(out, dtype=np.float64)
    if out.shape != (n, len(y)) or not np.all(np.isfinite(out)):
        raise ValueError(f'ml.iaaft devolvio forma invalida {out.shape}.')
    return out


def psd_error(y, sy):
    # rFFT sin frecuencia cero para enfocarse en variaciones temporales.
    a = np.abs(np.fft.rfft(y - np.mean(y)))**2
    b = np.abs(np.fft.rfft(sy - np.mean(sy)))**2
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.finfo(float).tiny))


def semilla_par(rep, pos_eps, tipo):
    # No depende del orden de ejecucion ni de la reanudacion.
    v = (SEED + 1_000_003 * (rep+1) + 65_537 * (pos_eps+1)
         + 977 * tipo)
    return int(v % (2**32-1))


def evaluar_punto(x, y, rep, pos_eps, conf):
    """Calcula un solo par (replica,epsilon) con nulos emparejados."""
    r = {}
    r['J_y'] = j_value(y)
    r['J_xy'] = j_value(x, y)
    individuales = []

    rng = np.random.default_rng(semilla_par(rep, pos_eps, 11))
    js, jx = [], []
    for n in range(conf['n_shuffle']):
        s = rng.permutation(y)
        v1, v2 = j_value(s), j_value(x, s)
        js.append(v1); jx.append(v2)
        individuales.append(dict(tipo='shuffle', numero=n+1,
                                  J_y=v1, J_xy=v2, error_rel_PSD=None,
                                  desplazamiento=None))
    r['J_y_shuffle'] = float(np.mean(js))
    r['J_xy_shuffle'] = float(np.mean(jx))

    sia = create_iaaft(y, conf['n_iaaft'],
                       semilla_par(rep,pos_eps,23), conf['maxiter_iaaft'])
    js, jx, psds = [], [], []
    for n, s in enumerate(sia):
        v1, v2 = j_value(s), j_value(x,s)
        p = psd_error(y,s)
        js.append(v1); jx.append(v2); psds.append(p)
        individuales.append(dict(tipo='IAAFT', numero=n+1,
                                  J_y=v1, J_xy=v2, error_rel_PSD=p,
                                  desplazamiento=None))
    r['J_y_IAAFT'] = float(np.mean(js))
    r['J_xy_IAAFT'] = float(np.mean(jx))
    r['error_rel_PSD_IAAFT'] = float(np.mean(psds))
    r['error_rel_PSD_IAAFT_max'] = float(np.max(psds))
    r['frac_IAAFT_PSD_mala'] = float(np.mean(np.asarray(psds)>PSD_ADVERTENCIA))

    # Nulo bivariante: mantiene exactamente la serie y y todo su PSD,
    # pero rompe la alineacion temporal entre conductor y respuesta.
    # Se excluyen |desplazamiento|<N/8 modulo N para evitar retrasos cortos.
    N = len(y)
    minshift = max(2, N//8)
    if N-2*minshift<2:
        raise ValueError('Serie demasiado corta para desplazamientos circulares.')
    rng = np.random.default_rng(semilla_par(rep, pos_eps, 31))
    shifts = rng.integers(minshift, N-minshift, size=conf['n_shift'])
    jshift = []
    for n, shift in enumerate(shifts):
        v = j_value(x, np.roll(y,int(shift)))
        jshift.append(v)
        individuales.append(dict(tipo='shift', numero=n+1,
                                  J_y=None, J_xy=v, error_rel_PSD=0.0,
                                  desplazamiento=int(shift)))
    r['J_xy_shift'] = float(np.mean(jshift))

    for name in ('shuffle', 'IAAFT'):
        r['Delta_J_y_'+name] = r['J_y'] - r['J_y_'+name]
        r['Delta_J_xy_'+name] = r['J_xy'] - r['J_xy_'+name]
    r['Delta_J_xy_shift'] = r['J_xy'] - r['J_xy_shift']
    return r, individuales


# ========================= GUARDADO Y REANUDACION =========================

def abrir_base(salida, configuracion, eps):
    salida.mkdir(parents=True, exist_ok=True)
    metadata = salida/'configuracion.json'
    conf_all = dict(configuracion, epsilon_grid=[float(v) for v in eps],
                    modelo='MackeyGlass_retardos17_23', dt=DT,
                    muestreo_cada=MUESTREO_CADA, tau_j=TAU_J,
                    semilla=SEED, tol_iaaft=TOL_IAAFT)
    serialized = json.dumps(conf_all, indent=2, sort_keys=True)
    if metadata.exists():
        existing = json.loads(metadata.read_text(encoding='utf8'))
        if existing != conf_all:
            raise RuntimeError('La configuracion anterior es distinta. '
                               'Elige otra carpeta --salida; no mezclar ejecuciones.')
    else:
        metadata.write_text(serialized,encoding='utf8')
    db = sqlite3.connect(str(salida/'checkpoint.sqlite'))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('''CREATE TABLE IF NOT EXISTS puntos (
        repeticion INTEGER NOT NULL, i_epsilon INTEGER NOT NULL,
        epsilon REAL NOT NULL, resumen TEXT NOT NULL,
        surrogados TEXT NOT NULL,
        PRIMARY KEY (repeticion, i_epsilon))''')
    db.commit()
    return db


def leidos(db):
    return {(int(r),int(i)) for r,i in db.execute(
        'SELECT repeticion,i_epsilon FROM puntos')}


def guardar_punto(db,rep,i,eps,resumen,individuales):
    # Commit atomico de todos los datos de un punto.
    with db:
        db.execute('''INSERT OR REPLACE INTO puntos
                      (repeticion,i_epsilon,epsilon,resumen,surrogados)
                      VALUES (?,?,?,?,?)''',
                   (rep+1, i, float(eps),json.dumps(resumen,allow_nan=False),
                    json.dumps(individuales,allow_nan=False)))


def exportar_csv(db, salida):
    rows=[]; individuales=[]
    for rep,i,eps,info,detalles in db.execute(
        'SELECT repeticion,i_epsilon,epsilon,resumen,surrogados '
        'FROM puntos ORDER BY repeticion,i_epsilon'):
        d=json.loads(info)
        rows.append(dict(repeticion=int(rep),epsilon=float(eps), **d))
        for entry in json.loads(detalles):
            individuales.append(dict(repeticion=int(rep),epsilon=float(eps),**entry))
    if rows:
        tmp=salida/'resultados_por_repeticion.tmp'
        pd.DataFrame(rows).to_csv(tmp,index=False)
        tmp.replace(salida/'resultados_por_repeticion.csv')
    if individuales:
        tmp=salida/'valores_individuales_surrogados.tmp'
        pd.DataFrame(individuales).to_csv(tmp,index=False)
        tmp.replace(salida/'valores_individuales_surrogados.csv')
    return pd.DataFrame(rows)


# ============================ BOOTSTRAP ============================

CONTRASTES = ['Delta_J_y_shuffle', 'Delta_J_y_IAAFT',
              'Delta_J_xy_shuffle', 'Delta_J_xy_IAAFT',
              'Delta_J_xy_shift']


def resumen_bootstrap_completo(df, eps, conf):
    """Solo usa replicas completas, emparejadas en TODA la rejilla."""
    E=len(eps)
    reps_buenas = df.groupby('repeticion')['epsilon'].nunique()
    ids = reps_buenas[reps_buenas==E].index.to_numpy(dtype=int)
    if len(ids)<3:
        print('No hay suficientes realizaciones completas para el bootstrap.')
        return None, None, None
    df=df[df['repeticion'].isin(ids)].sort_values(['repeticion','epsilon'])
    R=len(ids); B=conf['n_bootstrap']
    rng=np.random.default_rng(SEED+99931)
    indices=rng.integers(0,R,size=(B,R))
    keys=['lambda_c','error_auxiliar_RMS','error_xy_RMS',
          'J_x','J_y','J_xy','J_y_shuffle','J_xy_shuffle',
          'J_y_IAAFT','J_xy_IAAFT','J_xy_shift',
          'error_rel_PSD_IAAFT','error_rel_PSD_IAAFT_max',
          'frac_IAAFT_PSD_mala'] + CONTRASTES
    arrays={k: df[k].to_numpy(dtype=float).reshape(R,E) for k in keys}
    result=pd.DataFrame({'epsilon':eps})
    stats={}
    boots_d=[]
    for name,a in arrays.items():
        boot=np.empty((B,E),dtype=float)
        for start in range(0,B,256):
            stop=min(start+256,B)
            boot[start:stop]=a[indices[start:stop]].mean(axis=1)
        vals=dict(media=a.mean(axis=0), std=a.std(axis=0,ddof=1),
                  ic_inf=np.quantile(boot,0.025,axis=0),
                  ic_sup=np.quantile(boot,0.975,axis=0))
        stats[name]=vals
        for tag,v in vals.items():
            result[f'{name}_{tag}']=v
        if name in CONTRASTES:
            boots_d.append((name,boot))

    # Multiplicador max-t simultaneo para 5 contrastes x todos los epsilons.
    max_t=np.zeros(B,dtype=float)
    ses={}
    for name,boot in boots_d:
        m=stats[name]['media']
        se=boot.std(axis=0,ddof=1)
        ses[name]=se
        valid=se>1e-14
        if np.any(valid):
            max_t=np.maximum(max_t,np.max(np.abs((boot[:,valid]-m[valid])/se[valid]),axis=1))
    q95=float(np.quantile(max_t,.95))
    for name in CONTRASTES:
        m=stats[name]['media']
        se=ses[name]
        lo=m-q95*se; hi=m+q95*se
        stats[name]['sim_inf']=lo
        stats[name]['sim_sup']=hi
        result[f'{name}_sim_inf']=lo
        result[f'{name}_sim_sup']=hi
        result[f'{name}_excluye_cero_sim']=(lo>0)|(hi<0)
    result['estable_condicionalmente_media']=stats['lambda_c']['media']<0
    result['n_repeticiones_completas']=R
    return result, stats, q95


def crossings(eps, values):
    cs=[]
    for i in range(len(eps)-1):
        x,y=float(values[i]),float(values[i+1])
        if x*y<0:
            cs.append(eps[i]-x*(eps[i+1]-eps[i])/(y-x))
    return cs


def guardar_graficas(eps, stats, out, umbrales):
    fig, ax=plt.subplots(4,1,figsize=(11,13),sharex=True,constrained_layout=True)
    for k,name in enumerate(('J_y','J_xy')):
        for field,color,label in [(name,'black','Real'),
                                  (name+'_shuffle','tab:orange','Shuffle'),
                                  (name+'_IAAFT','tab:blue','IAAFT')]:
            s=stats[field]
            ax[k].plot(eps,s['media'],color=color,lw=1.4,label=label)
            ax[k].fill_between(eps,s['ic_inf'],s['ic_sup'],color=color,alpha=.12)
        if name=='J_xy':
            s=stats['J_xy_shift']
            ax[k].plot(eps,s['media'],color='tab:green',lw=1.3,label='Shift')
            ax[k].fill_between(eps,s['ic_inf'],s['ic_sup'],color='tab:green',alpha=.12)
        ax[k].set_ylabel(name)
        ax[k].legend(ncol=4,fontsize=9)
    s=stats['lambda_c']
    ax[2].plot(eps,s['media'],color='black',label='Lambda condicional')
    ax[2].fill_between(eps,s['ic_inf'],s['ic_sup'],color='black',alpha=.12)
    ax[2].axhline(0,color='tab:red',lw=1)
    ax[2].set_ylabel('Lyapunov condicional / tiempo')
    ax[3].semilogy(eps,np.maximum(stats['error_auxiliar_RMS']['media'],1e-15),
                   label='RMS(y-z)')
    ax[3].semilogy(eps,np.maximum(stats['error_xy_RMS']['media'],1e-15),
                   label='RMS(x-y)')
    ax[3].set_ylabel('RMS'); ax[3].set_xlabel('Acoplamiento epsilon'); ax[3].legend()
    for a in ax:
        a.grid(alpha=.18)
        for c in umbrales: a.axvline(c,color='.5',ls=':',lw=1.2)
    fig.suptitle('Mackey-Glass: J, nulos, estabilidad (IC puntuales del 95%)')
    fig.savefig(out/'J_real_vs_nulos_IC95.png',dpi=170)
    plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True,constrained_layout=True)
    group=[CONTRASTES[:2],CONTRASTES[2:]]
    colors=['tab:orange','tab:blue','tab:green']
    for i, names in enumerate(group):
        a=axs[i]; a.axhline(0,lw=1,color='.5')
        for name,color in zip(names,colors):
            s=stats[name]
            a.plot(eps,s['media'],lw=1.6,color=color,label=name)
            a.fill_between(eps,s['sim_inf'],s['sim_sup'],color=color,alpha=.14)
            sig=(s['sim_inf']>0)|(s['sim_sup']<0)
            a.scatter(eps[sig],s['media'][sig],color=color,s=15)
        a.set_ylabel('Delta J'); a.legend(fontsize=8)
        a.grid(alpha=.18)
        for c in umbrales: a.axvline(c,color='.5',ls=':',lw=1.2)
    axs[-1].set_xlabel('Acoplamiento epsilon')
    fig.suptitle('Delta J: bandas simultaneas 95%, correccion conjunta 5 x E')
    fig.savefig(out/'DeltaJ_nulos_bandas_simultaneas.png',dpi=170)
    plt.close(fig)


def resumen_desde_db(db,out,eps,conf,final=False):
    df=exportar_csv(db,out)
    if df.empty:
        return
    s,stats,q=resumen_bootstrap_completo(df,eps,conf)
    if s is None:
        return
    tmp=out/'resumen_bootstrap_por_epsilon.tmp'
    s.to_csv(tmp,index=False)
    tmp.replace(out/'resumen_bootstrap_por_epsilon.csv')
    cs=crossings(eps,stats['lambda_c']['media'])
    guardar_graficas(eps,stats,out,cs)
    lines=[f'Analisis de {int(s.n_repeticiones_completas.iloc[0])} repeticiones completas',
           f'DT={DT}; MUESTREO_CADA={MUESTREO_CADA}; Delta t analizado={DT*MUESTREO_CADA}',
           f'N_DATOS={conf["n_datos"]}; epsilons={len(eps)}',
           f'Intervalos simultaneos: 5 contrastes x {len(eps)} epsilons',
           f'Bootstrap={conf["n_bootstrap"]}; q95(max-t)={q:.6g}',
           'Umbrales lambda=0 (interpolacion): '+(', '.join(f'{v:.6f}' for v in cs) if cs else 'ninguno'),
           'Error PSD IAAFT (promedio de las medias)='+str(df['error_rel_PSD_IAAFT'].mean()),
           f'Fraccion IAAFT con error PSD > {PSD_ADVERTENCIA:g}: '+str(df['frac_IAAFT_PSD_mala'].mean())]
    for name in CONTRASTES:
        total=int(s[f'{name}_excluye_cero_sim'].sum())
        lines.append(f'{name}: {total}/{len(eps)} diferencias vs cero (simultaneas)')
    (out/'LEEME_resultados.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'+'\n'.join(lines),flush=True)


def main():
    ap=argparse.ArgumentParser(description='Experimento nocturno Mackey-Glass y J')
    ap.add_argument('--perfil',choices=PERFILES,default='noche')
    ap.add_argument('--salida',default=None,help='Directorio de resultados; resume automaticamente')
    ap.add_argument('--solo-resumen',action='store_true',help='Exportar parciales y reanalizar')
    args=ap.parse_args()
    conf=PERFILES[args.perfil].copy()
    eps=rejilla_epsilon(args.perfil)
    out=Path(args.salida or 'resultados_MackeyGlass_'+args.perfil)
    db=abrir_base(out, dict(perfil=args.perfil, **conf), eps)
    t0=time.monotonic()
    R=conf['n_repeticiones']; E=len(eps)
    hechas=leidos(db)
    print(f'Perfil {args.perfil}: {R} repeticiones x {E} acoplamientos',flush=True)
    print(f'N={conf["n_datos"]}, DT={DT}, MUESTREO_CADA={MUESTREO_CADA}, '
          f'Delta t={DT*MUESTREO_CADA}; listos: {len(hechas)}/{R*E}',flush=True)
    print(f'IAAFT = {conf["n_iaaft"]}, shuffle = {conf["n_shuffle"]}, '
          f'shift = {conf["n_shift"]} por par (rep,epsilon)',flush=True)
    if args.solo_resumen:
        resumen_desde_db(db,out,eps,conf,final=False)
        db.close()
        return
    try:
        for r in range(R):
            faltan=[i for i in range(E) if (r+1,i) not in hechas]
            if not faltan:
                print(f'Repeticion {r+1}/{R} ya guardada; omitida.',flush=True)
                continue
            x,ys,lam,aux,exy=simular_repeticion(eps,SEED+101*r,
                                                conf['n_transitorio'], conf['n_datos'])
            if not np.all(np.isfinite(ys)) or not np.all(np.isfinite(lam)):
                raise RuntimeError(f'DDE no finita en repeticion {r+1}.')
            jx=j_value(x)
            for i in faltan:
                e=float(eps[i]); y=ys[i]
                vals,ind=evaluar_punto(x,y,r,i,conf)
                base=dict(lambda_c=float(lam[i]),error_auxiliar_RMS=float(aux[i]),
                          error_xy_RMS=float(exy[i]),J_x=jx)
                base.update(vals)
                guardar_punto(db,r,i,e,base,ind)
                hechas.add((r+1,i))
                print(f'  Rep {r+1:02d}/{R}, epsilon={e:.3f}: '
                      f'{len(hechas)}/{R*E} puntos; '
                      f'{(time.monotonic()-t0)/60:.1f} min transcurridos',
                      flush=True)
            del ys
            exportar_csv(db,out) # CSV parciales al terminar cada realizacion
            print(f'Repeticion {r+1}/{R} guardada en SQLite y CSV.',flush=True)
    finally:
        # Si falla o el usuario detiene Ctrl+C, conserva los puntos ya terminados.
        exportar_csv(db,out)
        db.close()

    db=sqlite3.connect(str(out/'checkpoint.sqlite'))
    try:
        resumen_desde_db(db,out,eps,conf,final=True)
    finally:
        db.close()
    print(f'Experimento finalizado. Archivos en {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
