# -*- coding: utf-8 -*-
"""
SINCRONIZACION GENERALIZADA MACKKEY-GLASS + INDICE J
=================================================
Conductor x (autonomo):
    dx/dt = beta_x * x(t-tau_x)/(1+x(t-tau_x)^n_x) - gamma_x*x
Respuesta y, forzada unidireccionalmente:
    dy/dt = beta_y * y(t-tau_y)/(1+y(t-tau_y)^n_y) - gamma_y*y + eps*(x-y)
Sistema auxiliar z: misma ecuacion de y y mismo conductor x, historia distinta.

La sincronizacion generalizada se identifica con:
  1. lambda_c < 0: exponente condicional MAXIMO de la respuesta.
     Se integra la ecuacion variacional CON RETARDO (no solo derivada local)
     y se renormaliza TODA la historia de perturbaciones periodicamente.
  2. RMS(y-z) pequeno: replica auxiliar. Una igualdad exacta a precision
     de maquina no es por si sola una prueba suficiente de estabilidad.

Indices:
    J_x=ml.indice_J(x, tau=TAU_J)
    J_y=ml.indice_J(y, tau=TAU_J)
    J_xy=ml.indice_J(x, y, tau=TAU_J)

Nulos emparejados en cada epsilon y repeticion:
    J(shuffle(y)), J(IAAFT(y)), J(x,shuffle(y)), J(x,IAAFT(y)).
    x permanece fijo en los contrastes bivariantes.

Se remuestrean las realizaciones ENTERAS. Los intervalos simultaneos para
Delta J estan ajustados EN CONJUNTO por 4 contrastes x TODOS los epsilons.

Integracion: RK4 del sistema de ecuaciones con retardo, metodo de pasos
con dt fijo; valores retardados a mitad del paso por interpolacion lineal.
Los dos retardos deben ser multiples enteros de DT y mayores que DT.

Dependencias: numpy, pandas, matplotlib, numba, mi_libreria.
Uso:
    python experimento_J_MackeyGlass_generalizada_robustez.py
    python experimento_J_MackeyGlass_generalizada_robustez.py --quick

ATENCION: N=20192 y 30 replicas con IAAFT puede exigir muchas horas
segun el equipo y el coste de ml.indice_J/ml.iaaft.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import mi_libreria as ml

# ======================== CONFIGURACION ========================
SEED = 2026
BETA_X, GAMMA_X, N_X, RETARDO_X = 0.20, 0.10, 10, 17.0
BETA_Y, GAMMA_Y, N_Y, RETARDO_Y = 0.20, 0.10, 10, 23.0
# Desajuste de los retardos => las dos ecuaciones NO son identicas.

EPS_MIN, EPS_MAX, N_EPS = 0.0, 0.12, 62
N_REPETICIONES = 48
N_TRANSITORIO = 8_000.0              # unidades de tiempo, NO observaciones
N_DATOS = 40_384                     # datos muestreados para J
DT = 0.2                            # paso del integrador RK4
MUESTREO_CADA = 20                    # intervalo de datos = DT*5 = 0.5
RENORM_CADA = 10                     # renormaliza historia tangente cada 1.0

TAU_J = 1
N_SURROGADOS = 6                     # 2 shuffle + 2 IAAFT por realizacion/eps
TOL_IAAFT = 0.1
MAXITER_IAAFT = 1_000               # acotado por seguridad; configurable
N_BOOTSTRAP = 10_000
MOSTRAR_FIGURAS = False
SALIDA = Path('resultados_J_MackeyGlass_robustez')


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


# ============================ J Y NULOS ============================
def calcular_j(x,y=None):
    x = np.ascontiguousarray(x, dtype=float)
    if y is None:
        v = ml.indice_J(x, tau=TAU_J)
    else:
        v = ml.indice_J(x, np.ascontiguousarray(y,dtype=float), tau=TAU_J)
    v = float(np.asarray(v).item())
    if not np.isfinite(v):
        raise ValueError('ml.indice_J devolvio un valor no finito.')
    return v


def crear_iaaft(y, ns, seed, maxiter):
    # En implementaciones basadas en np.random global permite fijar semilla.
    # Si ml.iaaft usa default_rng propio, la reproducibilidad puede variar.
    np.random.seed(int(seed)%(2**32-1))
    s = ml.iaaft(np.ascontiguousarray(y,dtype=float), ns=ns,
                 tol_pc=TOL_IAAFT, verbose=False, maxiter=maxiter,
                 sorttype='quicksort')
    s = np.asarray(s,dtype=float)
    if s.shape != (ns, len(y)) or not np.all(np.isfinite(s)):
        raise ValueError(f'IAAFT devolvio forma {s.shape} o valores no finitos.')
    return s


def error_psd(y, s):
    p0 = np.abs(np.fft.rfft(y-y.mean()))**2
    p1 = np.abs(np.fft.rfft(s-s.mean()))**2
    return float(np.linalg.norm(p0-p1)/max(np.linalg.norm(p0),np.finfo(float).tiny))


def cruces(eps, lam):
    out=[]
    for i in range(len(eps)-1):
        a,b=lam[i],lam[i+1]
        if a*b < 0:
            out.append(float(eps[i]-a*(eps[i+1]-eps[i])/(b-a)))
    return out


def estadisticas_boot(a, indices):
    """IC percentil para media (mismas replicas en todo el barrido)."""
    a=np.asarray(a,dtype=float)
    nrep=len(a)
    means=a[indices].mean(axis=1)
    return {'media':a.mean(axis=0),
            'std':a.std(axis=0,ddof=1),
            'ic_inf':np.quantile(means,.025,axis=0),
            'ic_sup':np.quantile(means,.975,axis=0)}


def resumen_y_bootstrap(eps,metricas,nboot,seed):
    """Banda simultanea max-t compartida en los 4 contrastes y todos los eps."""
    rng=np.random.default_rng(seed)
    r=len(next(iter(metricas.values())))
    indices=rng.integers(0,r,size=(nboot,r))
    resultados={}
    df=pd.DataFrame({'epsilon':eps})
    for k,a in metricas.items():
        stats=estadisticas_boot(a,indices)
        resultados[k]=stats
        for tag,val in stats.items():
            df[k+'_'+tag]=val

    keys=['Delta_J_y_shuffle','Delta_J_y_IAAFT',
          'Delta_J_xy_shuffle','Delta_J_xy_IAAFT']
    a=np.stack([metricas[k] for k in keys])    # (4,R,E)
    mean=a.mean(axis=1)                      # (4,E)
    boot=np.empty((4,nboot,len(eps)))
    for b in range(nboot):
        boot[:,b,:]=a[:,indices[b],:].mean(axis=1)
    se=np.std(boot,axis=1,ddof=1)
    mask=se>1e-14
    z=np.zeros_like(boot)
    for j in range(4):
        for i in range(len(eps)):
            if mask[j,i]:
                z[j,:,i]=np.abs((boot[j,:,i]-mean[j,i])/se[j,i])
    q95=np.quantile(np.max(z,axis=(0,2)),.95)
    for j,k in enumerate(keys):
        bot=mean[j]-q95*se[j]
        top=mean[j]+q95*se[j]
        resultados[k]['sim_inf']=bot
        resultados[k]['sim_sup']=top
        df[k+'_sim_inf']=bot
        df[k+'_sim_sup']=top
        df[k+'_excluye_cero_sim']=(bot>0)|(top<0)
    return df,resultados,float(q95)


def marcar_cruces(ax,umbrales):
    for v in umbrales:
        ax.axvline(v,color='0.5',ls=':',lw=1.25)
    ax.grid(alpha=.2)


def graficar(eps,est,umbrales, Jx, carpeta):
    col={'real':'black','shuffle':'tab:orange','IAAFT':'tab:blue'}
    fig,axs=plt.subplots(4,1,figsize=(11,13),sharex=True,constrained_layout=True)
    for row,prefix in enumerate(('J_y','J_xy')):
        a=axs[row]
        for nul,color,label in (('',col['real'],'real'),
                                ('_shuffle',col['shuffle'],'shuffle'),
                                ('_IAAFT',col['IAAFT'],'IAAFT')):
            d=est[prefix+nul]
            a.plot(eps,d['media'],lw=1.6,label=label,color=color)
            a.fill_between(eps,d['ic_inf'],d['ic_sup'],color=color,alpha=.13)
        if row==0:
            a.axhline(float(np.mean(Jx)),lw=1,ls='--',color='0.6',label='J(x) control')
        a.set_ylabel(prefix)
        a.legend(fontsize=9,ncol=4)
    a=axs[2]
    d=est['lambda_c']
    a.plot(eps,d['media'],color='black',lw=1.8,label=r'$\lambda_c$')
    a.fill_between(eps,d['ic_inf'],d['ic_sup'],color='black',alpha=.10)
    a.axhline(0,color='tab:red',lw=1)
    a.set_ylabel(r'$\lambda_c$ [tiempo$^{-1}$]')
    a.legend()
    a=axs[3]
    d=est['error_auxiliar_RMS']
    a.semilogy(eps,np.maximum(d['media'],1e-15),color='tab:purple',label='RMS(y-z)')
    d2=est['error_xy_RMS']
    a.semilogy(eps,np.maximum(d2['media'],1e-15),color='tab:green',label='RMS(x-y)')
    a.set_ylabel('Errores RMS')
    a.set_xlabel(r'Acoplamiento $\epsilon$')
    a.legend()
    for a in axs:
        marcar_cruces(a,umbrales)
    fig.suptitle('Mackey-Glass: IC 95 % entre realizaciones (puntuales)')
    fig.savefig(carpeta/'J_real_vs_nulos_IC95.png',dpi=170)
    if not MOSTRAR_FIGURAS:
        plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True,constrained_layout=True)
    for ax,name in zip(axs,('Delta_J_y','Delta_J_xy')):
        ax.axhline(0,color='0.4',lw=1)
        for nul,color in (('shuffle',col['shuffle']),('IAAFT',col['IAAFT'])):
            d=est[name+'_'+nul]
            ax.plot(eps,d['media'],color=color,lw=1.7,label=nul)
            ax.fill_between(eps,d['sim_inf'],d['sim_sup'],color=color,alpha=.15)
            k=(d['sim_inf']>0)|(d['sim_sup']<0)
            ax.scatter(eps[k],d['media'][k],s=18,color=color,zorder=3)
        ax.legend(fontsize=9)
        ax.set_ylabel(name)
        marcar_cruces(ax,umbrales)
    axs[-1].set_xlabel(r'Acoplamiento $\epsilon$')
    fig.suptitle('Delta J = real - media nula; bandas simultaneas 95 %\n'
                 'Ajuste conjunto para 4 contrastes x todos los acoplamientos')
    fig.savefig(carpeta/'DeltaJ_nulos_bandas_simultaneas.png',dpi=170)
    if not MOSTRAR_FIGURAS:
        plt.close(fig)
    else:
        plt.show()


def main():
    parser=argparse.ArgumentParser(description='MG generalizada + J + IAAFT')
    parser.add_argument('--quick',action='store_true',help='Prueba de funcionamiento con pocos datos')
    args=parser.parse_args()
    t0=time.monotonic()
    R=N_REPETICIONES if not args.quick else 3
    E=N_EPS if not args.quick else 8
    ndata=N_DATOS if not args.quick else 512
    ntrans=N_TRANSITORIO if not args.quick else 150.0
    S=N_SURROGADOS if not args.quick else 1
    nboot=N_BOOTSTRAP if not args.quick else 300
    maxiter=MAXITER_IAAFT if not args.quick else 80
    out=SALIDA if not args.quick else Path(str(SALIDA)+'_PRUEBA')
    out.mkdir(parents=True,exist_ok=True)
    eps=np.linspace(EPS_MIN,EPS_MAX,E)
    if R<2 or S<1 or nboot<100:
        raise ValueError('Configuracion invalida: R>=2, S>=1, B>=100.')

    config=dict(beta_x=BETA_X,gamma_x=GAMMA_X,n_x=N_X,tau_x=RETARDO_X,
                beta_y=BETA_Y,gamma_y=GAMMA_Y,n_y=N_Y,tau_y=RETARDO_Y,
                eps_min=EPS_MIN,eps_max=EPS_MAX,n_eps=E,
                dt=DT,muestreo_cada=MUESTREO_CADA,n_datos=ndata,
                n_transitorio=ntrans,n_repeticiones=R,n_surrogados=S,
                n_bootstrap=nboot,maxiter_iaaft=maxiter,tol_iaaft=TOL_IAAFT,
                tau_J=TAU_J,seed=SEED)
    (out/'configuracion.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Mackey-Glass R={R}, E={E}, N={ndata}; eps={eps[0]:.4f}..{eps[-1]:.4f}')
    print(f'Tiempo real analizado por replica: {ndata*MUESTREO_CADA*DT:.1f}')
    print(f'Surrogados por tipo: {R*E*S} (IAAFT y shuffle)')

    Jx=np.empty(R)
    lam=np.empty((R,E))
    aux=np.empty((R,E))
    exy=np.empty((R,E))
    jy=np.empty((R,E))
    jxy=np.empty((R,E))
    psd=np.empty((R,E))
    n_y={name:np.empty((R,E)) for name in ('shuffle','IAAFT')}
    n_xy={name:np.empty((R,E)) for name in ('shuffle','IAAFT')}
    det=[]
    rng_perm=np.random.default_rng(SEED+999)

    for r in range(R):
        x,ys,lam[r],aux[r],exy[r]=simular_repeticion(eps,SEED+101*r,ntrans,ndata)
        if not np.all(np.isfinite(ys)) or not np.all(np.isfinite(lam[r])):
            raise RuntimeError(f'DDE inestable en repeticion {r+1}; revisar DT.')
        Jx[r]=calcular_j(x)
        for i,e in enumerate(eps):
            y=np.ascontiguousarray(ys[i])
            jy[r,i]=calcular_j(y)
            jxy[r,i]=calcular_j(x,y)
            jshu=np.empty(S)
            jxshu=np.empty(S)
            for s in range(S):
                ysh=rng_perm.permutation(y)
                jshu[s]=calcular_j(ysh)
                jxshu[s]=calcular_j(x,ysh)
            try:
                sia=crear_iaaft(y,S,SEED+10_000+r*E+i,maxiter)
            except Exception as exc:
                raise RuntimeError(f'IAAFT fallo en replica {r+1}, eps={e:.5f}') from exc
            jia=np.empty(S)
            jxia=np.empty(S)
            psderr=np.empty(S)
            for s in range(S):
                jia[s]=calcular_j(sia[s])
                jxia[s]=calcular_j(x,sia[s])
                psderr[s]=error_psd(y,sia[s])
                det.append(dict(repeticion=r+1,epsilon=e,sustituto=s+1,
                                J_y_shuffle=jshu[s],J_xy_shuffle=jxshu[s],
                                J_y_IAAFT=jia[s],J_xy_IAAFT=jxia[s],
                                error_rel_PSD_IAAFT=psderr[s]))
            psd[r,i]=psderr.mean()
            n_y['shuffle'][r,i]=jshu.mean()
            n_y['IAAFT'][r,i]=jia.mean()
            n_xy['shuffle'][r,i]=jxshu.mean()
            n_xy['IAAFT'][r,i]=jxia.mean()
        print(f'Repeticion {r+1}/{R}; {(time.monotonic()-t0)/60:.1f} min',flush=True)

    filas=[]
    for r in range(R):
        for i,e in enumerate(eps):
            row=dict(repeticion=r+1,epsilon=e,lambda_c=lam[r,i],
                     error_auxiliar_RMS=aux[r,i],error_xy_RMS=exy[r,i],
                     J_x=Jx[r],J_y=jy[r,i],J_xy=jxy[r,i],
                     error_rel_PSD_IAAFT=psd[r,i])
            for nul in ('shuffle','IAAFT'):
                row['J_y_'+nul]=n_y[nul][r,i]
                row['J_xy_'+nul]=n_xy[nul][r,i]
                row['Delta_J_y_'+nul]=jy[r,i]-n_y[nul][r,i]
                row['Delta_J_xy_'+nul]=jxy[r,i]-n_xy[nul][r,i]
            filas.append(row)
    pd.DataFrame(filas).to_csv(out/'resultados_por_repeticion.csv',index=False)
    pd.DataFrame(det).to_csv(out/'valores_individuales_surrogados.csv',index=False)

    metricas={'lambda_c':lam,'error_auxiliar_RMS':aux,'error_xy_RMS':exy,
              'error_rel_PSD_IAAFT':psd,'J_y':jy,'J_xy':jxy}
    for nul in ('shuffle','IAAFT'):
        metricas['J_y_'+nul]=n_y[nul]
        metricas['J_xy_'+nul]=n_xy[nul]
        metricas['Delta_J_y_'+nul]=jy-n_y[nul]
        metricas['Delta_J_xy_'+nul]=jxy-n_xy[nul]

    resumen,stats,q95=resumen_y_bootstrap(eps,metricas,nboot,SEED+34567)
    umbrales=cruces(eps,stats['lambda_c']['media'])
    resumen['estable_condicionalmente_media']=stats['lambda_c']['media']<0
    resumen.to_csv(out/'resumen_bootstrap_por_epsilon.csv',index=False)
    graficar(eps,stats,umbrales,Jx,out)

    lines=[f'R={R}; E={E}; N_DATOS={ndata}; dt={DT}; sample_dt={DT*MUESTREO_CADA}',
           f'MG conductor: beta={BETA_X}, gamma={GAMMA_X}, n={N_X}, tau={RETARDO_X}',
           f'MG respuesta: beta={BETA_Y}, gamma={GAMMA_Y}, n={N_Y}, tau={RETARDO_Y}',
           f'Lyapunov condicional (DDE variacional): dt={DT}, renorm={RENORM_CADA*DT}',
           f'IAAFT: ns={S}, maxiter={maxiter}, tolerancia={TOL_IAAFT}',
           f'Error PSD IAAFT media={psd.mean():.8g}; max={psd.max():.8g}',
           f'Bootstrap={nboot}; banda simultanea FWER 4*E, multiplicador max-t={q95:.4g}',
           'Cruces medios lambda_c=0 (interpolados): '+(', '.join(f'{v:.6f}' for v in umbrales) or 'ninguno'),
           'Lambda condicional por unidad de tiempo (NO por observacion).',
           'J x,y compara fases de Fourier de dos series, no angulos dinamicos.',
           'Importante: y,z podrian coincidir a precision de maquina aun con lambda>0.',
           'La significancia vs nulos NO demuestra por si sola asociacion con el umbral.',
           'Si el valor de lambda es muy cercano a 0, afinar paso dt y repetir.']
    for k in ('Delta_J_y_shuffle','Delta_J_y_IAAFT',
              'Delta_J_xy_shuffle','Delta_J_xy_IAAFT'):
        d=stats[k]
        count=int(np.sum((d['sim_inf']>0)|(d['sim_sup']<0)))
        lines.append(f'{k}: {count}/{E} diferencias significativas (ajuste global)')
    (out/'LEEME_resultados.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'+'\n'.join(lines))
    print(f'\nResultados guardados: {out.resolve()}')
    print(f'Total: {(time.monotonic()-t0)/60:.1f} min')


if __name__=='__main__':
    main()
