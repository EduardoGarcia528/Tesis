import numpy as np
import sys
from scipy.integrate import odeint


class LYAP(object):
    '''Compute the largest Lyapunov exponent of a timeseries'''
    def __init__(self, data):
        self.data = data
        self.datcnt = len(data)

    def basgen(self, tau, ndim, ires, maxbox):
        """
        returns ndim, ires, tau, datcnt, boxcnt, datmax, datmin,
        boxlen, datptr[:boxcnt], nxtbox[:boxcnt, :ndim]
        , where[:boxcnt,:ndim], nxtdat[:datcnt], data
        in a dictionary
        """
        delay = np.array([0,tau,(ndim-1)*tau],dtype=int)

        nxtbox = np.zeros((maxbox,ndim),dtype=int)
        where = np.zeros((maxbox,ndim),dtype=int)
        datptr =  np.full(maxbox,-1,dtype=int)
        nxtdat = np.zeros(self.datcnt,dtype=int)

        datmin = min(self.data)
        datmax = max(self.data)

        datmin = datmin - 0.01*(datmax - datmin)
        datmax = datmax + 0.01*(datmax - datmin)
        boxlen = (datmax - datmin)/ires 
    
        boxcnt = 1

        for i in range(int(self.datcnt-(ndim-1)*tau)):
            target = np.floor((self.data[i+delay]-datmin)/boxlen).astype(int)
            runner = 1
            chaser = 0

            j = 0
            while j < ndim:
                print('count8',flush=True)
                tmp = where[int(runner),j]-target[j]
                if tmp < 0:
                    chaser = runner
                    runner = nxtbox[int(runner),j]
                    if runner != 0:
                        continue
                if tmp != 0 :
                    boxcnt += 1

                    if boxcnt == maxbox:
                        print('Grid overflow, increase number of box count')
                        sys.exit()

                    for k in range(ndim):
                        where[boxcnt,k] = where[int(chaser),k]
                    where[boxcnt,j] = target[j]
                    nxtbox[int(chaser),j] = boxcnt
                    nxtbox[boxcnt,j] = runner
                    runner = boxcnt
                j += 1
            nxtdat[i] = datptr[int(runner)]
            datptr[int(runner)] = i
            
        used = 0
        for i in range(boxcnt+1):
            if datptr[i] != -1:
                used += 1
#       print('Created: ', boxcnt,flush=True)
#       print('Used: ', used,flush=True)
#       print('nxtbox',nxtbox[1:boxcnt+1, :ndim]-1)
#       print('where',where[1:boxcnt+1,:ndim]-1)
#       print('datptr',datptr[1:boxcnt+1])
        newdict = {'ndim':int(ndim), 'ires':int(ires), 'tau':int(tau), 'datcnt':self.datcnt, 'boxcnt':int(boxcnt), 'datmax':datmax, 'datmin':datmin, 'boxlen':boxlen, 'datptr':datptr[1:boxcnt+1], 'nxtbox':nxtbox[1:boxcnt+1, :ndim]-1, 'where':where[1:boxcnt+1,:ndim]-1, 'nxtdat':nxtdat[0:self.datcnt], 'data':self.data}
    
    
        return newdict


    def search(iflag,ndim,ires,datmin,
            boxlen,nxtbox,where,datptr,nxtdat,data,delay,oldpnt,newpnt,
            datuse,dismin,dismax,thmax,evolve):
        """
        searches for the most viable point for fet
        return bstpnt, bstdis, thbest
        """
        target = np.zeros(ndim,dtype=int)
        oldcrd = np.zeros(ndim)
        zewcrd = np.zeros(ndim)
    
        oldcrd = data[int(oldpnt)+delay]
        zewcrd = data[int(newpnt)+delay]
        igcrds = np.floor((oldcrd - datmin) / boxlen).astype(int) -1 #added -1
        oldist = np.sqrt(np.sum(np.power(oldcrd - zewcrd,2)))
        irange = int(np.round(dismin/boxlen))
        if irange == 0 :
            irange = 1
    
        thbest = thmax
        bstdis = dismax
        bstpnt = -1
    
        goto30 = 1
        while goto30 == 1:
            print('count4',flush=True)
            goto30 = 0
            for icnt in range(int(((2*irange+1)**ndim))):
                goto140 = 0
                icounter = icnt
                for i in range(ndim):
                    ipower = int(np.power(2*irange+1,ndim-(i+1)))
                    ioff = int(np.floor(icounter/ipower))
                    icounter = icounter - ioff*ipower
                    target[i] = igcrds[i] - irange + ioff
#                    print('ipower',ipower,' ',ioff,' ',icounter,' ',target,flush=True)
    
                    if target[i] < -1:
                        goto140 = 1
                        break
                    if target[i] > ires-2:
                        goto140 = 1
                        break
    
                if goto140 ==1:
                    continue

                if irange != 1:
                    iskip = 1
                    for i in range(ndim):
                        if abs(int(np.round(target[i] - igcrds[i]))) == irange:
                            iskip = 0
                    if iskip == 1:
                        continue
                runner = 0
                for i in range(ndim):
                    goto80 = 0
                    goto70 = 1
                    while goto70 == 1:
                        print('count5',flush=True)
                        goto70 = 0
                        if where[int(runner),i] == target[i]:
                            goto80 = 1
                            break
                        runner = nxtbox[int(runner),i]
                        if int(runner) !=-1 :
                            goto70 = 1

                    if goto80 == 1:
                        continue
                    goto140 = 1
                    break
                if goto140 == 1:
                    continue
    
                if int(runner) == -1:
                    continue
                runner = datptr[int(runner)]
                if int(runner) == -1:
                    continue
                goto90 = 1
                while goto90 == 1:
                    print('count6',flush=True)
                    goto90 = 0
                    while True:
                        print('count7',flush=True)
                        if abs(int(np.round(runner-oldpnt))) < evolve:
                            break
                        if abs(int(np.round(runner - datuse))) < (2*evolve):
                            break

                        bstcrd = data[int(runner)+delay]
                        abc1 = oldcrd - bstcrd
                        abc2 = oldcrd - zewcrd
                        tdist = np.sum(abc1*abc1)
                        tdist = np.sqrt(tdist)
                        dot = np.sum(abc1*abc2)
    
                        if tdist < dismin:
                            break
                        if tdist >= bstdis:
                            break
                        if tdist == 0:
                            break
                        goto120 = 0
                        if iflag == 0 :
                            goto120 = 1
                        if goto120 == 0:
                            ctheta = min(abs(dot/(tdist*oldist)),1)
                            theta = 57.3*np.arccos(ctheta)
                            if theta >= thbest:
                                break
                            thbest = theta
                        bstdis = tdist
                        bstpnt = runner
                        break
                    runner = nxtdat[int(runner)]
                    if runner != -1:
                        goto90 = 1
    
            irange += 1
            if irange <= (0.5 + int(np.round((dismax/boxlen)))):
                goto30 = 1
                continue
            return bstpnt, bstdis, thbest

    def fet(db, dt, evolve, dismin, dismax, thmax):
    
        out = []
        
        ndim = db['ndim']
        ires = db['ires']
        tau = db['tau']
        datcnt = db['datcnt']
        datmin = db['datmin']
        boxlen = db['boxlen']
        
        datptr = db['datptr']
        nxtbox = db['nxtbox']
        where = db['where']
        nxtdat = db['nxtdat']
        data = db['data']
    
        delay = np.array([0,tau,(ndim-1)*tau])
        datuse = datcnt - (ndim-1)*tau - evolve
    
        its = 0
        SUM = 0
        savmax = dismax
    
        oldpnt = 0     #1 in matlab original
        newpnt = 0     #1 in matlab original
        goto50 = 1
        while goto50 == 1:
            goto50 = 0
            print('count1',flush=True)
            bstpnt, bstdis, thbest = LYAP.search(0, ndim, ires, datmin, boxlen, nxtbox, where, \
                    datptr, nxtdat, data, delay, oldpnt, newpnt, datuse, dismin, dismax, \
                    thmax, evolve)
            while bstpnt == -1 :
                print('count2',flush=True)
                dismax = dismax * 2
                bstpnt, bstdis, thbest = LYAP.search(0, ndim, ires, datmin, boxlen, nxtbox, where, \
                    datptr, nxtdat, data, delay, oldpnt, newpnt, datuse, dismin, dismax, \
                    thmax, evolve)
    
            dismax = savmax 
            newpnt = bstpnt
            disold = bstdis
            iang = -1
    
            goto60 = 1
            while goto60 == 1:
                print('count3',flush=True)
                goto60 = 0
    
                oldpnt += evolve
                newpnt += evolve
    
                if oldpnt >= datuse:
                    print('Lyapunov exponent: ', zlyap)
                    return out, SUM, zlyap
    
                if newpnt >= datuse:
                    oldpnt = oldpnt - evolve
                    goto50 = 1
                    break
    
                p1 = data[int(oldpnt) + delay]
                p2 = data[int(newpnt) + delay]
                disnew = np.sqrt(np.sum(np.power(p2-p1,2)))
    
                its = its + 1
    
                SUM = SUM + np.log(disnew/disold)
                zlyap =  SUM/(its*evolve*dt*np.log(2))    # base 2 Lyapunov exponent 
                print('***********',flush=True)
                print('z_lyap:  ',zlyap,flush=True)
    
                out = [out, its*evolve, disold, disnew, zlyap, (oldpnt-evolve), (newpnt-evolve)]
    
                #if iang == -1:
                #   fprintf(fileID, out[end,0:3])
                #else:
                #   fprintf(fileID, [out[end,0:3i],iang])
                
                if disnew <= dismax:
                    disold = disnew
                    iang = -1
                    goto60 = 1
                    continue
                
                bstpnt, bstdis, thbest = LYAP.search(1, ndim, ires, datmin, boxlen, nxtbox, where, \
                    datptr, nxtdat, data, delay, oldpnt, newpnt, datuse, dismin, dismax, \
                    thmax, evolve)
                
                if bstpnt != -1: 
                    newpnt = bstpnt
                    disold = bstdis
                    iang = np.floor(thbest)
                    goto60 = 1
                    continue
                else:
                    goto50 = 1
                    break


    def lyap_e(self,tau=10,ndim=3,ires=10,maxbox=6000,
            dt=0.01,evolve=20,dismin=0.001,dismax=0.3,thmax=30):
    
        db = LYAP.basgen(self,tau,ndim,ires,maxbox)
    
        l = LYAP.fet(db,dt,evolve,dismin,dismax,thmax)
    
        return l[-1]  #returns the Lyapunov exponent in base 2
    

def _acf_tau_guess(x, max_lag=200):
    """Estimación simple de tau: primer mínimo del ACF (o primer cruce a negativo)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    # autocorrelación rápida vía FFT
    fft = np.fft.rfft(x, n*2)
    acf = np.fft.irfft(fft*np.conjugate(fft))[:n]
    acf /= acf[0]
    # busca primer mínimo local (>0 lag) o primer cruce a negativo
    ac = acf[1:min(max_lag, n-1)]
    # cruce a negativo
    neg = np.where(ac < 0)[0]
    if len(neg) > 0:
        return int(neg[0]) + 1
    # mínimo local
    for k in range(1, len(ac)-1):
        if ac[k] < ac[k-1] and ac[k] < ac[k+1]:
            return k + 1
    return 10  # fallback

def compute_lyap_from_txt(
    path,
    col=None,          # índice de columna con la señal; si None, usa la última
    time_col=None,     # índice de columna de tiempo; si None, intenta deducir o usa dt=1.0 si no se da
    dt=None,           # delta t; si None e incluye tiempo, lo estima
    tau=None,          # retardo de embedding; si None, lo estima con ACF
    ndim=3,            # dimensión de embedding (Wolf usa típicamente 3-7; empieza con 3)
    ires=16,           # resolución de la rejilla (10–50 suele ir bien)
    maxbox=60000,      # cajas máximas (sube si tu serie es larga)
    evolve=20,         # pasos de avance entre re-normalizaciones
    dismin=1e-3,       # distancia mínima inicial
    dismax=0.3,        # distancia máxima antes de re-elegir vecino
    thmax=30           # umbral angular (en grados)
):
    # Carga robusta (maneja encabezados vacíos, espacios, etc.)
    try:
        arr = np.loadtxt(path)
    except Exception:
        arr = np.genfromtxt(path, filling_values=np.nan)
    if arr.ndim == 1:
        data = np.asarray(arr, float)
        tvec = None
    else:
        ncols = arr.shape[1]
        if time_col is not None and 0 <= time_col < ncols:
            tvec = arr[:, time_col].astype(float)
        else:
            # si parece que la primera columna es monotónica, úsala como tiempo
            cand = arr[:, 0]
            tvec = cand if np.all(np.diff(cand) > 0) else None
        if col is None:
            col = ncols - 1
        data = arr[:, col].astype(float)

    # Limpieza simple: quita NaNs y escala
    mask = np.isfinite(data)
    if tvec is not None:
        mask &= np.isfinite(tvec)
        tvec = tvec[mask] if tvec is not None else None
    data = data[mask]
    if tvec is not None: 
        tvec = tvec[mask]

    # Estima dt si no lo diste
    if dt is None:
        if tvec is not None and len(tvec) >= 2:
            dt = float(np.median(np.diff(tvec)))
        else:
            dt = 1.0  # asume unidades de índice

    # Estima tau si no lo diste
    if tau is None:
        tau = _acf_tau_guess(data, max_lag=min(200, len(data)//10))

    # Normaliza (opcional pero ayuda)
    dstd = data.std()
    if dstd > 0:
        data = (data - data.mean()) / dstd

    # Verificación mínima de longitud
    min_len = (ndim - 1) * tau + evolve + 5
    if len(data) < min_len:
        raise ValueError(
            f"La serie es muy corta para ndim={ndim}, tau={tau}, evolve={evolve}. "
            f"Se requieren al menos ~{min_len} puntos y tienes {len(data)}."
        )

    # Ejecuta LYAP
    ly = LYAP(data)
    l2 = ly.lyap_e(
        tau=int(tau),
        ndim=int(ndim),
        ires=int(ires),
        maxbox=int(maxbox),
        dt=float(dt),
        evolve=int(evolve),
        dismin=float(dismin),
        dismax=float(dismax),
        thmax=float(thmax),
    )
    # l2 está en “bits por unidad de tiempo” (base 2, ya divide por dt*log(2))
    return dict(
        lyap_base2_per_time=l2,
        tau=int(tau),
        dt=float(dt),
        ndim=int(ndim),
        ires=int(ires),
        maxbox=int(maxbox),
        evolve=int(evolve)
    )

# --------- Ejemplos de uso ----------
# 1) Archivo con columnas: tiempo, señal
X = []
x = 0.4
for _ in range(2000):
    x = 3.56554*x*(1-x)
for _ in range(2000):
    x = 3.56554*x*(1-x)
    X.append(x)
np.savetxt('caos_logistic.txt', X)
res = compute_lyap_from_txt("caos_logistic.txt", time_col=0, col=1)

# 2) Archivo con solo la señal (sin tiempo) y tú sabes dt=0.01
# res = compute_lyap_from_txt("mi_array.txt", dt=0.01)

# 3) Forzar parámetros (por ejemplo tau=12, ndim=4)
# res = compute_lyap_from_txt("mi_array.txt", time_col=0, col=1, dt=None, tau=12, ndim=4)

print(res)
    



