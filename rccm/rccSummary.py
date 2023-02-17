# -*- coding: utf-8 -*-


import numpy as np
import math
from scipy import stats
from rccSim import rccSim



p=10
x = np.random.random((p,p))
M = np.random.random((p,p))
nu=20
def dwishart (x, M, nu, logged = True):
  x = (x + np.transpose(x)) / 2
  M = (M + np.transpose(M)) / 2
  p = x.shape[0]
  lnumr = (nu - p - 1) /( 2 * np.log(np.linalg.det(x))) - (nu / (2 * np.sum(np.diag(np. linalg. inv(x) * x))))  
  ldenom = (nu * p / 2) * np.log(2) + (nu / 2) * np.log(np.linalg.det(1 / nu * M)) + (p * (p - 1) / 4) * \
                                   np.log(np.pi) + np.sum([math.lgamma(nu / 2 + (1 - j) / 2) for j in range(1, p+1)])
  return (lnumr - ldenom) if logged else (np.exp(lnumr - ldenom))
    
              
def adj(mat, thresh = 0.001):
    return((abs(mat) > thresh) + 0)


def zToA(z):
    K = len(z)
    A = np.zeros((K,K))
    for r in range(K):
        for s in range(K):
           if z[r] != 0 and z[s] != 0:
               A[r, s] = int(z[r] == z[s])
           else:
               A[r, s] = 0
    return A




              
def randCalc(x,y):
        
        Ahat = zToA(x)[np.tril_indices_from(zToA(x ), k= -1)]
        A0 = zToA(y)[np.tril_indices_from(zToA(y), k= -1)]
        return((sum((Ahat - A0) == 2) + sum((Ahat - A0) == 0)) / math.comb(len(x), 2))
        
       



G,clustSize,p,n, overlap = 2,(67,37),10,177,0.5
rho,esd, gtype, eprob = 0.10,0.05,"hub",0.5


sim = rccSim(G,clustSize,p,n,overlap,rho,esd,gtype,eprob)



x = sim["simDat"]
g0s = sim["g0s"]
omega0s = sim["Omega0s"]
gks = sim["gks"]
omegaks = sim["Omegaks"]




G = len(omega0s)
K = len(omegaks)


ws = np.random.randint(2,size=(K,G))

lambda2 = 135


def rccmLogLike(omegaks,omega0s,x,ws,lambda2):

    
    mll = 0
    for k in range(K):
        lk1 = np.sum(stats.multivariate_normal.logpdf(x[k],mean = np.mean(x[k], axis =0) ,\
                                                      cov = np.linalg.inv(omegaks[k])))
        # lk1 = np.sum(stats.multivariate_normal.logpdf(x[k], mean=np.zeros(len(omegaks[k])), \
        #              cov = np.linalg.inv(omegaks[k])))
        if any(ws[k,:] == 1):
          lk2 = dwishart(omegaks[k], M = omega0s[np.where(ws[k,:] == 1)[0][0]],logged = False, nu = lambda2)
          mll+= lk1 + lk2
        else:
           list_g = [g for g in range(G)]
           lk2 = np.log(sum(list(map(lambda g: ws[k,g]*dwishart(omegaks[k], \
                                                 M = omega0s[g],logged = False, nu = lambda2),list_g))))
               
           # lk2 = np.log(np.sum(np.array([ws[k,g]*dwishart(omegaks[k], nu=lambda2,\
           #                                      M = omega0s[g], logged=False) for g in range(G)])))
           mll+= lk1 + lk2  
    return mll
     
    


def aic(omegaks, omega0s, ws, x, lambda2):
    K = len(omegaks)
    G = len(omega0s)
    
    X =[k for k in range(K)]
    dfks = map(lambda k: sum(adj(omegaks[k][np.tril_indices_from(omegaks[k])])),X)
    
    list_G = [g for g in range(G)]
    dfgs = map(lambda g: sum(adj(omega0s[g][np.tril_indices_from(omega0s[g])])),list_G)
    
    
    modelDim = sum(list(dfks), list(dfgs))
    mll = rccmLogLike(omegaks = omegaks, omega0s = omega0s, ws = ws, x = x, lambda2 = lambda2)

    aic = 2*modelDim - 2*mll
    return(aic)

    
    
    
    
    
    
    
    
    