# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import stats
from rccSim import rccSim



# p=10
# x = np.random.random((p,p))
# M = np.random.random((p,p))
# nu=20
def dwishart (x, M, nu, logged = True):
    
  
    """
    Probability Density Function for Wishart Distribution:
    
    This function provides the probability density function for the Wishart distribution.
    
    Parameters
    ----------
    x : numpy.ndarray
        p x p positive definite matrix.
    M : numpy.ndarray
        p x p mean matrix. Note that M = nu * V where V is the scale matrix.
    nu : float
        Degrees of freedom.
    logged : bool, optional
        If True, probability given on log scale.
    
    Returns
    -------
    float
        The probability density function value.
    
    """
    x = (x + np.transpose(x)) / 2
    M = (M + np.transpose(M)) / 2
    p = x.shape[0]
    lnumr = (nu - p - 1) /( 2 * np.log(np.linalg.det(x))) - (nu / (2 * np.sum(np.diag(np. linalg. inv(x) * x))))  
    ldenom = (nu * p / 2) * np.log(2) + (nu / 2) * np.log(np.linalg.det(1 / nu * M)) + (p * (p - 1) / 4) * \
                                   np.log(np.pi) + np.sum([math.lgamma(nu / 2 + (1 - j) / 2) for j in range(1, p+1)])
    return (lnumr - ldenom) if logged else (np.exp(lnumr - ldenom))
    
              
def adj(mat, thresh = 0.001):
    """
    Adjacency Matrix:
    
    This function calculates an adjacency matrix for the matrix mat based on the absolute value threshold of thresh.
    
    Parameters
    ----------
    mat : numpy.ndarray
        Numeric matrix.
    thresh : float, optional
        Threshold for absolute value of entries.
    
    Returns
    -------
    numpy.ndarray
        An adjacency matrix containing 1's and 0's.
    
    """
    return((abs(mat) > thresh) + 0)


def zToA(z):
    """
    Z to Adjacency Matrix:
    
    This function calculates an adjacency matrix based on an integer vector of cluster memberships.
    
    Parameters
    ----------
    z : numpy.ndarray
        Integer vector of cluster memberships.
    
    Returns
    -------
    numpy.ndarray
        An adjacency matrix containing 1's and 0's.
    
    Examples
    --------
    # Calculate adjacency matrix for clustering
    zToA(np.concatenate([np.repeat(1, 2), np.repeat(2, 2), np.repeat(3, 2)]))
    
    """
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
    """
   Rand Index:
   
   This function calculates the rand index describing the amount of agreement between two integer vectors of cluster memberships.
   
   Parameters
   ----------
   x : numpy.ndarray
       First integer vector of cluster memberships.
   y : numpy.ndarray
       Second integer vector of cluster memberships.
   
   Returns
   -------
   float
       The rand index value, bounded between 0 and 1.
   
   """
        
    Ahat = zToA(x)[np.tril_indices_from(zToA(x ), k= -1)]
    A0 = zToA(y)[np.tril_indices_from(zToA(y), k= -1)]
    return((sum((Ahat - A0) == 2) + sum((Ahat - A0) == 0)) / math.comb(len(x), 2))
        
       

def rccmLogLike(omegaks,omega0s,x,ws,lambda2):
    """
   Model Log-Likelihood:
       
   This function calculates the model log-likelihood
   for the random covariance clustering model (RCCM)

   Parameters
   ----------
   omegaks: np.ndarray
       K x p x p array of K number of estimated subject-level precision matrices.
   omega0s: np.ndarray
       nclusts x p x p array of nclusts number of estimated cluster-level precision matrices.
   ws: np.ndarray
       nclusts x K matrix of estimated cluster weights for each subject (weights).
   x: list
       List of K data matrices each of dimension n_k x p.
   lambda2: float
       Non-negative scalar value used as input to rccm function to obtain estimates.

   Returns
   -------
   float: Model log-likelihood

   Examples
   --------
   # Generate data
   np.random.seed(1994)
   myData = rccSim(G=2, clustSize=10, p=10, n=100, overlap=0.50, rho=0.10)

   # Analyze with RCCM
   resultRccm = rccm(x=myData['simDat'], lambda1=20,
                     lambda2=325, lambda3=0.01, nclusts=2)

   # Calculate model log-likelihood
   rccmLogLike(omegaks=resultRccm['Omegas'], omega0s=resultRccm['Omega0'],
               ws=resultRccm['weights'], x=myData['simDat'], lambda2=325)
   """

    G = len(omega0s)
    K = len(omegaks)

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
    """
   AIC:
   
   This function calculates the AIC value for the random covariance clustering model (RCCM).
   
   Parameters:
   omegaks (np.ndarray): K x p x p array of K number of estimated subject-level precision matrices.
   omega0s (np.ndarray): nclusts x p x p array of nclusts number of estimated cluster-level precision matrices.
   ws (np.ndarray): nclusts x K matrix of estimated cluster weights for each subject (weights).
   x (list): List of K data matrices each of dimension n_k x p.
   lambda2 (float): Non-negative scalar value used as input to rccm function to obtain estimates.

   Returns:
   float: Numeric AIC value.
   
   Examples:
       
    # Generate data
    set.seed(1994)
    myData = rccSim(G = 2, clustSize = 10, p = 10, n = 100, overlap = 0.50, rho = 0.10)
    
    # Analyze with RCCM
    resultRccm = rccm(x = myData['simDat'], lambda1 = 20,
                   lambda2 = 325, lambda3 = 0.01, nclusts = 2)

   # Calculate AIC
   aic(omegaks = resultRccm['Omegas'], omega0s = resultRccm['Omega0'],
          ws = resultRccm['weights'], x = myData['simDat'], lambda2 = 325)
   
   """
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

    