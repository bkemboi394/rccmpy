# -*- coding: utf-8 -*-


"""
Random Covariance Clustering Model

This function implements the Random Covariance Clustering Model (RCCM) for joint estimation of
sparse precision matrices belonging to multiple clusters or groups. 

Optimization is conducted using block coordinate descent.

Parameters:
x : list
List of K data matrices each of dimension n_k x p
lambda1 : float
Non-negative scalar. Induces sparsity in subject-level matrices.
lambda2 : float
Non-negative scalar. Induces similarity between subject-level matrices and cluster-level matrices.
lambda3 : float
Non-negative scalar. Induces sparsity in cluster-level matrices.
nclusts : int
Number of clusters or groups.
delta : float
Threshold for convergence.
max.iters : int
Maximum number of iterations for block coordinate descent optimization.
z0s : list
Vector of length K with initial cluster memberships.

Returns:
result : dict
A dictionary containing:
- Omega0: nclusts x p x p array of nclusts number of estimated cluster-level precision matrices.
- Omegas: K x p x p array of K number of estimated subject-level precision matrices.
- weights: nclusts x K matrix of estimated cluster weights for each subject.

Examples:
Generate data with 2 clusters with 12 and 10 subjects respectively,15 variables for each subject, 
100 observations for each variable for each subject,the groups sharing about 50% of network connections, 
and 10% of differential connections within each group.

myData = rccSim(G=2, clustSize=[12, 10], p=15, n=100, overlap=0.50, rho=0.10)

Analyze simulated data with RCCM
result = rccm(x=myData['simDat'], lambda1=10, lambda2=50, lambda3=2, nclusts=2, delta=0.001)
"""


import numpy as np
from sklearn.covariance import GraphicalLasso
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def rccm(x,  nclusts, lambda1, lambda2, lambda3=0, delta=0.001, max_iters=100, z0s=None, ncores=1):
        
   # Function for making almost symmetric matrix symmetric
    def make_symmetric(x):
        return (x + x.T) / 2
    #Inputs
    K = len(x)
    G = nclusts
    p = x[0].shape[1]
    Sl = np.array([np.cov(i, rowvar=False) for i in x])
    nks = [i.shape[0] for i in x]
    
    # Initializing subject-level matrices
    Omegas = []
    for k in range(K):
        pdStart = Sl[k,:,:] + np.diag(np.repeat(1e-6, p))
        gl = GraphicalLasso(alpha=0.001, mode='cd', tol=1e-6, verbose=False, 
                            enet_tol=1e-6, assume_centered=True,max_iter=500)#warm_start=True)
        gl.fit(pdStart)
        Omegas.append(make_symmetric(gl.precision_))
    Omegas = np.array(Omegas)
    
    
    # Initializing weights using hierarchical clustering based on dissimilarity matrix of
     # Frobenius norm of glasso matrix differences

    K = len(Omegas)
    distMat = np.empty((K, K))
    for r in range(K):
        for s in range(K):
            distMat[r, s] = np.linalg.norm(Omegas[r,:,:] - Omegas[s,:,:], ord='fro')
    
    
    

    if z0s is None:
        linkage_matrix = linkage(squareform(distMat), method='ward')
        cl0 = fcluster(linkage_matrix, G, criterion='maxclust')
    else:
        cl0 = z0s
        
    wgk = np.empty((G, K))
    for i in range(G):
        for j in range(K):
            wgk[i, j] = 1 if cl0[j] == (i + 1) else 0
            
    # Initializing cluster-level matrices to be all 0's
    Omega0 = np.zeros((G,p, p))
    Omegas_old = np.zeros((K,p, p))
    Omega0_old = np.zeros((G,p, p))
    counter = 0
    
    # Initializing vector for deltas and array for weights across iterations
    deltas = []
    wArray = np.full((max_iters + 1, G, K), np.nan)
    wArray[0,:, :] = wgk
    
    
    
    
    
    
    #Start BCD Algorithm
    while np.max(abs(Omega0 - Omega0_old)) > delta or np.max(abs(Omegas - Omegas_old)) > delta or counter < 1:

        counter += 1
        
        # Exit if exceeds max.iters
        if counter >= max_iters:
            res = {"Omega0": Omega0, "Omegas": Omegas, "weights": wgk}
            print(f"Omegas fail to converge for lambda1 = {lambda1}, lambda2 = {lambda2}, lambda3 = {lambda3}, delta = {deltas[counter - 1]}")
            # Returning results
            return res
        
        # record current Omega0 & Omegas
        Omega0_old = Omega0
        Omegas_old = Omegas
        
        # 1st step: Updating pi's
        pigs = 1/(K*np.sum(wgk, axis=1))
        
        # 2nd step: updating cluster-level precision matrices
        
        # Calculating weighted-sum of subject-level matrices
        inv0 = np.zeros((G,p, p))
        for g in range(G):
            wks = np.array([wgk[g, k] * Omegas[k,:, :] for k in range(K)])
            s0 = np.sum(wks, axis=0)
            
            # s0 = s0/ np.sum(wgk[g, :])
            # penMat = np.full((p, p), lambda3 / (lambda2 * np.sum(wgk[g, :])))
            # np.fill_diagonal(penMat, 0)
            
            # if counter > 1:
            #     L = np.linalg.cholesky(s0)
            #     s0_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(p)))
            #     Omega0[g, :, :] = s0_inv - np.diag(np.diag(s0_inv)) + penMat / lambda2
            # else:
            #     Omega0[g, :, :] = np.linalg.inv(s0)
                
                

        for g in range(G):
            S0 = s0[g:, :] / np.sum(wgk[g, :])
            penMat = np.full((p,p), lambda3 / (lambda2 * np.sum(wgk[g, :])))
            np.fill_diagonal(penMat, 0)
            if counter > 1:
                model = GraphicalLasso(alpha=penMat, tol=delta, max_iter=100)
                model.fit(Omega0[g,:, :]@ S0 @ Omega0[g,:, :])
                Omega0[g,:, :] = model.precision_
            else:
                model = GraphicalLasso(alpha=penMat, tol=delta, max_iter=100)
                model.fit(S0)
                Omega0[g,:, :] = model.precision_

        inv0[g, :, :] = np.linalg.inv(Omega0[g, :, :])
            

       
        # for g in range(G):
        #             S0 = s0[g,:, :] / np.sum(wgk[g, :])
        #             penMat = np.full((p, p), lambda3 / (lambda2 * np.sum(wgk[g, :])))
        #             np.fill_diagonal(penMat, 0)
        #             if counter > 1:
        #                 Omega0[g,:, :] = spcov.spcov(Sigma=Omega0[g,:, :], S=S0, lambd=penMat,\
        #                                               tol_outer=delta, step_size=100)['Sigma']
        #             else:
        #                 Omega0[g,:, :] = spcov.spcov(Sigma=np.linalg.inv(S0), S=S0, lambd=penMat,\
        #                                               tol_outer=delta, step_size=100)['Sigma']
                            
        #             # Calculating inverse of Omega_g for Omega_k and w_gk updates
        #             inv0[g,:,:] = np.linalg.inv(Omega0[g,:,:])
                    
         # 2b step: updating weights

         # Weight matrix where each row is for a cluster and each column for a subject
                    
        for i in range(G):
            det0 = np.linalg.det(Omega0[i, :, :])
            for j in range(K):
                wgk[i, j] = np.log(pigs[i]) - lambda2 / 2 * np.sum(np.diag(np.dot(inv0[i, :, :],\
                                Omegas[j,:, :]))) + (-lambda2 / 2) * (np.log(1 / lambda2 ** p) + np.log(det0))
        wgk = np.apply_along_axis(lambda column: np.exp(column - np.max(column))\
                                  / np.sum(np.exp(column - np.max(column))), axis=0, arr=wgk)
            
            # 3rd step: updating subject-level precision matrices
        sk = np.array([((nks[k] * Sl[k, :, :] + lambda2 * np.sum([wgk[g, k] * inv0[g, :, :] \
                        for g in range(G)], axis=(0, 1))) / (nks[k] + lambda2 - p - 1)) for k in range(K)])
        
        #rhoMat = np.array([np.full((p, p), lambda1 / (nks[x] + lambda2 - p - 1)) for x in range(K)])
        
        for k in range(K):
                    #np.fill_diagonal(rhoMat[k,:, :], 0)
                    rho = lambda1 / (nks[k] + lambda2 - p - 1)
                    gl_model = GraphicalLasso(alpha=rho, tol=1e-4)
                    gl_model.fit(sk[k,:,:])
                    Omegas[k,:, :] = gl_model.precision_
        # 4th step: updating weights

         # Weight matrix where each row is for a cluster and each column for a subject
        for i in range(G):
                det0 = np.linalg.det(Omega0[i,:, :])
                for j in range(K):
                    wgk[i, j] = np.log(pigs[i]) - lambda2 / 2 * np.sum(np.diag(inv0[i, :, :] \
                                   @ Omegas[j, :, :])) - (lambda2 / 2) * (np.log(1 / lambda2**p) + np.log(det0))          
                
        wgk = np.apply_along_axis(lambda column: np.exp(column - np.max(column)), axis=1, arr=wgk)
        wgk = np.apply_along_axis(lambda column: column / np.sum(column), axis=1, arr=wgk)
        if np.sum(np.isnan(wgk)) > 0:
             raise Exception("NaN value detected in wgk")
         # Record BCD iteration
         
        counter += 1
        deltas = np.concatenate((deltas, [np.max(np.abs(Omega0 - Omega0_old)), np.max(np.abs(Omegas - Omegas_old))]))
        wArray[counter+1, :, :] = wgk


    res = {"Omega0": Omega0, "Omegas": Omegas, "weights": wgk}
    return res


    
    
    
    
    
    
    
    
    
    
    


