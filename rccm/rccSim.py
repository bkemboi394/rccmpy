# -*- coding: utf-8 -*-

import numpy as np
import math


# Change the variable values to desired

G, clustSize, p, n, overlap = 2, (67, 37), 10, 177, 0.5
rho, esd, gtype, eprob = 0.10, 0.05, "hub", 0.5


def rccSim(G, clustSize, p, n, overlap, rho, esd, graphtype, eprob):
    # Calculating total number of subjects
    K = 0
    if len(clustSize) == 1:
        K += G * clustSize[0]
    else:
        K += sum(clustSize)
    
    g0s = np.zeros((G,p, p))
    gks = np.zeros((K,p, p))
    Omega0s = np.zeros((G,p, p))
    Omegaks = np.zeros((K,p, p))


    if len(clustSize) != 1 and len(clustSize) != G:

        raise ValueError("clustSize must be of length 1 or of length equal to the number of clusters")

    else:

        if (len(clustSize) > 1):

            Zgks = []
            for g in range(len(clustSize)):
                zgks = list(np.repeat(g + 1, clustSize[g]))
                Zgks.extend(zgks)

        else:
            for i in range(G):
                zgks.repeat(1 + G, clustSize[0])
    
    simData = list()

    # Manually generating cluster-level graphs and precision matrices

    # Number of hubs
    J = math.floor(math.sqrt(p))

    def symmPosDef(m):  # m is a numpy array
        m = m + np.transpose(m)
        eigen_values = np.linalg.eigvals(m)
        smallE = min(eigen_values)
        if smallE <= 0:
            m = m + np.diag(np.repeat(abs(smallE) + 0.10 + 0.10, m.shape[0]))
        return m

    # Determining edges to be shared across groups
    numE = p - J
    q = int(math.comb(p, 2))
    if gtype == "hub":
        numshare = math.floor(numE * overlap)
    else:
        numshare = math.floor(q * overlap)
    
    eshare = np.array([np.array(np.tril_indices(p, k=-1)).T[x] for x in\
                       np.random.randint(0, q, size=numshare)])

    

    
    shared = np.random.choice([1,0], len(eshare), p=[eprob, 1 - eprob])
    

    # Different graphs if balanced clusters or not
    if len(clustSize) > 1:
        balanced = "_unbal"
    else:
        balanced = "balanced"

    # lower triangular(d=F) rows and cols indices to be used for subsetting below
    rows_l, cols_l = np.tril_indices(p, k=-1)

    #sum_offInds = int(sum(np.sum(np.tril(np.ones((p, p))), axis=1)))

    # Group level matrices
    while np.min(np.linalg.eigvals(Omega0s).min(axis=0)) <= 0:
        for g in range(0, G):
            g0s[g,:, :] = np.zeros((p, p))

            ##hubs
            if gtype == "hub":
                hubs = np.array_split(np.random.permutation(p), J)
                

                for h in range(0, J):
                    for v in hubs[h]:
                        g0s[g,:,:][hubs[h][0], v] = 1
                


            elif gtype == "random":
                ##using rows and cols of lower triangular matrix(d=F) to subset g0s
                rows_l, cols_l = np.tril_indices(p, k=-1)
                g0s[g,:,:][[rows_l], [cols_l]] = np.random.choice((1, 0), \
                                int(sum(np.sum(np.tril(np.ones((p, p))), axis=1))), p=(0.5, 0.5))
                

            # Adding in numShare shared edges
            for e in range(0, len(eshare)):
                g0s[g,:,:][eshare[e, 0], eshare[e, 1]] = shared[e]

            # Saving graphs to keep constant across simulations
            g0s[g,:,:] = (g0s[g,:,:] + np.transpose(g0s[g,:,:]) > 0.001) + 0

            # Making graph triangular for precision matrix generation and storing row edge count
            g0s[g,:,:] = (g0s[g,:,:] + np.transpose(g0s[g,:,:]) > 0.001) + 0
            rwSum = [np.sum(g0s[g,:,:], axis=0)]

            # Using upper triangular matrix(d = True) row and col indices tosubset g0s
            rows_ud, cols_ud = np.triu_indices(p, k=0)
            g0s[g,:,:][[rows_ud], [cols_ud]] = 0

            Omega0s[g,:,:] = np.multiply(g0s[g,:,:],np.random.uniform(0.50, 1.00, size=(p, p))\
                                         * np.random.choice([1, -1], size=(p, p), replace=True))
            

            if g > 0:  # python starts indexing at 0 compared to 1 in R
                for e in range(len(eshare)):
                    Omega0s[g,:,:][eshare[e, 0], eshare[e, 1]] = \
                                                 Omega0s[g-1,:,:][eshare[e, 0], eshare[e, 1]]
                

            # Making matrix symmetric and positive definite
            
            Omega0s[g,:,:] = symmPosDef(Omega0s[g,:,:])
            

            # Making graph full again, not just lower triangular
            g0s[g,:,:] = g0s[g,:,:] + np.transpose(g0s[g,:,:])
        
            
            

    while np.min(np.linalg.eigvals(Omegaks).min(axis=0)) <= 0:
        
        for k in range(K):
            # Creating subject-level graph to be exactly same as group-level graph for now
        
            # using lower triangular(d=F) rows and cols to subset gks
            rows_l2, cols_l2 = np.tril_indices(p, k=-1)
            gks[k,:,:][[rows_l2], [cols_l2]] = g0s[Zgks[k]-1] [[rows_l2], [cols_l2]]
            

            # Forcing subject-level matrix to have similar value as group-level matrix
                    
            Omegaks[k,:,:] = gks[k,:,:] * (Omega0s[Zgks[k]-1] + np.random.normal(scale=esd, size=(p, p)))

            

            # Changing edge presence for floor(rho * E) pairs of vertices from group-level graph
            if np.floor((rho * gks[k,:,:].sum())) > 0:

                swaps = np.array(np.tril_indices(p, k=-1)).T[np.random.randint(0, p * (p - 1) // 2,\
                                                                               size=int(np.floor(rho * gks[k,:,:].sum())))]
                print(len(swaps))
                for s in range(len(swaps)):
                    gks[k,:,:][swaps[s, 0], swaps[s, 1]] = abs(gks[k,:,:][swaps[s, 0], swaps[s, 1]] - 1)
                
                    if gks[k,:,:][swaps[s, 0], swaps[s, 1]] == 1:
                        Omegaks[k,:,:][swaps[s, 0], swaps[s, 1]] = np.multiply(np.random.uniform(low=0.5, high=1, size=1),
                                                                        np.random.choice((-1, 1), 1))
                    else:
                        Omegaks[k,:,:][swaps[s, 0], swaps[s, 1]] = 0
            
            
            # Making graph symmetric
            gks[k,:,:] = gks[k,:,:] + np.transpose(gks[k,:,:])
            
            

            # Making matrix symmetric and positive definite
            Omegaks[k,:,:] = symmPosDef(Omegaks[k,:,:])
            
            
            
        
    

    # Generating subject data
    simData = [np.random.multivariate_normal(mean=np.zeros(p), cov=np.linalg.inv(Omegaks[k,:, :]), size=n) for k in range(K)]
        
    # Centering generated data
    simData = [d - d.mean(axis=0) for d in simData]
    
    

    results = {"simDat": simData, "g0s": g0s, "Omega0s": Omega0s, "Omegaks": Omegaks, "zgks": zgks}

    return results















         
            
