# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import eig
import math
from math import factorial
import random
from scipy.linalg import sqrtm

# Change the variable values to desired

G, clustSize, p, n, overlap = 2, (67, 37), 10, 177, 0.5
rho, esd, gtype, eprob = 0.10, 0.05, "hub", 0.5


def rccsim(G, clustSize, p, n, overlap, rho, esd, graphtype, eprob):
    # Calculating total number of subjects
    K = 0
    if len(clustSize) == 1:
        K += G * clustSize[0]
    else:
        K += sum(clustSize)
    # Cluster Networks --------------------------------------------------------
    g0s = np.zeros((p, p))
    gks = np.zeros((p, p))
    Omega0s = np.zeros((p, p))
    Omegaks = np.zeros((p, p))

    if len(clustSize) != 1 and len(clustSize) != G:

        print("clustSize must be of length 1 or of length equal to the number of clusters")

    else:

        if (len(clustSize) > 1):

            Zgks = []
            for g in range(len(clustSize)):
                zgks = list(np.repeat(g + 1, clustSize[g]))
                Zgks.extend(zgks)

        else:
            for i in range(G):
                zgks.repeat(1 + G, clustSize[0])
    Zgks = np.array(Zgks)
    simDat = list()

    # Manually generating cluster-level graphs and precision matrices

    # Number of hubs
    J = math.floor(math.sqrt(p))

    def symmPosDef(m):  # m is a numpy array
        m = m + np.transpose(m)
        va, ve = eig(m)
        smallE = min(va)
        if smallE <= 0:
            nrows, ncols = m.shape
            v = np.repeat(abs(smallE) + 0.10 + 0.10, nrows)
            m = m + np.diag(v)
        return m

    # Determining edges to be shared across groups
    numE = p - J
    q = factorial(p) / (factorial(2) * factorial(p - 2))
    if gtype == "hub":
        numshare = math.floor(numE * overlap)
    else:
        numshare = math.floor(q * overlap)

    # I will reuse the indices from the below operation for any lower triangular matrix indices(d=False)
    ones_matrix = np.ones((p, p))
    m_lower = np.tril(ones_matrix, k=-1)
    m_bool = m_lower.astype(bool)
    r, c = m_bool.shape
    indices_l = []
    ##eshare
    for x in range(r):
        for y in range(c):
            if m_bool[x, y] == True:
                indices_l.append((x, y))
    random_sample = [np.random.randint(0, q) for i in range(numshare)]
    matrix = [indices_l[x] for x in random_sample]
    eshare = np.array(matrix)
    # print(eshare)

    ##shared
    numberlist = [1, 0]
    shared = np.random.choice(numberlist, len(eshare), p=[eprob, 1 - eprob])

    # Different graphs if balanced clusters or not
    if len(clustSize) > 1:
        balanced = "_unbal"
    else:
        balanced = "balanced"

    # extracting lower triangular(d=F) rows and cols indices to be used for subsetting
    rows_l = []
    cols_l = []
    for i in indices_l:
        rows_l.append(i[0])
        cols_l.append(i[1])

    sum_offInds = int(sum(np.sum(m_lower, axis=1)))
    # print(sum_offInds)

    # Group level matrices
    val = eig(Omega0s)[0]
    eig_valuesGs = []
    min_eigvalue = min(val)
    while min_eigvalue <= 0:
        g0s_list = []
        Omega0s_list = []
        Omega0s_unsymmPosDef_list = []
        for g in range(0, G):

            g0s = np.zeros((p, p))
            ##hubs
            if gtype == "hub":
                data = random.sample(range(0, p), p)
                d = np.array_split(np.array(data), J)
                # print(d)
                hubs = [d[i] for i in range(len(d))]
                # print(hubs[0])

                for h in range(0, J):
                    for v in hubs[h]:
                        g0s[hubs[h][1], v] = 1
                # print(g0s)


            elif gtype == "random":
                ##uses rws and cols for lower triangular matrix mentioned before
                g0s[[rows_l], [cols_l]] = np.random.choice((1, 0), sum_offInds, p=(0.5, 0.5))
                # print(g0s)

            # Adding in numShare shared edges
            for e in range(0, len(eshare)):
                g0s[eshare[e, 0], eshare[e, 1]] = shared[e]

            # Saving graphs to keep constant across simulations
            g0s = (g0s + np.transpose(g0s) > 0.001) + 0

            # Making graph triangular for precision matrix generation and storing row edge count
            g0s = (g0s + np.transpose(g0s) > 0.001) + 0
            rwSum = [np.sum(g0s, axis=0)]

            # Creating upper triangular matrix(d = True) to get the indices to be used for subsetting
            matrix_u = np.ones((p, p))
            ud_matrix = np.triu(matrix_u, k=0)
            u_bool = ud_matrix.astype(bool)
            nrows, ncols = u_bool.shape
            indices_ud = []
            for x in range(nrows):
                for y in range(ncols):
                    if u_bool[x, y] == True:
                        indices_ud.append((x, y))

            rows_ud = []
            cols_ud = []
            for i in indices_ud:
                rows_ud.append(i[0])
                cols_ud.append(i[1])
            g0s[[rows_ud], [cols_ud]] = 0

            unif_s = np.random.uniform(low=0.5, high=1, size=p * p)
            rand_s = np.random.choice((-1, 1), p * p)
            product = np.multiply(unif_s, rand_s)
            product = product.reshape(1, p * p)
            product_to_list = product.tolist()
            product_list = product_to_list[0]
            Omega0s = np.multiply(g0s, np.random.choice(product_list, size=(p, p)))
            # print(Omega0s)

            if g > 0:  # python starts indexing at 0 compared to 1 in R
                Omega0s_g_previous = Omega0s_unsymmPosDef_list[-1]

                print(Omega0s_g_previous)
                for e in range(len(eshare)):
                    Omega0s[eshare[e, 0], eshare[e, 1]] = Omega0s_g_previous[eshare[e, 0], eshare[e, 1]]
                # print(Omega0s)

            # Making matrix symmetric and positive definite
            Omega0s_unsymmPosDef_list.append(Omega0s)
            Omega0s_s = symmPosDef(Omega0s)
            # print(Omega0s_s)

            # Making graph full again, not just lower triangular
            g0s = g0s + np.transpose(g0s)
            # print(g0s)
            g0s_list.append(g0s)
            # print(g0s_list)
            Omega0s_list.append(Omega0s_s)
            Omega0s = Omega0s_s
            value = eig(Omega0s)[0]
            value = value.tolist()
            eig_valuesGs.extend(value)
        # print(eig_values)
        min_eigvalue = min(eig_valuesGs)  # to update the while loop
        # print(Omega0s_list)
        print(Omega0s_unsymmPosDef_list)

    vl = eig(Omegaks)[0]
    eig_valuesKs = []
    min_eigenvalue = min(vl)
    while min_eigenvalue <= 0:
        # print(min_eigenvalue)

        # print(K)
        # print(Zgks)
        gks_list = []
        Omegaks_list = []
        for i in range(K):
            # Creating subject-level graph to be exactly same as group-level graph for now
            zind = Zgks[i] - 1
            # print(zind)
            gks = np.zeros((p, p))
            g0s = g0s_list[zind]
            Omega0s = Omega0s_list[zind]
            # print(g0s)

            # Also uses lower triangular(d=F) rows and cols mentioned above
            gks[[rows_l], [cols_l]] = g0s[[rows_l], [cols_l]]
            # print(gks)

            # Forcing subject-level matrix to have similar value as group-level matrix
            rnorm = np.random.normal(0, esd, p * p)
            rnorm_matrix = np.random.choice(rnorm, size=(p, p))
            # print(r.shape)

            Omegaks = np.add((gks * Omega0s), rnorm_matrix)
            # print(Omegaks)

            # Changing edge presence for floor(rho * E) pairs of vertices from group-level graph
            x = math.floor((rho * gks.sum()))
            if x > 0:

                # print(x)
                random_sample_2 = [np.random.randint(0, (p * (p - 1) / 2)) for i in range(x)]
                # print(random_sample_2)
                matrix_vals = [indices_l[x] for x in random_sample_2]
                # print(matrix_vals)
                swaps = np.array(matrix_vals)
                # print(swaps)
                for s in range(len(swaps)):
                    gks[swaps[s, 0], swaps[s, 1]] = abs(gks[swaps[s, 0], swaps[s, 1]] - 1)
                    # print(gks)
                    if gks[swaps[s, 0], swaps[s, 1]] == 1:
                        Omegaks[swaps[s, 0], swaps[s, 1]] = np.multiply(np.random.uniform(low=0.5, high=1, size=1),
                                                                        np.random.choice((-1, 1), 1))
                    else:
                        Omegaks[swaps[s, 0], swaps[s, 1]] = 0
            # print(Omegaks)
            # Making graph symmetric
            gks = gks + np.transpose(gks)
            # print(gks)
            gks_list.append(gks)

            # Making matrix symmetric and positive definite
            Omegaks_symm = symmPosDef(Omegaks)
            # print(Omegaks)
            Omegaks_list.append(Omegaks_symm)
            eigenvalues = eig(Omegaks_symm)[0]
            eigenvalues = eigenvalues.tolist()
            eig_valuesKs.extend(eigenvalues)
        # print(len(eig_valuesKs))
        min_eigenvalue = min(eig_valuesKs)  # to update the while loop
    # print(len(gks_list))
    # print(len(Omegaks_list))

    # Generating and centering subject data
    for k in range(K):
        omegaks_k = Omegaks_list[i]
        means = np.zeros((p))
        obs = np.random.multivariate_normal(means, np.eye(p),
        n, check_valid = 'warn')
        # print(obs.shape)

        sqrtcovs = np.array(sqrtm(np.cov(omegaks_k)))
        v = np.einsum('ij,kj->ij', obs, sqrtcovs)
        m = np.expand_dims(means, axis=0)
        t = v + m
        # print(t.shape)
        simDat.append(t)
    # print(len(simDat))
    # print(len(simDat[0]))

    results = (simDat, g0s_list, Omega0s_list, gks_list, Omegaks_list, Zgks)
    return results


sim = rccsim(G, clustSize, p, n, overlap, rho, esd, gtype, eprob)

simData = sim[0]
g0s = sim[1]
Omega0s = sim[2]
gks = sim[3]
Omegaks = sim[4]
Zgks = sim[5]

# Uncomment below to assess each of the outputs

# print(len(g0s))
# print(len(simData))
# print(len(simData[0]))
# print(len(Omega0s))
# print(len(gks))
# print(len(Omegaks))
# print(len(Zgks))












