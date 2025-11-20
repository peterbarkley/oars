import numpy as np
from datetime import datetime
from oars.algorithms.cabra import getPA, permute
from oars.matrices.prebuilt import getCaraFull
from oars.matrices.core import ipf

def getVar(data):
    data['indices'] = {}
    start = 0
    for k, klength in zip(data['varlist'], data['varshapes']):
        stop = start+klength
        data['indices'][k] = range(start, stop)
        start = stop
    return np.zeros(stop)

def getPermutedDiagonal(n, Z, PA):
    """
    Returns a permuted list taking the diagonals of the p Z_k entries in Z and returning them as a 
    list of n vectors where PA[k] gives the ordered list of the matrices in n which use entry k

    Args:
        n (int): number of A operators
        Z (list): list of :math:`p` Zk matrices with diagonal Dk
        PA (list): list of :math:`p` ordered lists of operators which use variable k
    """
    DA = [[] for _ in range(n)]
    for k, pk in enumerate(PA):
        for idx, i in enumerate(pk):
            DA[i].append(Z[k][idx, idx])
    DA = [np.array(D) for D in DA]
    return DA

def getFeedersL(Z, PA, A, p):
    """
    Return a length len(A) list of lists where the fdr[i] contains entries for j < i such that j and i share some k and Z[k][s(i,k), s(j,k)] != 0
    Entries are of the form (j, i_idxs, j_idxs, wt) where j is the index of the other operator, i_idxs gives the indices in x[i] of the shared variables, j_idxs gives the indices in x[j] of the shared variables, and wt gives the set of weights -2*Z[k][s(i,k), s(j,k)] of the appropriate varlength for the ordered shared variables k 
    """
    n = len(A)
    fdrs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            sharedk = set(A[i].vars) & set(A[j].vars)
            nonzerok = {k for k in sharedk if not np.isclose(Z[k][PA[k].index(i), PA[k].index(j)], 0.0)}
            sharedk = sorted(sharedk & nonzerok)
            if len(sharedk) > 0:
                i_idxs = [idx for k in sharedk for idx in A[i].indices[k]]
                j_idxs = [idx for k in sharedk for idx in A[j].indices[k]]
                wts = np.array([-2.0*p[k][PA[k].index(j)]*Z[k][PA[k].index(i), PA[k].index(j)] for k in sharedk for idx in A[i].indices[k]])
                fdrs[i].append((j, i_idxs, j_idxs, wts))
    return fdrs

def getFeedersW(W, PA, A, p):
    """
    Return a length len(A) list of lists where the fdr[i] contains entries such that j and i share some k and W[k][s(i,k), s(j,k)] != 0
    Entries are of the form (j, i_idxs, j_idxs, wt) where j is the index of the operator, i_idxs gives the indices in x[i] of the shared variables, j_idxs gives the indices in x[j] of the shared variables, and wt gives the set of weights W[k][s(i,k), s(j,k)] of the appropriate varlength for the ordered shared variables k 
    """
    n = len(A)
    fdrs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sharedk = set(A[i].vars) & set(A[j].vars)
            nonzerok = {k for k in sharedk if not np.isclose(W[k][PA[k].index(i), PA[k].index(j)], 0.0)}
            sharedk = sorted(sharedk & nonzerok)
            if len(sharedk) > 0:
                i_idxs = [idx for k in sharedk for idx in A[i].indices[k]]
                j_idxs = [idx for k in sharedk for idx in A[j].indices[k]]
                wts = np.array([p[k][PA[k].index(j)]*W[k][PA[k].index(i), PA[k].index(j)] for k in sharedk for idx in A[i].indices[k]])
                fdrs[i].append((j, i_idxs, j_idxs, wts))
    return fdrs

def getFullVariable(x, A, PA):
    y = []
    for k, PAk in enumerate(PA):
        if len(PAk) > 0:
            ybar = np.mean([x[i][A[i].indices[k]] for i in PAk], axis=0)
        else:
            ybar = 0.0
        y.append(ybar)
    return y

def redPhAlgorithm(p, data, A, W, Z, I, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=1.0, alpha=1.0, verbose=False, callback=None):
    """
    Run the adaptive reduced PH splitting algorithm in serial

    Args:
        p (list): list of :math:`p` weight vectors
        data (list): list of :math:`n` initialization dictionaries for A, each of which contains a varlist key with a list of variable indexes as its value
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
        I (list): list of :math:`p` lists giving the functions which use each variable
        warmstartprimal (dictionary, optional): dictionary with :math:`p` integer subvector ids as keys and primal estimate ndarrays as the value 
        warmstartdual (list, optional): list of length :math:`n` giving a dictionary for each resolvent with keys for each subvector id pertaining to that resolvent and values giving the subgradient estimate for that subvector in that resolvent. The sum of the subgradients over the resolvents for each subvector must be zero.
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in (0,2) for :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the positive resolvent step size in :math:`x^{k+1} = J_{\\alpha A_i}(y^k)`
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (list): list of :math:`p` mean values of the subvectors over the node solutions at termination
        logs (list): list of n logs for the operators
        all_x (list): list of :math:`n` ndarrays of the node solutions
        all_v (list): list of :math:`n` ndarrays of the node consensus iterates at solution

    Examples:
    """
    # Initialize the operators
    nn = len(data)
    pp = len(Z)

    # Initialize the variables
    Ds = getPermutedDiagonal(nn, Z, I)
    for i in range(nn):
        data[i]['D'] = np.array([p[k][I[k].index(i)] for k in data[i]['varlist']])*Ds[i]
    all_x = [getVar(data[i]) for i in range(nn)]
    if warmstartdual is not None:
        all_v = warmstartdual
    else:
        all_v = [all_x[i].copy() for i in range(nn)]
    if verbose or callback is not None:
        all_y = [all_x[i].copy() for i in range(nn)]

    for i in range(nn):
        A[i] = A[i](**data[i])

    # Get feeders and weights
    gammaW = [gamma*Wk for Wk in W]
    # PA = getPA([Ai.vars for Ai in A], p)
    fdr = getFeedersL(Z, I, A, p)
    wfdr = getFeedersW(gammaW, I, A, p)

    # Warm start primal
    if warmstartprimal is not None:
        print('Not implemented!')
        return 0
        # for k, v in warmstartprimal.items():
        #     for idx, i in enumerate(PA[k]):
        #         all_v[i][A[i].indices[k]] = (1.0 + 2.0*np.sum(Z[k][idx,:idx]))*v

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||x-bar(x)||_Q\t||sum dual||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(nn):
            np.copyto(all_x[i],all_v[i])
            for (j, i_idxs, j_idxs, wt) in fdr[i]:
                all_x[i][i_idxs] += all_x[j][j_idxs]*wt
            if verbose or callback is not None:
                np.copyto(all_y[i],all_x[i])
            all_x[i] = A[i].prox(all_x[i], alpha)
            
        if callback is not None and callback(itr, all_x, all_v, all_y, A): break

        if verbose and (itr+1) % checkperiod == 0:
            ysqdiff = 0.0
            wt_sqdiff = 0.0
            for k in range(pp):
                if len(I[k]) > 1:
                    ybar = np.mean([all_x[i][A[i].indices[k]] for i in I[k]], axis=0)
                    ysqdiff += sum(np.linalg.norm(all_x[i][A[i].indices[k]] - ybar)**2 for i in I[k])
                    wt_sqdiff +=sum(p[k][I[k].index(i)]*np.linalg.norm(all_x[i][A[i].indices[k]] - ybar)**2 for i in I[k])
            subg_sum_norm = sum([np.linalg.norm(sum([p[k][I[k].index(i)]*(all_y[i][A[i].indices[k]]-p[k][I[k].index(i)]*Z[k][I[k].index(i), I[k].index(i)]*all_x[i][A[i].indices[k]]) for i in I[k]]))**2 for k in range(pp)])**0.5
            print(f"{datetime.now()}\t{itr}\t{ysqdiff**0.5:.3e}\t{wt_sqdiff**0.5:.3e}\t{subg_sum_norm:.3e}")

        for i in range(nn):
            for (j, i_idxs, j_idxs, wt) in wfdr[i]:
                all_v[i][i_idxs] -= wt*all_x[j][j_idxs]

        
    ybar = getFullVariable(all_x, A, I)
    
    # Build logs list
    logs = []
    for i in range(nn):
        if hasattr(A[i], 'log'):
            logs.append(A[i].log)
        else:
            logs.append([])

    return ybar, logs, all_x, all_v

