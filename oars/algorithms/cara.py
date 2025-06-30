import numpy as np
from datetime import datetime
from oars.algorithms.cabra import getPA
from oars.matrices.prebuilt import getCaraFull
from oars.matrices.core import ipf

def getVar(data):
    data['indices'] = {}
    start = 0
    for k, klength in zip(data['varlist'], data['varshapes']):
        stop = start+klength
        data['indices'][k] = np.arange(start, stop)
        start = stop
    return np.zeros(stop)

def getFeedersL(Z, PA, A):
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
                wts = np.array([-2.0*Z[k][PA[k].index(i), PA[k].index(j)] for k in sharedk for idx in A[i].indices[k]])
                fdrs[i].append((j, i_idxs, j_idxs, wts))
    return fdrs

def getFeedersW(W, PA, A):
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
                wts = np.array([W[k][PA[k].index(i), PA[k].index(j)] for k in sharedk for idx in A[i].indices[k]])
                fdrs[i].append((j, i_idxs, j_idxs, wts))
    return fdrs

def getFullVariable(x, A, PA):
    y = []
    for k, PAk in enumerate(PA):
        if len(PAk) > 0:
            ybar = np.mean([x[i][A[i].indices[k]] for i in PAk])
        else:
            ybar = 0.0
        y.append(ybar)
    return y

def caraAlgorithm(data, A, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=1.0, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward-forward-backward resolvent splitting algorithm in serial

    Args:
        data (list): list of :math:`n` initialization dictionaries for A, each of which contains a varlist key with a list of variable indexes as its value
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
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
    p = len(Z)

    # Initialize the variables
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
    PA = getPA([Ai.vars for Ai in A], p)
    fdr = getFeedersL(Z, PA, A)
    wfdr = getFeedersW(gammaW, PA, A)

    # Warm start primal
    if warmstartprimal is not None:
        for k, v in warmstartprimal.items():
            for idx, i in enumerate(PA[k]):
                all_v[i][A[i].indices[k]] = (1.0 + 2.0*np.sum(Z[k][idx,:idx]))*v

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(nn):
            all_x[i] = all_v[i].copy()
            for (j, i_idxs, j_idxs, wt) in fdr[i]:
                all_x[i][i_idxs] += all_x[j][j_idxs]*wt
            if verbose or callback is not None:
                all_y[i] = all_x[i].copy()
            all_x[i] = A[i].prox(all_x[i], alpha)
            
        if callback is not None and callback(itr, all_x, all_v, all_y, A): break

        if verbose and itr % checkperiod == 0:
            ysqdiff = 0.0
            for k in range(p):
                if len(PA[k]) > 1:
                    ybar = np.mean([all_x[i][A[i].indices[k]] for i in PA[k]], axis=0)
                    ysqdiff += sum(np.linalg.norm(all_x[i][A[i].indices[k]] - ybar)**2 for i in PA[k])
            subg_sum_norm = sum([np.linalg.norm(sum([all_y[i][A[i].indices[k]]-all_x[i][A[i].indices[k]] for i in PA[k]]))**2 for k in range(p)])**0.5
            print(f"{datetime.now()}\t{itr}\t{ysqdiff**0.5:.3e}\t{subg_sum_norm:.3e}")

        for i in range(nn):
            for (j, i_idxs, j_idxs, wt) in wfdr[i]:
                all_v[i][i_idxs] -= wt*all_x[j][j_idxs]

        
    ybar = getFullVariable(all_x, A, PA)
    
    # Build logs list
    logs = []
    for i in range(nn):
        if hasattr(A[i], 'log'):
            logs.append(A[i].log)
        else:
            logs.append([])

    return ybar, logs, all_x, all_v


def constantCaraAlgorithm(data, A, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=1.0, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward-forward-backward resolvent splitting algorithm in serial

    Args:
        data (list): list of :math:`n` initialization dictionaries for A, each of which contains a varlist key with a list of variable indexes as its value
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
        warmstartprimal (ndarray, optional): dictionary with :math:`p` integer subvector ids as keys and primal estimate ndarrays as the value 
        warmstartdual (ndarray, optional): list of length :math:`n` giving a dictionary for each resolvent with keys for each subvector id pertaining to that resolvent and values giving the subgradient estimate for that subvector in that resolvent. The sum of the subgradients over the resolvents for each subvector must be zero.
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (ndarray): resolvent.shape ndarray of the mean over the node solutions at termination
        logs (list): list of n+m logs for the operators
        all_x (ndarray): n x resolvent.shape ndarray of the node solution
        all_v (ndarray): n x resolvent.shape ndarray of the consensus iterates at solution

    Examples:
    """
    # Initialize the operators
    n = len(A)
    p = len(Z)
    for i in range(n):
        A[i] = A[i](**data[i])

    # Initialize the variables
    all_x = [getVar(A[i]) for i in range(n)] # length n list of length k_i \\leq p dict of ndarrays
    all_v = [getVar(A[i]) for i in range(n)] # length n list of length k_i \\leq p dict of ndarrays
    if verbose or callback is not None:
        all_y = all_x.copy()
    gammaW = [gamma*Wk for Wk in W]

    # Get feeders and weights
    PA = getPA([Ai.vars for Ai in A], p)
    fdr = getFeedersL(n, Z, PA)

    # Warm start -- to do!
    if warmstartprimal is not None:
        for k, v in warmstartprimal.items():
            for idx, i in enumerate(PA[k]):
                all_v[i][k] = (1.0 + 2.0*np.sum(Z[k][idx,:idx]))*v
    if warmstartdual is not None:
        for i in range(n):
            for k in A[i].vars:
                all_v[i][k] += warmstartdual[i][k]

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            for k in A[i].vars:
                all_x[i][k] = all_v[i][k].copy()
                for (j, wt) in fdr[i][k]:
                    all_x[i][k] += all_x[j][k]*wt
            if verbose or callback is not None:
                all_y[i] = all_x[i].copy()
            A[i].prox_step(all_x[i], alpha)
            
        if callback is not None and callback(itr, all_x, all_v, all_y): break

        if verbose and itr % checkperiod == 0:
            ysqdiff = 0.0
            for k in range(p):
                if len(PA[k]) > 1:
                    ybar = np.mean([all_x[i][k] for i in PA[k]], axis=0)
                    ysqdiff += sum(np.linalg.norm(all_x[i][k] - ybar)**2 for i in PA[k])
            subg_sum_norm = sum([sum([all_y[i][k]-all_x[i][k] for i in PA[k]])**2 for k in range(p)])**0.5
            print(f"{datetime.now()}\t{itr}\t{ysqdiff**0.5:.3e}\t{subg_sum_norm:.3e}")

        for i in range(n):
            for k in A[i].vars:
                idx = PA[k].index(i)
                all_v[i][k] -= sum(gammaW[k][idx, jdx]*all_x[j][k] for jdx, j in enumerate(PA[k]))

        
    ybar = [np.mean([all_x[i][k] for i in PA[k]], axis=0) if len(PA[k]) > 0 else np.array([0.0]) for k in range(p) ]
    
    # Build logs list
    logs = []
    for i in range(n):
        if hasattr(A[i], 'log'):
            logs.append(A[i].log)
        else:
            logs.append([])

    return ybar, logs, all_x, all_v


def easyCara(data, A, **kwargs):
    '''
    Run CABRA with full matrices built from the varlists of the proxes

     
    Args:
        data (list): list of :math:`n` initialization dictionaries for A, each of which contains a varlist key with a list of variable indexes as its value
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        **kwargs: keyword arguments for the algorithm

    Returns:

    Examples:

    '''
    PA = getPA([di['varlist'] for di in data])
    # for varops in PA:
    #     assert len(varops) > 1
    Zs = [getCaraFull(len(varops)) for varops in PA]

    return caraAlgorithm(data, A, Zs, Zs, **kwargs)
        
def easyCaraGraph(data, A, mask, **kwargs):
    '''
    Run CABRA with full matrices built from the varlists of the proxes

     
    Args:
        data (list): list of :math:`n` initialization dictionaries for A, each of which contains a varlist key with a list of variable indexes as its value
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        mask (array): n x n numpy array with 1.0 in permissible connections and 0.0 in impermissible connections
        **kwargs: keyword arguments for the algorithm

    Returns:

    Examples:

    '''
    Zs = getZs(data, mask)

    return caraAlgorithm(data, A, Zs, Zs, **kwargs)
        
def getZs(data, mask):
    PA = getPA([di['varlist'] for di in data])
    Zs = []
    for varops in PA:
        nk = len(varops)
        if nk< 2:
            Zs.append(getCaraFull(len(varops)))
        else:
            AA = np.zeros((nk, nk))
            for idx, i in enumerate(varops):
                for jdx, j in enumerate(varops):
                    if idx != jdx and mask[i,j] > 0.0:
                        AA[idx, jdx] = 1.0
                        AA[jdx, idx] = 1.0
            Zs.append(np.eye(nk) - ipf(AA))
            assert np.linalg.eigvalsh(Zs[-1])[1] > 0.0
    return Zs
