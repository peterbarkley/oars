import numpy as np
from datetime import datetime
from oars.algorithms.cabra import getPA, getVar, getFeedersL
from oars.matrices.prebuilt import getCaraFull
from oars.matrices.core import ipf


def constantCaraAlgorithm(data, A, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=1.0, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward-forward-backward resolvent splitting algorithm in serial

    Args:
        data (list): list of :math:`n` initialization dictionaries for A, each of which contains a varlist key with a list of variable indexes as its value
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
        warmstartprimal (ndarray, optional): resolvent.shape ndarray for :math:`x` in v^0, or length :math:`p` list of such
        warmstartdual (ndarray, optional): n_k x resolvent.shape ndarray for :math:`u` which sums to 0 in v_k^0, or length :math:`p` list of such
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
    all_y = [np.array(0) for i in range(n)]
    # Warm start -- to do!
    # if warmstartprimal is None:
    #     all_v = np.zeros((n,) + m)
    # else:
    #     all_v = getWarmPrimal(warmstartprimal, Z)
    # if warmstartdual is not None:
    #     all_v += warmstartdual
    gammaW = [gamma*Wk for Wk in W]

    # Get feeders and weights
    PA = getPA([Ai.vars for Ai in A], p)
    fdr = getFeedersL(n, Z, PA)

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||') #\t||sum dual at x||
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            for k in A[i].vars:
                all_x[i][k] = all_v[i][k].copy()
                for (j, wt) in fdr[i][k]:
                    all_x[i][k] += all_x[j][k]*wt
            all_y[i] = np.array(list(all_x[i].values())).flatten()
            A[i].prox_step(all_x[i], alpha)
            
        if callback is not None and callback(itr, all_x, all_v, all_y): break

        if verbose and itr % checkperiod == 0:
            ysqdiff = 0.0
            for k in range(p):
                if len(PA[k]) > 1:
                    ybar = np.mean([all_x[i][k] for i in PA[k]], axis=0)
                    ysqdiff += sum(np.linalg.norm(all_x[i][k] - ybar)**2 for i in PA[k])
            # dualsum = np.linalg.norm(sum(getDualsCabra(all_v, all_x, all_b, Z)))
            # dualsum = 0.0
            print(f"{datetime.now()}\t{itr}\t{ysqdiff:.3e}") # \t{dualsum:.3e}

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

    return constantCaraAlgorithm(data, A, Zs, Zs, **kwargs)
        
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
    PA = getPA([di['varlist'] for di in data])
    Zs = []
    for varops in PA:
        nk = len(varops)
        if nk< 2:
            Zs.append(getCaraFull(len(varops)))
        else:
            A = np.zeros((nk, nk))
            for idx, i in enumerate(varops):
                for jdx, j in enumerate(varops):
                    if idx != jdx and mask[i,j] > 0.0:
                        A[idx, jdx] = 1.0
                        A[jdx, idx] = 1.0
            Zs.append(np.eye(nk) - ipf(A))
        assert np.linalg.eigvalsh(Zs[-1])[1] > 0.0

    return constantCaraAlgorithm(data, A, Zs, Zs, **kwargs)
        
