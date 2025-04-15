import numpy as np
from datetime import datetime

def getBorder(A, B, K, PA, PB):
    """
    Get the set of B operators which can be calculated after each A

    Args:
        A (list): list of A operators
        B (list): list of B operators
        K (list): list of :math:`p` submatrices Kk which provide weights for A operator output to B
        PA (list): list of :math:`p` lists where PA[k] has an ordered list of A operators using variable k
        PB (list): list of :math:`p` lists where PB[k] has an ordered list of B operators using variable k

    """
    n = len(A)
    m = len(B)

    # executableBs[i] give the list of B that can be executed after A operator i and before op j
    executableBs = [[] for _ in range(n)]
    
    # loop over B operators
    for j in range(m):
        ibar = 0 # index of last required A operator
        for k in B[j].vars: # loop through variables used by j
            jdx = PB[k].index(j)

            # find last nonzero column 
            ibk = len(K[k][jdx]) - 1 
            for Kk_ji in K[k][jdx][::-1]: 
                if not np.isclose(Kk_ji, 0.0):
                    break
                ibk -= 1
            # find operator index corresponding to ibk
            i = PA[k][ibk]
            ibar = max(ibar, i)
        executableBs[ibar].append(j)

    return executableBs

def getFeedersK(B, K, PA, PB):
    """
    Returns a list of dictionaries such that 
    fdr[j] = {k_1: [(i_1, (K_k)_{ji_1}), (i_2, (L_k)_{ji_2})], ...}
    where i_1, i_2, etc, are A operators which hold variable k and feed their output to B operator j,
    and (K_k)_{ji} gives the weight in K_k for ji
    """
    m = len(B)
    fdr = [{} for j in range(m)]
    for j in range(m):
        for k in B[j].vars:
            jdx = PB[k].index(j)
            fdr[j][k] = []
            for idx, wt in enumerate(K[k][jdx]):
                if not np.isclose(wt, 0.0):
                    fdr[j][k].append((PA[k][idx], wt))

    return fdr

def getFeedersQ(A, Q, PA, PB):
    """
    Return a list of dictionaries such that
    fdr[i] = {k_1: [(j_1, (Q_k)_{ij_1}), (j_2, (Q_k)_{ij_2})], ...}
    where j_1, j_2, etc, are B operators which hold variable k and feed their output to operator i,
    and (Q_k)_{ij} gives the weight in Z_k for ij

    
    Args:
        A (list): list of A operators
        Q (list): list of :math:`p` submatrices in which each column sums to one and the rows correspond to the ordered set of A operators using variable k and the columns correspond to the ordered set of B operators supplying variable k
        PA (list): list of :math:`p` lists where PA[k] has an ordered list of A operators using variable k
        PB (list): list of :math:`p` lists where PB[k] has an ordered list of B operators using variable k
    """
    n = len(A)
    fdr = [{} for i in range(n)]
    for i in range(n):
        for k in A[i].vars:
            idx = PA[k].index(i)
            fdr[i][k] = []
            if len(Q[k]) > 0:
                for qdx, wt in enumerate(Q[k][idx]):
                    if not np.isclose(wt, 0.0):
                        fdr[i][k].append((PB[k][qdx], wt))

    return fdr

def getVar(op):
    """
    Return the variable (or list of variables? from operator A)
    """
    if isinstance(op.vars, list):
        return {k: np.zeros(shape) for k, shape in zip(op.vars, op.varshapes)}
    elif isinstance(op.shape, tuple):
        return {0:np.zeros(op.shape)}
    
def getFeedersL(n, Z, P):
    """
    Get a list of dictionaries such that
    fdr[i] = {k_1: [(j_1, (L_k)_{ij_1}), (j_2, (L_k)_{ij_2})], ...}
    where j_1, j_2, etc, are previous operators which hold variable k and feed their output to operator i,
    and (L_k)_{ij} gives the negative of weight in Z_k for ij

    
    Args:
        n (int): number of operators
        X (list): list of :math:`p` submatrices
        P (list): list of :math:`p` lists where P[k] has an ordered list of operators using variable k
    """
    fdr = [{} for i in range(n)]
    for k, pk in enumerate(P):
        for idx, i in enumerate(pk):
            fdr[i][k] = []
            for jdx, j in enumerate(pk[:idx]):
                if not np.isclose(Z[k][idx, jdx], 0.0):
                    fdr[i][k].append((j, -2.0*Z[k][idx, jdx]))

    return fdr


def getDA(n, Z, PA):
    """
    Returns a list with the ordered list of diagonal elements for each A operator

    Args:
        n (int): number of A operators
        Z (list): list of :math:`p` Zk matrices with diagonal Dk
        PA (list): list of :math:`p` ordered lists of operators which use variable k
    """
    DA = [[] for _ in range(n)]
    for k, pk in enumerate(PA):
        for idx, i in enumerate(pk):
            DA[i].append(Z[k][idx, idx])
    DA = [np.diag(D) for D in DA]
    return DA

def cabraAlgorithm(A, B, W, Z, K, Q, PA, PB, data, varshapes, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward-forward-backward resolvent splitting algorithm in serial

    Args:
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        B (list): list of :math:`m` initializable cocoercive operators callable via a grad function
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
        K (list): list of :math:`p` within-iteration A->B coordination ndarrays
        Q (list): list of :math:`p` within-iteration B->A coordination ndarrays
        PA (list): list of :math:`p` lists where PA[k] has an ordered list of operators in A using variable k
        PB (list): list of :math:`p` lists where PB[k] has an ordered list of operators in B using variable k
        data (list): list of lists where data[0] has a list of :math:`n` initialization dictionaries for A, and data[1] contains the :math:`m` initialization dictionaries for B
        varshapes (list): list of :math:`p` tuples containing the shape of the ndarrays for the variables
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
    m = len(B)
    p = len(PA)
    for i in range(n):
        A[i] = A[i](**data[0][i])

    for j in range(m):
        B[j] = B[j](**data[1][j])

    # Initialize the variables
    all_x = [getVar(A[i]) for i in range(n)] # length n list of length k_i \\leq p dict of ndarrays
    all_v = [getVar(A[i]) for i in range(n)] # length n list of length k_i \\leq p dict of ndarrays
    all_b = [getVar(B[j]) for j in range(m)] # length m list of length k_j \\leq p dict of ndarrays

    # Warm start -- to do!
    # if warmstartprimal is None:
    #     all_v = np.zeros((n,) + m)
    # else:
    #     all_v = getWarmPrimal(warmstartprimal, Z)
    # if warmstartdual is not None:
    #     all_v += warmstartdual
    gammaW = [gamma*Wk for Wk in W]

    # Get feeders and weights
    fdr = getFeedersL(n, Z, PA)

    # print('Lfeeders')
    # for i in range(n):
    #     print(i, fdr[i])
    Q_fdr = getFeedersQ(A, Q, PA, PB)
    print('qfeeders')
    for i in range(n):
        print(i, Q_fdr[i])
    K_fdr = getFeedersK(B, K, PA, PB)
    print('Kfeeders')
    for j in range(m):
        print(j, K_fdr[j])

    # Get B calculation order
    Bready = getBorder(A, B, K, PA, PB)
    print('Bready')
    for i in range(n):
        print(i, Bready[i])

    # Get D_A
    DA = getDA(n, Z, PA)
    # print('DA')
    # for i in range(n):
    #     print(i, DA[i])

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual at x||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            for k in A[i].vars:
                all_x[i][k] = all_v[i][k].copy()
                for (j, wt) in fdr[i][k]:
                    all_x[i][k] += all_x[j][k]*wt
                for (j, wt) in Q_fdr[i][k]:
                    all_x[i][k] -= all_b[j][k]*wt
                # all_x[i][k] *= DA_inv[i][k]
            A[i].prox_step(all_x[i], alpha, DA[i])
            for j in Bready[i]:
                # Build argument for B.grad
                for k in B[j].vars:
                    b_input_k = sum(all_x[j][k]*wt for j,wt in K_fdr[j][k])
                    all_b[j][k] = b_input_k
                
                print('before', i, j, all_b[j])
                B[j].grad(all_b[j])
                
                print('after', i, j, all_b[j])
                for k in B[j].vars:
                    all_b[j][k] *= alpha
            
        if callback is not None and callback(itr, all_x, all_v, all_b): break

        if verbose and itr % checkperiod == 0:
            ybar = [np.mean([all_x[i][k] for i in PA[k]]) for k in range(p)]
            ysqdiff = [sum((all_x[i][k] - ybar[k])**2 for i in PA[k]) for k in range(p)]
            # dualsum = np.linalg.norm(sum(getDualsCabra(all_v, all_x, all_b, Z)))
            dualsum = 0.0
            print(f"{datetime.now()}\t{itr}\t{sum(ysqdiff)[0]:.3e}\t{dualsum:.3e}")

        for i in range(n):
            for k in A[i].vars:
                idx = PA[k].index(i)
                all_v[i][k] -= sum(gammaW[k][idx, jdx]*all_x[j][k] for jdx, j in enumerate(PA[k]))

        
    ybar = [np.mean([all_x[i][k] for i in PA[k]]) for k in range(p)]
    
    # Build logs list
    logs = []
    for i in range(n):
        if hasattr(A[i], 'log'):
            logs.append(A[i].log)
        else:
            logs.append([])
    for j in range(m):
        if hasattr(B[j], 'log'):
            logs.append(B[j].log)
        else:
            logs.append([])

    return ybar, logs, all_x, all_v