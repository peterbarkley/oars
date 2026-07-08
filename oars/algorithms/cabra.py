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
    fdr = [{} for _ in range(m)]
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
            if Q is not None and len(Q) > 0 and Q[k] is not None and len(Q[k]) > 0:
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

def getPA(varlist, p=None):
    """
    Use the variables assigned to each operator to build the 
    set of operators assigned to each variable

    Args:
        varlist (list): list of :math:`n` operators with a vars attribute giving the ordered variable for each operator
        p (int): maximum variable index

    Returns:
        PA (list): list of :math:`p` lists with the ordered operators for each variable 
    """
    n = len(varlist)
    if p is None:
        p = 0
        for i in range(n):
            p = max(p, max(varlist[i]))
        p += 1
    PA = [[] for _ in range(p)]
    for i in range(n):
        for k in varlist[i]:
            PA[k].append(i)

    return PA

def permute(n, Z, PA):
    """
    Returns a permuted list of lists taking the diagonals of the p entries in Z and returning them as a 
    list of n diagonal matrices where PA[k] gives the ordered list of the matrices in n which use entry k

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

def cabraAlgorithm(data, A, B, W, Z, K=None, Q=None, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward-forward-backward resolvent splitting algorithm in serial

    Args:
        data (list): list of lists where data[0] has a list of :math:`n` initialization dictionaries for A, and data[1] contains the :math:`m` initialization dictionaries for B
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        B (list): list of :math:`m` initializable cocoercive operators callable via a grad function
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
        K (list): list of :math:`p` within-iteration A->B coordination ndarrays, optional
        Q (list): list of :math:`p` within-iteration B->A coordination ndarrays, optional
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
    p = len(Z)
    for i in range(n):
        A[i] = A[i](**data[0][i])

    for j in range(m):
        B[j] = B[j](**data[1][j])

    # Initialize the variables
    all_x = [getVar(A[i]) for i in range(n)] # length n list of length k_i \\leq p dict of ndarrays
    all_v = [getVar(A[i]) for i in range(n)] # length n list of length k_i \\leq p dict of ndarrays
    all_b = [getVar(B[j]) for j in range(m)] # length m list of length k_j \\leq p dict of ndarrays

    if verbose or callback is not None:
        all_y = [getVar(A[i]) for i in range(n)] 

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
    PB = getPA([Bj.vars for Bj in B], p)
    fdr = getFeedersL(n, Z, PA)
    Q_fdr = getFeedersQ(A, Q, PA, PB)
    K_fdr = getFeedersK(B, K, PA, PB)

    # Get B calculation order
    Bready = getBorder(A, B, K, PA, PB)

    # Get D_A
    DA = permute(n, Z, PA)

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum subgradients||')
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

            if verbose or callback is not None:
                for k in A[i].vars:
                    all_y[i][k] = all_x[i][k].copy()
            A[i].prox_step(all_x[i], alpha, DA[i])
            for j in Bready[i]:
                # Build argument for B.grad
                for k in B[j].vars:
                    b_input_k = sum(all_x[j][k]*wt for j,wt in K_fdr[j][k])
                    all_b[j][k] = b_input_k
                
                # print('before', i, j, all_b[j])
                B[j].grad(all_b[j])
                
                # print('after', i, j, all_b[j])
                for k in B[j].vars:
                    all_b[j][k] *= alpha
            
        if callback is not None and callback(itr, all_x=all_x, all_v=all_v, all_b=all_b, all_y=all_y): break

        if verbose and (itr+1) % checkperiod == 0:
            xbar = [np.mean([all_x[i][k] for i in PA[k]], axis=0) for k in range(p)]
            xsqdiff = np.sum([(all_x[i][k] - xbar[k])**2 for k in range(p) for i in PA[k] ])
            dualnorm = get_sum_subgradients(all_x, all_y, all_b, PA, Z, PB, alpha)
            print(f"{datetime.now()}\t{itr+1}\t{xsqdiff**0.5:.3e}\t{dualnorm:.3e}")

        for i in range(n):
            for k in A[i].vars:
                idx = PA[k].index(i)
                all_v[i][k] -= sum(gammaW[k][idx, jdx]*all_x[j][k] for jdx, j in enumerate(PA[k]))

        
    xbar = [np.mean([all_x[i][k] for i in PA[k]], axis=0) for k in range(p)]

    return xbar, all_x, all_v


def cabraFullAlgorithm(data, A, B, W, Z, K=None, Q=None, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward-forward-backward resolvent splitting algorithm in serial

    Args:
        data (list): list of lists where data[0] has a list of :math:`n` initialization dictionaries for A, and data[1] contains the :math:`m` initialization dictionaries for B
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function 
        B (list): list of :math:`m` initializable cocoercive operators callable via a grad function
        W (list): list of :math:`p` between-iteration consensus ndarrays
        Z (list): list of :math:`p` within-iteration coordination ndarrays
        K (list): list of :math:`p` within-iteration A->B coordination ndarrays, optional
        Q (list): list of :math:`p` within-iteration B->A coordination ndarrays, optional
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
    p = len(Z)
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
    PA = getPA([Ai.vars for Ai in A], p)
    PB = getPA([Bj.vars for Bj in B], p)
    fdr = getFeedersL(n, Z, PA)
    Q_fdr = getFeedersQ(A, Q, PA, PB)
    K_fdr = getFeedersK(B, K, PA, PB)

    # Get B calculation order
    Bready = getBorder(A, B, K, PA, PB)

    # Get D_A
    DA = permute(n, Z, PA)

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||')
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
                
                # print('before', i, j, all_b[j])
                B[j].grad(all_b[j])
                
                # print('after', i, j, all_b[j])
                for k in B[j].vars:
                    all_b[j][k] *= alpha
            
        if callback is not None and callback(itr, all_x, all_v, all_b): break

        if verbose and itr % checkperiod == 0:
            ybar = [np.mean([all_x[i][k] for i in PA[k]], axis=0) for k in range(p)]
            ysqdiff = sum(sum(np.linalg.norm(all_x[i][k] - ybar[k])**2 for i in PA[k]) for k in range(p))
            print(f"{datetime.now()}\t{itr}\t{ysqdiff:.3e}")

        for i in range(n):
            for k in A[i].vars:
                idx = PA[k].index(i)
                all_v[i][k] -= sum(gammaW[k][idx, jdx]*all_x[j][k] for jdx, j in enumerate(PA[k]))

        
    ybar = [np.mean([all_x[i][k] for i in PA[k]], axis=0) for k in range(p)]
    
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


def cabraStarAlgorithm(data, A, B, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=1.0, alpha=1.0, verbose=False, callback=None):
    """
    Run the coupled adaptive backward forward backward resolvent splitting algorithm with star matrices W=Z, K fed only from operator 1, and Q even distributed across all operators. 
    Assumes the first operator is the center of the star.
    Assumes the first operator contains all variables.

    Args:
        data (list): list of lists where data[0] has a list of :math:`n` initialization dictionaries for A,
            each of which requires a varlist key with a list of variable indexes as its value, and data[1] contains the :math:`m` initialization dictionaries for B.
            The first A operator will need the D vector with its degree for each variable provided in the data.
        A (list): list of :math:`n` initializable maximal monotone operators callable via a prox function.
        B (list): list of :math:`m` initializable beta_j-cocoercive operators with m \\leq n-1 and beta_j >= 1, callable via a grad function.
        warmstartprimal (dictionary, optional): array with estimate of x_0
        warmstartdual (list, optional): list of length :math:`n` giving a dictionary for each resolvent with keys for each subvector id pertaining to that resolvent and values giving the subgradient estimate for that subvector in that resolvent. The sum of the subgradients over the resolvents for each subvector must be zero.
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in (0,2) for :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the positive resolvent step size in :math:`x^{k+1} = J_{\\alpha A_i}(y^k)`
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (list): value of x_0 at termination
        all_v (list): list of :math:`n` ndarrays of the node consensus iterates at solution

    Examples:
    """
    # Initialize the operators
    nn = len(data[0])
    m = len(data[1])
    assert(len(A) == len(data[0]))
    assert(len(B) == len(data[1]))
    vi = [] # variable indices for A 
    for di in data[0]:
        vi.append(di['varlist'])
    b_vi = [] # variable indices for B
    for di in data[1]:
        b_vi.append(di['varlist'])

    counts = data[0][0]['D'] + 1 # number of proxs for each variable

    # Initialize the variables
    all_x = [np.zeros(len(vi[i])) for i in range(nn)]
    all_b = [np.zeros(len(b_vi[j])) for j in range(m)]
    if warmstartdual is not None:
        all_v = warmstartdual
    else:
        all_v = [all_x[i].copy() for i in range(nn)]
    if verbose or callback is not None:
        all_y = [all_x[i].copy() for i in range(nn)]
    xbar = all_x[0].copy()
    for i in range(nn):
        A[i] = A[i](**data[0][i])
    for j in range(m):
        B[j] = B[j](**data[1][j])

    # Get intersection of variables for each B operator and variables for each A operator (1 to n-1)
    intersect_vars = {(i, j): list(set(vi[i]).intersection(set(b_vi[j]))) for i in range(1, nn) for j in range(m)}
    A_ind_ij = {(i, j): [vi[i].index(k) for k in intersect_vars[(i, j)]] for i in range(1, nn) for j in range(m)}
    B_ind_ij = {(i, j): [b_vi[j].index(k) for k in intersect_vars[(i, j)]] for i in range(1, nn) for j in range(m)}
    # print('intersect_vars', intersect_vars)
    # print('A_ind_ij', A_ind_ij)
    # print('B_ind_ij', B_ind_ij)

    # Get counts for the number of B operators using each variable
    b_counts = np.zeros(len(data[0][0]['D']), dtype=int)
    for j in range(m):
        b_counts[b_vi[j]] += 1

    # replace zeros in b_counts with ones to avoid divide by zero
    b_counts[b_counts == 0] = 1

    # Warm start primal
    if warmstartprimal is not None:
        all_v[0] += data[0][0]['D']*warmstartprimal
        for i in range(1, nn):
            all_v[i] -= warmstartprimal[vi[i]]

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        # Prox loop
        # First iterate (center of star)
        np.copyto(all_x[0], all_v[0])
        if verbose or callback is not None:
            np.copyto(all_y[0],all_x[0])
        all_x[0] = A[0].prox(all_x[0], alpha)

        # B operators
        for j in range(m):
            all_b[j] = B[j].grad(all_x[0][b_vi[j]])

        # Points of star
        for i in range(1, nn):
            # fill all_x[i] with zeros
            all_x[i].fill(0.0)
            # add intersection of B operators
            for j in range(m):
                if len(intersect_vars[(i, j)]) > 0:
                    all_x[i][A_ind_ij[(i, j)]] -= all_b[j][B_ind_ij[(i, j)]]
            # divide by the number of B operators using each variable
            all_x[i] /= b_counts[vi[i]]
            all_x[i] *= alpha
            all_x[i] += 2*all_x[0][vi[i]]
            all_x[i] += all_v[i]
            if verbose or callback is not None:
                np.copyto(all_y[i],all_x[i])
            all_x[i] = A[i].prox(all_x[i], alpha)
            
        if callback is not None and callback(itr=itr, all_x=all_x, all_v=all_v, all_y=all_y, all_b=all_b, A=A, data=data, vi=vi): break

        if verbose and (itr+1) % checkperiod == 0:
            # Norm of the sum of the differences from the mean value
            xbar = getXbar(all_x, xbar, data[0], counts)
            xsqdiff = sum((xbar-all_x[0])**2)
            for indexi, xi in zip(vi[1:], all_x[1:]):
                xsqdiff += sum((xbar[indexi]-xi)**2)

            # Norm of the sum of the subgradients
            subg = all_y[0] - data[0][0]['D']*all_x[0]
            for indexi, yi, xi in zip(vi[1:], all_y[1:], all_x[1:]):
                subg[indexi] += yi - xi
            subg /= alpha
            for indexj, bj in zip(b_vi, all_b):
                subg[indexj] += bj
            subg_sum_norm = np.linalg.norm(subg)
            print(f"{datetime.now()}\t{itr+1}\t{xsqdiff**0.5:.3e}\t{subg_sum_norm:.3e}")

        # v updates
        zero_update = data[0][0]['D']*all_x[0]
        for i in range(1, nn):
            zero_update[vi[i]] -= all_x[i]
            all_x[i] -= all_x[0][vi[i]]
            all_x[i] *= gamma
            all_v[i] -= all_x[i]
        zero_update *= gamma
        all_v[0] -= zero_update

    return all_x[0], all_v


def getXbar(all_x, xbar, data, counts):
    np.copyto(xbar, all_x[0])
    for i in range(1, len(data)):
        xbar[data[i]['varlist']] += all_x[i]
    xbar /= counts
    return xbar

def get_sum_subgradients(all_x, all_y, all_b, PA, Z, PB, alpha):
    s = {}
    for k, Pk in enumerate(PA):
        s[k] = np.sum([all_y[i][k] - Z[k][idx,idx]*all_x[i][k] for idx, i in enumerate(Pk)],axis=0)
        
    for k, Pk in enumerate(PB):
        s[k] += np.sum([all_b[j][k] for j in Pk], axis=0)
    
    return (np.sum([s[k]**2 for k in range(len(PA))])/(alpha**2))**0.5