import numpy as np
import cvxpy as cvx
from scipy.linalg import ldl

def getCore(n, fixed_Z={}, fixed_W={}, c=None, eps=0.0, gamma=1.0, adj=False, **kwargs):
    '''
    Get core variables and constraints for the algorithm design SDP

    :math:`W \\mathbb{1} = 0`

    :math:`Z \\mathbb{1} = 0`

    :math:`\\lambda_{1}(W) + \\lambda_{2}(W) \\geq c`

    :math:`Z - W \\succeq 0`

    :math:`\\mathrm{diag}(Z) = Z_{11}\\mathbb{1}`

    :math:`2 - \\varepsilon \\leq Z_{11} \\leq 2 + \\varepsilon`

    Args:
        n (int): number of nodes
        fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
        fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
        c (float): connectivity parameter (default 2*(1-cos(pi/n)))
        eps (float): epsilon for Z[0,0] = 2 + eps constraint
        gamma (float): scaling parameter for Z
        adj (bool): whether to use the edge adjacency formulation
        kwargs: additional keyword arguments for the algorithm
        
    Returns:
        Z (ndarray): n x n cvxpy decision variable matrix for Z
        W (ndarray): n x n cvxpy decision variable matrix for W
        cons (list): list of cvxpy constraints

    Examples:
        >>> import cvxpy as cvx
        >>> from oars.matrices import getCore
        >>> Z, W, cons = getCore(4, fixed_W={(3, 0): 0})
        >>> obj = cvx.Minimize(cvx.norm(Z-W, 'fro'))
        >>> prob = cvx.Problem(obj, cons)
        >>> prob.solve()
        >>> print(Z.value)
        [[ 2. -1. -1. -0.]
        [-1.  2. -0. -1.]
        [-1. -0.  2. -1.]
        [-0. -1. -1.  2.]]
        >>> print(W.value)
        [[ 2. -1. -1. -0.]
        [-1.  2. -0. -1.]
        [-1. -0.  2. -1.]
        [-0. -1. -1.  2.]]

    '''
    if c is None:
        c = 2*(1-np.cos(np.pi/n))

    # Variables
    if not adj:
        W = cvx.Variable((n,n), PSD=True)
        Z = cvx.Variable((n,n), symmetric=True)
    else:
        Mz = getIncidenceFixed(n, fixed_Z)
        Mw = getIncidenceFixed(n, fixed_W)
        ez = Mz.shape[0]
        ew = Mw.shape[0]

        # Variables
        z = cvx.Variable(ez, nonneg=True)
        w = cvx.Variable(ew, nonneg=True)
        Z = Mz.T @ cvx.diag(z) @ Mz
        W = Mw.T @ cvx.diag(w) @ Mw

    # Constraints
    cons = [gamma*Z >> W, # Z - W is PSD
            cvx.lambda_sum_smallest(W, 2) >= c, # Fiedler value constraint
            cvx.sum(W, axis=1) == 0, # W sums to zero
            cvx.sum(Z, axis=1) == 0, # Z sums to zero
            2-eps <= Z[0,0],
            Z[0,0] <= 2+eps] # bounds on Z diagonal entries
    
    cons += [Z[i,i] == Z[0,0] for i in range(1,n)] # Z diagonal entries equal to one another

    # Set fixed Z and W values
    cons += [Z[idx] == val for idx,val in fixed_Z.items()]
    cons += [W[idx] == val for idx,val in fixed_W.items()]

    return Z, W, cons    

def postprocess(prob, Z, W, eps=0.0, **kwargs):
    '''
    Postprocess the results of the optimization

    Args:
        prob (cvxpy problem): cvxpy problem object
        eps (float): allowable deviation from 2 in Z diagonal
        Z (cvxpy variable): n x n cvxpy decision variable matrix for Z
        W (cvxpy variable): n x n cvxpy decision variable matrix for W

    Returns:
        Z (ndarray): n x n numpy array of resolvent multipliers
        W (ndarray): n x n numpy array of consensus multipliers
        alpha (float): scaling factor for resolvent if eps is nonzero
    '''

    alpha = 1
    if prob.status == 'infeasible':
        Z = None
        W = None
    else:
        if eps!=0.0:
            alpha = 2.0/Z[0,0]
        Z *= alpha
        W *= alpha
    if eps == 0.0:
        return Z, W

    return Z, W, alpha

def getSimilar(n, **kwargs):
    '''
    Find convergence matrix W and consensus matrix Z
    that minimize :math:`\\|Z-W\\|`

    Args:
        n (int): number of resolvents
        kwargs: keyword arguments

            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation

    Returns:
        Z (ndarray): n x n consensus matrix
        W (ndarray): n x n resolvent matrix
        alpha (float): scaling factor for resolvent if eps is nonzero
    '''
    # Set default values
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']

    Z, W, cons = getCore(n, **kwargs)

    # Objective function
    obj = cvx.Minimize(cvx.norm(Z-W))
    
    # Solve
    prob = cvx.Problem(obj, cons)
    prob.solve()

    # Print results
    if verbose:
        print(prob.status)
        print(prob.value)

    return postprocess(prob, Z.value, W.value, **kwargs)

def getMaxConnectivity(n, vz=1.0, vw=1.0, **kwargs):
    '''
    Find convergence matrix W and consensus matrix Z
    that maximize the sum of the algebraic connectivity for W and Z

    Args:
        n (int): number of resolvents
        fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
        fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
        vz (float): weight for Z Fiedler value
        vw (float): weight for W Fiedler value
        **kwargs: keyword arguments for verbosity and cvxpy solver

                    - c (float): connectivity parameter
                    - eps (float): allowable deviation from 2 in Z diagonal
                    - gamma (float): scaling parameter for Z
                    - adj (bool): whether to use the edge adjacency formulation

    Returns:
        Z (ndarray): n x n resolvent matrix
        W (ndarray): n x n consensus matrix
        alpha (float): scaling factor for resolvent if eps is nonzero
    '''

    # Set default values
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']

    Z, W, cons = getCore(n, **kwargs)

    # Objective function
    cw = cvx.lambda_sum_smallest(W, 2)
    cz = cvx.lambda_sum_smallest(Z, 2)
    obj_fun = vw*cw + vz*cz 

    # Solve
    obj = cvx.Maximize(obj_fun)
    prob = cvx.Problem(obj, cons)
    prob.solve()

    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal cw", cw.value)
        print("optimal cz", cz.value)
        print("optimal W")
        print(W.value)
        print("optimal Z")
        print(Z.value)

    return postprocess(prob, Z.value, W.value, **kwargs)

def getMinSLEM(n, vz=1.0, vw=1.0, **kwargs):
    """
    Find convergence matrix W and consensus matrix Z
    that minimize the sum of the SLEM values for W and Z

    Args:
        n (int): number of resolvents
        fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
        fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
        vz (float): weight for Z SLEM value
        vw (float): weight for W SLEM value
        **kwargs: keyword arguments for verbosity and cvxpy solver

            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation            
    Returns:
        Z (ndarray): n x n resolvent matrix
        W (ndarray): n x n consensus matrix
        alpha (float): scaling factor for resolvent if eps is nonzero
    """

    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']

    Z, W, cons = getCore(n, **kwargs)
    # Variables
    gw = cvx.Variable(1)  # SLEM value for W
    gz = cvx.Variable(1)  # SLEM value for Z

    # Constraints
    ZP = np.eye(n) - Z/2 # Find adjacency matrix of Z scaled to sum to 1, and with diagonal entries <= 1
    ZPvU = ZP - np.ones((n,n))/n # difference between scaled Z graph and uniform graph
    WP = np.eye(n) - W/2 # Find adjacency matrix of W scaled to sum to 1, and with diagonal entries <= 1
    WPvU = WP - np.ones((n,n))/n # difference between scaled W graph and uniform graph
    cons += [-gz*np.eye(n) << ZPvU, ZPvU << gz*np.eye(n)] # ZPvU is bounded by gz
    cons += [-gw*np.eye(n) << WPvU, WPvU << gw*np.eye(n)] # WPvU is bounded by gw

    
    obj_fun = vz*gz + vw*gw

    # Solve
    obj = cvx.Minimize(obj_fun)
    prob = cvx.Problem(obj, cons)
    prob.solve()

    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal gw", gw.value)
        print("optimal gz", gz.value)
        print("optimal W")
        print(W.value)
        print("optimal Z")
        print(Z.value)

    return postprocess(prob, Z.value, W.value, **kwargs)

def getMinResist(n, vz=1.0, vw=1.0, **kwargs):
    """
    Find convergence matrix W and consensus matrix Z
    that minimize the sum of the total effective resistances for W and Z

    Args:
        n (int): number of resolvents
        fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
        fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
        vz (float): weight for Z TER value
        vw (float): weight for W TER value
        **kwargs: keyword arguments for verbosity and cvxpy solver

            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation
    Returns:
        Z (ndarray): n x n resolvent matrix
        W (ndarray): n x n consensus matrix
        alpha: scaling factor for resolvent if eps is nonzero
    """

    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']

    Z, W, cons = getCore(n, **kwargs)
    
    # Objective function
    terz = cvx.tr_inv(Z + np.ones((n,n))/n)
    terw = cvx.tr_inv(W + np.ones((n,n))/n)
    obj_fun = vz*terz + vw*terw

    # Solve
    obj = cvx.Minimize(obj_fun)
    prob = cvx.Problem(obj, cons)
    prob.solve()

    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal TER Z", terz.value)
        print("optimal TER W", terw.value)
        print("optimal W")
        print(W.value)
        print("optimal Z")
        print(Z.value)
    
    return postprocess(prob, Z.value, W.value, **kwargs)

def getBlockFixed(n, m):
    '''
    Get prohibited values for block size(s) m

    Args:
        n (int): number of resolvents
        m (int or list of ints): block size, either an integer or a list of integers

    Returns:
        Z_fixed (dict): dictionary of fixed Z values with keys as (i,j) tuples
        W_fixed (dict): dictionary of fixed W values with keys as (i,j) tuples
        
    '''
    if isinstance(m, int):
        Z_fixed = {(i,j): 0 for i in range(n) for j in range(n) if j<i and (i//m == j//m)}
        W_fixed = {(i,j): 0 for i in range(n) for j in range(n) if j<i and abs(j//m - i//m) > 1}
    else:
        Z_fixed = {}
        block_starts = np.zeros(len(m))
        for i in range(1,len(m)):
            block_starts[i] = block_starts[i-1] + m[i-1]

        # Set Z_fixed[i,j] = 0 for all i,j where i//m_k = j//m_k for k in block_sizes
        i = 0
        for k in m:
            for i1 in range(i,i+k):
                for j1 in range(i1+1,i+k):
                    Z_fixed[(i1,j1)] = 0
            i += k
        W_fixed = {}
        for i in range(n):
            block_i = block_starts.searchsorted(i, side='right')-1
            for j in range(i+1,n):
                block_j = block_starts.searchsorted(j, side='right')-1
                if block_j - block_i > 1:
                    W_fixed[(i,j)] = 0
    return Z_fixed, W_fixed

def getBlockMin(n, m, objective=getSimilar, **kwargs):
    '''
    Get the block-size m matrices for n resolvents
    using the objective function specified

    Args:
        n (int): number of resolvents
        m (int or list of ints): block size, either an integer or a list of integers
        objective (function): objective function
        kwargs: keyword arguments for the objective function

            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation        
    Returns:
        Z (ndarray): (n,n) resolvent matrix
        W (ndarray): (n,n) consensus matrix
        alpha (float): scaling factor for resolvent if eps is nonzero
    
    '''
    Z_fixed, W_fixed = getBlockFixed(n, m)

    return objective(n, fixed_Z=Z_fixed, fixed_W=W_fixed, **kwargs)

def getMfromWCholesky(W):
    '''
    Reconstruct M from W via the cholesky 
    decomposition, as described in the paper.

    Args:
        W (ndarray): n x n symmetric psd numpy array w/ Null(W) = 1

    Returns:
        M (ndarray): (n-1) x n array such that M.T @ M = W
    '''

    lu, d, perm = ldl(W, lower=0)
    assert(np.all(d >= -1e-6)) #the values of d shouldn't be too small
    #b/c W is psd
    assert(np.all(np.diag(np.diag(d)) == d)) #make sure d is diagonal
    #for complex matrices it can be 2x2 block diagonal, but for real
    #it shouldn't be.
    diag_d = np.diag(d)
    assert((diag_d <= 1e-6).sum() == 1) #there should be exactly 1 zero value
    #in diag(d), since W is rank n-1 and lu is full rank.
    where_d_nonzero = diag_d >= 1e-6
    return (lu[:,where_d_nonzero] * np.sqrt(diag_d[where_d_nonzero])).T

def getMfromWEigen(W):
    '''
    Reconstruct M from W via the eigenvalue 
    decomposition, as described in the paper.

    Args:
        W (ndarray): n x n symmetric psd numpy array w/ Null(W) = 1

    Returns:
        M (ndarray): (n-1) x n array such that M.T @ M = W
    '''
    vals, vecs = np.linalg.eigh(W)
    assert(np.all(vals >= -1e-6)) #W is psd so these eigvals shouldn't be too negative
    assert((vals <= 1e-6).sum() == 1) #rank of null space should be 1
    #this must the first eigval since eigvals are in ascending order
    #according the numpy docs
    return (vecs[:, 1:] * np.sqrt(vals[1:])).T    

def getIncidence(W):
    '''
    Convert a Stieltjes graph Laplacian W to an incidence matrix
    
    Args:
        W (ndarray): n x n numpy array of graph Laplacian
                     with only negative off-diagonal entries
    
    Returns:
        M (ndarray): m x n numpy array of incidence matrix
                     where m is the number of edges
    '''
    n = W.shape[0]
    P = -np.tril(W, k=-1)
    # Count non-zero entries
    nnz = np.count_nonzero(P)
    M = np.zeros((nnz, n))
    k = 0
    for i in range(n):
        for j in range(i):
            if P[i,j] >= 1e-6:
                val = P[i,j]**0.5
                M[k, i] = val
                M[k, j] = -val
                k += 1
    return M
