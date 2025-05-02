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
        eps (float): nonnegative epsilon for Z[0,0] = 2 + eps constraint (default 0.0)
        gamma (float): scaling parameter for Z (default 1.0)
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
    t = cvx.Variable()
    if not adj:
        W = cvx.Variable((n,n), symmetric=True)
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
    ones = 4*np.outer(np.ones(n), np.ones(n))
    cons = [W + ones - t*np.eye(n) >> 0, # Fiedler value constraint
            t >= c, # Connectivity constraint
            gamma*Z >> W, # Z - W is PSD            
            cvx.sum(W, axis=1) == 0, # W sums to zero
            cvx.sum(Z) == 0] # Z sums to zero

    if eps != 0.0:
        cons += [
            2-eps <= Z[0,0],
            Z[0,0] <= 2+eps] # bounds on Z diagonal entries
        cons += [Z[i,i] == Z[0,0] for i in range(1,n)] # Z diagonal entries equal to one another
    else:
        cons += [Z[i,i] == 2 for i in range(n)] # Z diagonal entries equal to 2

    # Set fixed Z and W values
    cons += [Z[idx] == val for idx,val in fixed_Z.items() if not adj or val != 0]
    cons += [W[idx] == val for idx,val in fixed_W.items() if not adj or val != 0]

    return Z, W, cons    


def getCabra(n, m, fixed_Z={}, fixed_W={}, fixed_Q={}, fixed_K={}, cutoffs=None, c=None, adj=False, **kwargs):
    '''
    Get core variables and constraints for the algorithm design SDP

    :math:`W \\mathbb{1} = 0`

    :math:`Z \\mathbb{1} = 0`

    :math:`\\lambda_{1}(W) + \\lambda_{2}(W) \\geq c`

    :math:`Z - W \\succeq 0`

    :math:`\\mathrm{diag}(Z) = Z_{11}\\mathbb{1}`

    :math:`2 - \\varepsilon \\leq Z_{11} \\leq 2 + \\varepsilon`

    Args:
        n (int): number of maximal monotone operators (resolvents)
        m (int): number of cocoercive operators (direct)
        fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
        fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
        cutoffs (list): list of :math:`m` indices at and above which the entries in the rows of K must be zero
        c (float): connectivity parameter (default :math:`2*(1-cos(\\pi/n)`))
        gamma (float): scaling parameter for Z (default 1.0)
        adj (bool): whether to use the edge adjacency formulation
        kwargs: additional keyword arguments for the algorithm
        
    Returns:
        Z (ndarray): n x n cvxpy decision variable matrix for Z
        W (ndarray): n x n cvxpy decision variable matrix for W
        Q (ndarray): n x m cvxpy decision variable matrix for Q
        K (ndarray): m x n cvxpy decision variable matrix for K
        cons (list): list of cvxpy constraints

    Examples:
        >>> import cvxpy as cvx
        >>> from oars.matrices import getCabra
        >>> Z, W, Q, K, cons = getCore(4, fixed_W={(3, 0): 0})
        >>> obj = cvx.Minimize(cvx.norm(Z-W, 'fro'))
        >>> prob = cvx.Problem(obj, cons)
        >>> prob.solve()

    '''
    if c is None:
        c = 2*(1-np.cos(np.pi/n))
    if cutoffs is None:
        cutoffs = [min(n-1, j) for j in range(1, m+1)]
    else:
        assert len(cutoffs) == m
    cons = []

    # Variables
    
    Q = cvx.Variable((n,m))
    K = cvx.Variable((m,n))
    if not adj:
        W = cvx.Variable((n,n), symmetric=True)
        Z = cvx.Variable((n,n), symmetric=True)
        cons += [cvx.sum(Z) == 0] # Z sums to zero
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
    ones = np.outer(np.ones(n), np.ones(n))
    cons += [Z >> W, # Z - W is PSD            
            cvx.bmat([[W, Q-K.T],
                      [Q.T-K, np.eye(m)]]) >> 0,
            cvx.sum(K, axis=1) == 1,
            cvx.sum(Q, axis=0) == 1]
    cons += [K[j,i] == 0 for j in range(m) for i in range(cutoffs[j], n)]
    cons += [Q[i,j] == 0 for j in range(m) for i in range(cutoffs[j])]
    # cons += [K[i,j] == 0 for i in range(m) for j in range(max(0,n-(m-i)),n)]
    # cons += [Q[i,j] == 0 for i in range(min(m,n)) for j in range(i,m)]

    # Set fixed Z and W values
    if not adj:
        cons += [Z[idx] == val for idx,val in fixed_Z.items()]
        cons += [W[idx] == val for idx,val in fixed_W.items()]

    cons += [Q[idx] == val for idx,val in fixed_Q.items()]
    cons += [K[idx] == val for idx,val in fixed_K.items()]
    cons += [W + n**2 * ones >> c*np.eye(n)] # Fiedler value constraint #cvx.trace(Z) <= n**2,

    return Z, W, Q, K, cons    

def getCabraTight(n, m, verbose=False, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    ZminusU = cvx.bmat([[Z, Q-K.T],
                        [Q.T-K, np.eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(ZminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve()
    
    U = getU(Q.value, K.value)
    if verbose:
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U.value

def getCabraTightD(n, m, D, verbose=0, solverargs=None, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    cons += [cvx.diag(Z) == D]
    ZminusU = cvx.bmat([[Z, Q-K.T],
                        [Q.T-K, np.eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(ZminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose-1), **solverargs)
    
    U = getU(Q.value, K.value)
    if verbose > 0:
        print(prob.status)
        print(prob.value)
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U

def getU(Q, K):
    return (Q - K.T)@(Q.T - K)

def getIncidenceFixed(n, fixed):
    '''
    Converts fixed dictionary to incidence matrix

    Args:
        n (int): dimension of matrix
        fixed (dict): dictionary with entries (r,c): 0 for edges to exclude

    Returns:
        M (ndarray): m x n numpy array of incidence matrix
                     where m is the number of edges
    '''
    edges = 0
    M = []
    for i in range(n):
        for j in range(i):
            if fixed.get((i,j), 1) == 0 or fixed.get((j,i), 1) == 0:
                continue
            else:
                row = np.zeros(n)
                row[i] = 1
                row[j] = -1
                M.append(row)
    return np.array(M)

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

def getMinSpectralDifference(n, **kwargs):
    '''
    Find convergence matrix W and consensus matrix Z
    that minimize :math:`\\|Z-W\\|`

    Args:
        n (int): number of resolvents
        kwargs: keyword arguments

            - fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
            - fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation

    Returns:
        Z (ndarray): n x n consensus matrix
        W (ndarray): n x n resolvent matrix
        alpha (float): scaling factor for resolvent if eps is nonzero

    Examples:
        >>> from oars.matrices import getMinSpectralDifference
        >>> Z, W = getMinSpectralDifference(4, fixed_W={(3, 0): 0}, fixed_Z={(1, 0): 0})
        >>> print(Z)
        [[ 2.    -0.    -1.645 -0.355]
        [-0.     2.    -0.355 -1.645]
        [-1.645 -0.355  2.    -0.   ]
        [-0.355 -1.645 -0.     2.   ]]
        >>> print(W)
        [[ 1.645 -0.17  -1.475 -0.   ]
        [-0.17   1.918 -0.273 -1.475]
        [-1.475 -0.273  1.918 -0.17 ]
        [-0.    -1.475 -0.17   1.645]]

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

    Examples:
        >>> from oars.matrices import getMaxConnectivity
        >>> Z, W = getMaxConnectivity(4, fixed_W={(3, 0): 0}, fixed_Z={(1, 0): 0})
        >>> print(Z)
        [[ 2.  0. -1. -1.]
        [ 0.  2. -1. -1.]
        [-1. -1.  2.  0.]
        [-1. -1.  0.  2.]]
        >>> print(W)
        [[ 1.    -0.5   -0.5   -0.   ]
        [-0.5    1.459 -0.459 -0.5  ]
        [-0.5   -0.459  1.459 -0.5  ]
        [-0.    -0.5   -0.5    1.   ]]
    '''

    # Set default values
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']

    Z, W, cons = getCore(n, **kwargs)

    # Additional variables
    t = cvx.Variable()
    s = cvx.Variable()

    # Additional parameter
    ones = 4*np.ones((n,n))

    # Constraints
    cons = cons[2:] # Remove previous connectivity constraint
    cons += [W + ones - t*np.eye(n) >> 0, # Fiedler value constraint
            Z + ones - s*np.eye(n) >> 0] # Fiedler value constraint
    
    # Objective function
    obj_fun = vw*t + vz*s 

    # Solve
    obj = cvx.Maximize(obj_fun)
    prob = cvx.Problem(obj, cons)
    prob.solve()

    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal Fiedler value for W", t.value)
        print("optimal Fiedler value for Z", s.value)
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
        vz (float): weight for Z SLEM value
        vw (float): weight for W SLEM value
        **kwargs: keyword arguments for verbosity and cvxpy solver

            - fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
            - fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation            
    Returns:
        Z (ndarray): n x n resolvent matrix
        W (ndarray): n x n consensus matrix
        alpha (float): scaling factor for resolvent if eps is nonzero

    Examples:
        >>> from oars.matrices import getMinSLEM
        >>> Z, W = getMinSLEM(4, fixed_W={(3, 0): 0}, fixed_Z={(1, 0): 0})
        >>> print(Z)
        [[ 2.    -0.    -1.333 -0.667]
        [-0.     2.    -0.667 -1.333]
        [-1.333 -0.667  2.    -0.   ]
        [-0.667 -1.333 -0.     2.   ]]
        >>> print(W)
        [[ 1.333 -0.667 -0.667  0.   ]
        [-0.667  1.333 -0.    -0.667]
        [-0.667 -0.     1.333 -0.667]
        [ 0.    -0.667 -0.667  1.333]]
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

    Examples:
        >>> from oars.matrices import getMinResist
        >>> Z, W = getMinResist(4, fixed_W={(3, 0): 0}, fixed_Z={(1, 0): 0}, gamma=1.5)
        >>> print(Z)
        [[ 2.     0.    -1.174 -0.826]
        [ 0.     2.    -0.826 -1.174]
        [-1.174 -0.826  2.     0.   ]
        [-0.826 -1.174  0.     2.   ]]
        >>> print(W)
        [[ 1.76  -0.704 -1.056  0.   ]
        [-0.704  2.6   -0.84  -1.056]
        [-1.056 -0.84   2.6   -0.704]
        [ 0.    -1.056 -0.704  1.76 ]]
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
        
    Examples:
        >>> from oars.matrices import getBlockFixed
        >>> Z_fixed, W_fixed = getBlockFixed(4, 2)
        >>> print(Z_fixed)
        {(1, 0): 0, (3, 2): 0}
        >>> print(W_fixed)
        {}
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

def getBlockMin(n, m, builder=getMinSpectralDifference, **kwargs):
    '''
    Get the d-Block design with block size m (or list of block sizes with length d). 
    If m is an integer, :math:`d = \\text{ceil}(n/m)`.
    Uses a provided builder function to specify the objective function and any other constraints in addition to the d-Block constraints.
    The default builder is getMinSpectralDifference.

    Args:
        n (int): number of resolvents
        m (int or list of ints): block size, either an integer or a list of integers
        builder (function): builder function that takes n (int), fixed_Z (dict), fixed_W (dict) and kwargs and returns Z, W, alpha
        kwargs: keyword arguments for the builder function

            - c (float): connectivity parameter
            - eps (float): allowable deviation from 2 in Z diagonal
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation    

    Returns:
        Z (ndarray): (n,n) resolvent matrix
        W (ndarray): (n,n) consensus matrix
        alpha (float): scaling factor for resolvent if eps is nonzero
    
    Examples:
        >>> from oars.matrices import getBlockMin, getMinResist
        >>> Z, W = getBlockMin(6, 2, builder=getMinResist)
        >>> print(Z)
        [[ 2.   0.  -0.5 -0.5 -0.5 -0.5]
        [ 0.   2.  -0.5 -0.5 -0.5 -0.5]
        [-0.5 -0.5  2.   0.  -0.5 -0.5]
        [-0.5 -0.5  0.   2.  -0.5 -0.5]
        [-0.5 -0.5 -0.5 -0.5  2.   0. ]
        [-0.5 -0.5 -0.5 -0.5  0.   2. ]]
        >>> print(W)
        [[ 1.5 -0.5 -0.5 -0.5  0.   0. ]
        [-0.5  1.5 -0.5 -0.5  0.   0. ]
        [-0.5 -0.5  2.   0.  -0.5 -0.5]
        [-0.5 -0.5  0.   2.  -0.5 -0.5]
        [ 0.   0.  -0.5 -0.5  1.5 -0.5]
        [ 0.   0.  -0.5 -0.5 -0.5  1.5]]
    '''
    Z_fixed, W_fixed = getBlockFixed(n, m)

    return builder(n, fixed_Z=Z_fixed, fixed_W=W_fixed, **kwargs)

def getMfromWCholesky(W):
    '''
    Reconstruct M from W via the cholesky 
    decomposition, as described in the paper.

    Args:
        W (ndarray): n x n symmetric psd numpy array w/ Null(W) = 1

    Returns:
        M (ndarray): (n-1) x n array such that M.T @ M = W

    Examples:
        >>> from oars.matrices import getMfromWCholesky, getTwoBlockSimilar
        >>> Z, W = getTwoBlockSimilar(4)
        >>> M = getMfromWCholesky(W)
        >>> print(M)
        [[-1.     1.     0.     0.   ]
        [-0.707 -0.707  1.414  0.   ]
        [-0.707 -0.707  0.     1.414]]
    '''

    lu, d, perm = ldl(W, lower=0)
    assert(np.all(d >= -1e-6)) #the values of d shouldn't be too small
    #b/c W is psd
    assert(np.all(np.diag(np.diag(d)) == d)) #make sure d is diagonal
    #for complex matrices it can be 2x2 block diagonal, but for real
    #it shouldn't be.
    diag_d = np.diag(d)
    # assert((diag_d <= 1e-6).sum() == 1) #there should be exactly 1 zero value
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

    Examples:
        >>> from oars.matrices import getMfromWEigen, getTwoBlockSimilar
        >>> Z, W = getTwoBlockSimilar(4)
        >>> M = getMfromWEigen(W)
        >>> print(M)
        [[-1.  1. -0. -0.]
        [ 0.  0. -1.  1.]
        [-1. -1.  1.  1.]]
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

    Examples:
        >>> from oars.matrices import getIncidence, getTwoBlockSimilar
        >>> Z, W = getTwoBlockSimilar(4)
        >>> M = getIncidence(W)
        >>> print(M)
        [[-1.  0.  1.  0.]
        [ 0. -1.  1.  0.]
        [-1.  0.  0.  1.]
        [ 0. -1.  0.  1.]]

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

def getZfromGraph(A, **kwargs):
    '''
    Get the resolvent matrix Z from a graph adjacency matrix A

    Args:
        A (ndarray): n x n numpy array of graph adjacency matrix

    Returns:
        Z (ndarray): n x n numpy array of resolvent matrix

    Examples:
        >>> from oars.matrices import getZfromGraph
        >>> A = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        >>> Z = getZfromGraph(A)
        >>> print(Z)
    '''
    n = A.shape[0]

    # Doubly stochastic matrix
    Y = ipf(A, **kwargs)

    # Resolvent matrix
    Z = 2*np.eye(n) - Y - Y.T

    return Z



def ipf(A, itrs=100, tol=1e-6, verbose=False):
    '''
    Iterative Proportional Fitting for forming doubly stochastic matrices

    Args:
        A (ndarray): n x n numpy array of nonnegative matrix
        itrs (int): maximum number of iterations
        tol (float): tolerance for convergence
        verbose (bool): whether to print convergence information

    Returns:
        X (ndarray): n x n numpy array of doubly stochastic matrix

    Examples:
        >>> from oars.matrices import ipf
        >>> import numpy as np
        >>> X = np.array([[0, 1, 1, 1], 
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [1, 1, 1, 0]], dtype=float)
        >>> Y = ipf(X.copy())
        >>> print(Y)
        [[0.  0.5 0.5 0. ]
         [0.5 0.  0.  0.5]
         [0.5 0.  0.  0.5]
         [0.  0.5 0.5 0. ]]
    '''
    X = A.astype(float)
    rows, cols = X.shape
    for i in range(itrs):
        for r in range(rows):
            X[r] = X[r] / np.sum(X[r])
        for c in range(cols):
            X[:, c] = X[:, c] / np.sum(X[:, c])
        if np.allclose(np.sum(X, axis=1), np.ones(rows), atol=tol) and np.allclose(np.sum(X, axis=0), np.ones(cols), atol=tol):
            break
    if verbose:
        print("IPF converged after {} iterations".format(i))
    return X

def ipf_sparse(X, itrs=100, tol=1e-6, verbose=False):
    '''
    Iterative Proportional Fitting for forming doubly stochastic matrices
    using a sparse matrix representation in CSC format

    Args:
        X (csc_matrix): n x n scipy sparse array of nonnegative matrix
        itrs (int): maximum number of iterations
        tol (float): tolerance for convergence
        verbose (bool): whether to print convergence information

    Returns:
        X (csc_matrix): n x n scipy sparse array of doubly stochastic matrix
    '''

    rows, cols = X.shape
    for i in range(itrs):
        rowsum = X.sum(axis=1)
        for r in range(rows):
            X[r, :] = X[r, :] / float(rowsum[r])
            # X[r] = X[r] / float(np.sum(X[r]))
            # Numpy_csc_norm(X.data, X.indptr)
        colsum = X.sum(axis=0)
        for c in range(cols):
            X[:, c] = X[:, c] / colsum[0, c]
        if np.allclose(np.sum(X, axis=1), np.ones(rows), atol=tol) and np.allclose(np.sum(X, axis=0), np.ones(cols), atol=tol):
            break
    if verbose:
        print("IPF converged after {} iterations".format(i))
    return X


def Numpy_csc_norm(data,indptr):
    for i in range(indptr.shape[0]-1):
        xs = np.sum(data[indptr[i]:indptr[i+1]])
        #Modify the view in place
        data[indptr[i]:indptr[i+1]]/=xs    

def testMatrices(Z, W):
    """Test that L and W are valid consensus and resolvent matrices"""
    n = Z.shape[0]

    L = -np.tril(Z,-1)
    # Test that L is a resolvent matrix
    # Z sums to 0
    assert(np.isclose(np.sum(Z), 0.0))
    
    # Z is PSD
    assert(np.all(np.linalg.eigvals(Z) >= -1e-7))

    # Diagonal of L is 0
    # assert(np.all(np.diag(L) == 0))

    # Test that W is a consensus matrix
    # W is row stochastic
    assert(np.all(np.isclose(np.sum(W, axis=1), 0)))
    # W is PSD
    assert(np.all(np.linalg.eigvals(W) >= -1e-7))
    # W is symmetric
    assert(np.all(np.isclose(W, W.T)))
    # W second smallest eigenvalue is positive
    assert(sorted(np.linalg.eigvals(W))[1] > 0) #1-np.cos(np.pi/n))
    # At least n-1 entries in the lower triangle of W are non-zero
    assert(np.sum(np.tril(W,-1) != 0) >= n-1)

    # Test that L and W are related
    # Z = 2*np.eye(n) - L - L.T
    D = Z - W
    # Z - W is PSD
    v = np.linalg.eigvals(D)
    assert(np.all(v >= -1e-7))
