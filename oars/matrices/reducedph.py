import cvxpy as cvx
import numpy as np

def getCore(p, fixed_Z={}, fixed_W={}, c=None, gamma=1.0, adj=False, **kwargs):
    '''
    Get core variables and constraints for the algorithm design SDP

    :math:`W p = 0`

    :math:`Z p = 0`

    :math:`\\lambda_{1}(W) + \\lambda_{2}(W) \\geq c`

    :math:`Z - W \\succeq 0`

    Args:
        p (arraylike): vector giving the desired null space of Z and W
        fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
        fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
        c (float): connectivity parameter (default 2*(1-cos(pi/n)))
        gamma (float): scaling parameter for Z (default 1.0)
        adj (bool): whether to use the edge adjacency formulation
        kwargs: additional keyword arguments for the algorithm
        
    Returns:
        Z: cvxpy decision variable matrix for Z
        W: cvxpy decision variable matrix for W
        t: cvxpy decision variable for W Fiedler value
        cons: list of cvxpy constraints

    Examples:
        >>> import cvxpy as cvx
        >>> from oars.reducedph import getCore
        >>> p = [.1,.2,.3,.4]
        >>> Z, W, t, cons = getCore(p, fixed_W={(3, 0): 0})
        >>> obj = cvx.Minimize(cvx.norm(Z-W, 'fro') + cvx.norm(cvx.diag(Z) - [1,1,1,1]))
        >>> prob = cvx.Problem(obj, cons)
        >>> prob.solve()
        >>> print(Z.value)
        [[ 1.    -0.169 -0.221  0.   ]
         [-0.169  1.     0.333 -0.708]
         [-0.221  0.333  1.    -0.862]
         [ 0.    -0.708 -0.862  1.   ]]
        >>> print(W.value)
        [[ 1.    -0.169 -0.221  0.   ]
         [-0.169  1.     0.333 -0.708]
         [-0.221  0.333  1.    -0.862]
         [ 0.    -0.708 -0.862  1.   ]]

    '''
    n = len(p)

    if c is None:
        c = 2*(1-np.cos(np.pi/n))

    # Variables
    t = cvx.Variable()
    if not adj:
        W = cvx.Variable((n,n), symmetric=True)
        Z = cvx.Variable((n,n), symmetric=True)
    else:
        Mz = getIncidenceFixed(n, fixed_Z)
        Mw = getIncidenceFixed(p, fixed_W)
        ez = Mz.shape[0]
        ew = Mw.shape[0]

        # Variables
        z = cvx.Variable(ez)
        w = cvx.Variable(ew)
        Z = Mz.T @ cvx.diag(z) @ Mz
        W = Mw.T @ cvx.diag(w) @ Mw

    # Constraints
    nullmat = np.outer(p, p)/np.dot(p,p)
    cons = [t >= c, # Connectivity constraint
            W >> t*(np.eye(n) - nullmat), # Fiedler value constraint            
            gamma*Z >> W, # Z - W is PSD            
            W@p == 0, # p in null space of W
            Z@p == 0] # p in null space of Z

    # Set fixed Z and W values
    cons += [Z[idx] == val for idx,val in fixed_Z.items() if not adj or val != 0]
    cons += [W[idx] == val for idx,val in fixed_W.items() if not adj or val != 0]

    return Z, W, t, cons    

def getSmall(p, fixed_Z={}, c=None):
    n = len(p)

    if c is None:
        c = 2*(1-np.cos(np.pi/n))

    # Variables
    t = cvx.Variable()
    Z = cvx.Variable((n,n), symmetric=True)
    nullmat = np.outer(p, p)/np.dot(p,p)
    cons = [t >= c, # Connectivity constraint
            Z >> t*(np.eye(n) - nullmat), # Fiedler value constraint   
            Z@p == 0] # p in null space of Z

    # Set fixed Z and W values
    cons += [Z[idx] == val for idx,val in fixed_Z.items()]

    return Z, t, cons    

def getOneDiag(p, **kwargs):
    n = len(p)
    Z, t, cons = getSmall(p, **kwargs)
    obj = cvx.Minimize(cvx.norm(cvx.diag(Z) - np.ones(n)))
    prob = cvx.Problem(obj, cons)
    prob.solve()

    if prob.status == 'optimal':
        return Z.value
    return None

def getIncidenceFixed(p, fixed):
    '''
    Converts fixed dictionary to incidence matrix

    Args:
        p (arraylike): vector giving the desired null space of Z and W
        fixed (dict): dictionary with entries (r,c): 0 for edges to exclude

    Returns:
        M (ndarray): m x n numpy array of incidence matrix
                     where m is the number of edges
    '''
    n = len(p)
    invp = 1.0/np.array(p)
    M = []
    for i in range(n):
        for j in range(i):
            if fixed.get((i,j), 1) == 0 or fixed.get((j,i), 1) == 0:
                continue
            else:
                row = np.zeros(n)
                row[i] = invp[i]
                row[j] = -invp[j]
                M.append(row)
    return np.array(M)

def postprocess(prob, Z, W):
    '''
    Postprocess the results of the optimization

    Args:
        prob (cvxpy problem): cvxpy problem object
        Z (cvxpy variable): n x n cvxpy decision variable matrix for Z
        W (cvxpy variable): n x n cvxpy decision variable matrix for W

    Returns:
        Z (ndarray): n x n numpy array of resolvent multipliers
        W (ndarray): n x n numpy array of consensus multipliers
    '''

    if prob.status == 'infeasible':
        Z = None
        W = None

    return Z, W

def getMinSpectralDifference(p, verbose=False, **kwargs):
    '''
    Find resolvent matrix Z and consensus matrix W
    that minimize :math:`\\|Z-W\\|`

    Args:
        p (int): number of resolvents
        kwargs: keyword arguments

            - fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
            - fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
            - c (float): connectivity parameter
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation

    Returns:
        Z (ndarray): resolvent matrix
        W (ndarray): consensus matrix

    Examples:
        >>> from oars.matrices import getMinSpectralDifference
        >>> p = [.1,.2,.3,.4]
        >>> Z, W = getMinSpectralDifference(p, fixed_W={(3, 0): 0}, fixed_Z={(1, 0): 0})
        >>> print(Z)
        [[ 1.015  0.    -0.338 -0.   ]
         [ 0.     0.826 -0.236 -0.236]
         [-0.338 -0.236  0.891 -0.466]
         [-0.    -0.236 -0.466  0.468]]
        >>> print(W)
        [[ 1.015  0.    -0.338 -0.   ]
         [ 0.     0.826 -0.236 -0.236]
         [-0.338 -0.236  0.891 -0.466]
         [-0.    -0.236 -0.466  0.468]]

    '''

    Z, W, t, cons = getCore(p=p, verbose=verbose, **kwargs)

    # Objective function
    obj = cvx.Minimize(cvx.norm(Z-W))
    
    # Solve
    prob = cvx.Problem(obj, cons)
    prob.solve()

    # Print results
    if verbose:
        print(prob.status)
        print(prob.value)
        print(Z.value)
        print(W.value)

    return postprocess(prob, Z.value, W.value)

def getMaxConnectivity(p, z_weight=1.0, w_weight=1.0, verbose=False, **kwargs):
    '''
    Find resolvent matrix Z and consensus matrix W
    that maximize the sum of the algebraic connectivity for W and Z

    Args:
        p (arraylike): vector giving the desired null space of Z and W
        z_weight (float): weight for Z Fiedler value
        w_weight (float): weight for W Fiedler value
        **kwargs: keyword arguments for verbosity and cvxpy solver

            - fixed_Z (dict): dictionary of fixed Z values with keys as (i,j) tuples
            - fixed_W (dict): dictionary of fixed W values with keys as (i,j) tuples
            - c (float): connectivity parameter
            - gamma (float): scaling parameter for Z
            - adj (bool): whether to use the edge adjacency formulation

    Returns:
        Z (ndarray): resolvent matrix
        W (ndarray): consensus matrix

    Examples:
        >>> from oars.matrices import getMaxConnectivity
        >>> p = [.1,.2,.3,.4]
        >>> Z, W = getMaxConnectivity(p, fixed_W={(3, 0): 0}, fixed_Z={(1, 0): 0})
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

    Z, W, t, cons = getCore(p=p, verbose=verbose, **kwargs)

    # Additional variable
    s = cvx.Variable()

    # Constraints
    nullmat = np.outer(p, p)/np.dot(p,p)
    n = len(p)
    cons = cons[1:] # Remove previous connectivity constraint
    cons += [Z >> s*(np.eye(n) - nullmat)] # Fiedler value constraint    

    # Solve
    obj = cvx.Maximize(w_weight*t + z_weight*s)
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

def getOnes(dim):
    '''Returns a matrix with 1 on the diagonal and -1/(n-1) elsewhere'''
    return (np.eye(dim)*(dim) - np.ones((dim, dim)))/(dim-1)

def getPH(p):

    return getMix(p, np.ones((len(p), len(p))))

def getMix(p, X):

    Z = np.diag(np.array(p)**(-1)*(X@p))
    return Z - X

def getTwoBlock(p):

    assert(len(p) % 2 == 0)
    n = len(p)//2
    Z = np.zeros((n,n))
    O = np.ones((n,n))/n
    X = np.block([[Z, O],
                  [O, Z]])
    return getMix(p, X)


def testMatrices(p, Z, W):
    """Test that Z and W are valid consensus and resolvent matrices with p"""

    # Z sums to 0
    assert(np.all(np.isclose(Z@p, 0)))
    
    # Z is symmetric
    assert(np.all(np.isclose(Z, Z.T)))

    # Z is PSD
    assert(np.all(np.linalg.eigvals(Z) >= -1e-7))

    # W is row stochastic
    assert(np.all(np.isclose(W@p, 0)))

    # W is symmetric
    assert(np.all(np.isclose(W, W.T)))

    # W is PSD
    assert(np.all(np.linalg.eigvals(W) >= -1e-7))

    # W second smallest eigenvalue is positive
    assert(sorted(np.linalg.eigvals(W))[1] > 0) 

    # Z - W is PSD
    D = Z - W
    v = np.linalg.eigvals(D)
    assert(np.all(v >= -1e-7))
