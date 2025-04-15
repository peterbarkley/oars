from numpy import cos, pi, eye, ones
import cvxpy as cvx
from .core import getIncidenceFixed

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
        c = 2*(1-cos(pi/n))
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
        cons += [cvx.sum(Z, axis=0) == 0] # Z sums to zero
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
    cons += [Z >> W, # Z - W is PSD            
            cvx.bmat([[Z, Q-K.T],
                      [Q.T-K, eye(m)]]) >> 0,
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
    cons += [W + n**2 * ones((n, n)) >> c*eye(n)] # Fiedler value constraint #cvx.trace(Z) <= n**2,

    return Z, W, Q, K, cons    

def getCabraTightZ(n, m, verbose=False, solverargs={}, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    ZminusU = cvx.bmat([[Z, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(ZminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value)
    if verbose > 0:
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U

def getCabraMinZ(n, m, verbose=False, solverargs={}, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    obj = cvx.Minimize(cvx.lambda_max(Z))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value)
    if verbose > 0:
        print('Z', Z.value)
        print('W', W.value)
        print('U', U)
        print('Q', Q.value)
        print('K', K.value)
    return Z.value, W.value, Q.value, K.value, U

def getCabraMinZD(n, m, D=None, alpha=None, verbose=False, solverargs={}, **kwargs):
    if alpha is None:
        alpha = cvx.Variable(1)
    if D is None:
        D = ones(n)
    
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    cons += [cvx.diag(Z) == alpha*D]
    obj = cvx.Minimize(cvx.lambda_max(Z))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value)
    if verbose > 0:
        print('Z', Z.value)
        print('W', W.value)
        print('U', U)
        print('Q', Q.value)
        print('K', K.value)
    return Z.value, W.value, Q.value, K.value, U

def getCabraTightW(n, m, verbose=False, solverargs={}, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    WminusU = cvx.bmat([[W, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(WminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value)
    if verbose > 0:
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U

def getCabraTightZD(n, m, D, verbose=0, solverargs={}, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    cons += [cvx.diag(Z) == D]
    ZminusU = cvx.bmat([[Z, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(ZminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
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

def getCabraTightWD(n, m, D, verbose=0, solverargs={}, **kwargs):
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    cons += [cvx.diag(Z) == D]
    WminusU = cvx.bmat([[W, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(WminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
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

