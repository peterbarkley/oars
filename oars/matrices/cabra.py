from numpy import cos, pi, eye, ones, tril, diag
from numpy.linalg import inv, eigvals, norm
import cvxpy as cvx
from .core import getIncidenceFixed, getMfromWCholesky
from scipy.linalg import sqrtm

def getCabra(n, m, fixed_Z={}, fixed_W={}, fixed_Q={}, fixed_K={}, cutoffs=None, beta=None, c=None, adj=False, **kwargs):
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
    if beta is None:
        beta = ones(m)
    cons = []

    # Variables
    if m > 0:
        Q = cvx.Variable((n,m))
        K = cvx.Variable((m,n))
    else:
        Q = None
        K = None
    if not adj:
        W = cvx.Variable((n,n), PSD=True)
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
    if m > 0:
        cons += [cvx.bmat([[Z, Q-K.T],
                        [Q.T-K, diag(beta)]]) >> 0,
                cvx.sum(K, axis=1) == 1,
                cvx.sum(Q, axis=0) == 1]
        cons += [K[j,i] == 0 for j in range(m) for i in range(cutoffs[j], n)]
        cons += [Q[i,j] == 0 for j in range(m) for i in range(cutoffs[j])]
        cons += [Q[idx] == val for idx,val in fixed_Q.items()]
        cons += [K[idx] == val for idx,val in fixed_K.items()]

    # Set fixed Z and W values
    if not adj:
        cons += [Z[idx] == val for idx,val in fixed_Z.items()]
        cons += [W[idx] == val for idx,val in fixed_W.items()]

    
    cons += [Z >> W, # Z - W is PSD      
             W + n**2 * ones((n, n)) >> c*eye(n)] # Fiedler value constraint #cvx.trace(Z) <= n**2,

    return Z, W, Q, K, cons    

# def getCabraTightZ(n, m, verbose=False, solverargs={}, **kwargs):
#     Z, W, Q, K, cons = getCabra(n, m, **kwargs)
#     obj = cvx.Minimize(cvx.lambda_max(ZminusU))
#     prob = cvx.Problem(obj, cons)
#     prob.solve(verbose=(verbose==2), **solverargs)
    
#     U = getU(Q.value, K.value)
#     if verbose > 0:
#         print(Z.value)
#         print(W.value)
#         print(K.value)
#         print(Q.value)
#         print(U)
#     return Z.value, W.value, Q.value, K.value, U

def getCabraMinZ(n, m, beta=None, verbose=False, solverargs={}, **kwargs):
    if beta is None:
        beta = ones(m)
    Z, W, Q, K, cons = getCabra(n, m, beta=beta, **kwargs)
    obj = cvx.Minimize(cvx.lambda_max(Z))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    if m > 0:
        Qv = Q.value
        Kv = K.value
        U = getU(Qv, Kv, beta)
    else:
        U = Qv = Kv = None
    if verbose > 0:
        print('Z', Z.value)
        print('W', W.value)
        if m > 0:
            print('Q', Qv)
            print('K', Kv)
            print('U', U)
    return Z.value, W.value, Qv, Kv, U

def getCabraMinZD(n, m, D=None, alpha=None, beta=None, verbose=False, solverargs={}, **kwargs):
    
    if beta is None:
        beta = ones(m)
    if alpha is None:
        alpha = cvx.Variable(1)
    if D is None:
        D = ones(n)
    
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    cons += [cvx.diag(Z) == alpha*D]
    obj = cvx.Minimize(cvx.lambda_max(Z))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value, beta)
    if verbose > 0:
        print('Z', Z.value)
        print('W', W.value)
        print('U', U)
        print('Q', Q.value)
        print('K', K.value)
    return Z.value, W.value, Q.value, K.value, U

def getCabraTightW(n, m, beta=None, verbose=False, solverargs={}, **kwargs):
    if beta is None:
        beta = ones(m)
    Z, W, Q, K, cons = getCabra(n, m, beta=beta, **kwargs)
    WminusU = cvx.bmat([[W, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(WminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value, beta)
    if verbose > 0:
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U

def getCabraTightZD(n, m, D, beta=None, verbose=0, solverargs={}, **kwargs):
    if beta is None:
        beta = ones(m)
    Z, W, Q, K, cons = getCabra(n, m, beta=beta, **kwargs)
    cons += [cvx.diag(Z) == D]
    ZminusU = cvx.bmat([[Z, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(ZminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value, beta)
    if verbose > 0:
        print(prob.status)
        print(prob.value)
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U

def getCabraTightWD(n, m, D, beta=None, verbose=0, solverargs={}, **kwargs):
    if beta is None:
        beta = ones(m)
    Z, W, Q, K, cons = getCabra(n, m, **kwargs)
    cons += [cvx.diag(Z) == D]
    WminusU = cvx.bmat([[W, Q-K.T],
                        [Q.T-K, eye(m)]])
    obj = cvx.Minimize(cvx.lambda_max(WminusU))
    prob = cvx.Problem(obj, cons)
    prob.solve(verbose=(verbose==2), **solverargs)
    
    U = getU(Q.value, K.value, beta)
    if verbose > 0:
        print(prob.status)
        print(prob.value)
        print(Z.value)
        print(W.value)
        print(K.value)
        print(Q.value)
        print(U)
    return Z.value, W.value, Q.value, K.value, U

def trilc(Z):
    n = Z.shape[0]
    L = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            L[i][j] = Z[i,j]
    return cvx.bmat(L)

def getCabraClose(n, m=0, da=None, db=None, fixed_Z={}, fixed_W={}, fixed_Q={}, fixed_K={}, cutoffs=None, c=None, beta=None, alpha=0.5, LD=False, verbose=False, adj=False, QKnn=True):
    """
        Args:
        n (int): number of maximal monotone operators (resolvents)
        m (int): number of cocoercive operators (direct)
        da (array): target diagonal for Z coming from A
        db (array): target diagonal for Z coming from B as diag(Q@db)
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
    """
    if c is None:
        c = min(alpha, 0.5)*(1-cos(pi/n))
    if da is None:
        da = ones(n)
    if cutoffs is None:
        cutoffs = [min(n-1, j) for j in range(1, m+1)]
    if beta is None:
        beta = ones(m)
    no_W = fixed_Z == fixed_W
    cons = []

    # Variables
    if not adj:
        Z = cvx.Variable((n,n), PSD=True)
        cons += [cvx.sum(Z, axis=0) == 0] # Z sums to zero
        cons += [Z[idx] == val for idx,val in fixed_Z.items()]
        if not no_W: 
            W = cvx.Variable((n,n), PSD=True)
            cons += [W[idx] == val for idx,val in fixed_W.items()]
            cons += [Z >> W,
                     W + n**2 * ones((n, n)) >> c*eye(n)]
    else:        
        Mz = getIncidenceFixed(n, fixed_Z)
        ez = Mz.shape[0]
        z = cvx.Variable(ez, nonneg=True)
        Z = Mz.T @ cvx.diag(z) @ Mz
        if not no_W:            
            Mw = getIncidenceFixed(n, fixed_W)
            ew = Mw.shape[0]
            w = cvx.Variable(ew, nonneg=True)            
            W = Mw.T @ cvx.diag(w) @ Mw

    if no_W:
        cons += [cvx.lambda_sum_smallest(Z, 2) >= c]
    tgt = diag(da)
    if m > 0:
        Q = cvx.Variable((n,m), nonneg=QKnn)
        K = cvx.Variable((m,n), nonneg=QKnn)
        cons += [cvx.bmat([[Z, Q-K.T],
                            [Q.T-K, diag(beta)]]) >> 0,
                cvx.sum(K, axis=1) == 1,
                cvx.sum(Q, axis=0) == 1]
        cons += [K[j,i] == 0 for j in range(m) for i in range(cutoffs[j], n)]
        cons += [Q[i,j] == 0 for j in range(m) for i in range(cutoffs[j])]
        cons += [Q[idx] == val for idx,val in fixed_Q.items()]
        cons += [K[idx] == val for idx,val in fixed_K.items()]
        tgt = tgt + cvx.diag(Q@db)
        

    # Objective
    if LD:
        # Not tested - implementation may need a tweak to fix the tril function on a cvxpy var
        obj = cvx.Minimize(cvx.norm(cvx.diag(cvx.diag(Z)) + 2*trilc(Z) - alpha*tgt))
    else:
        obj = cvx.Minimize(cvx.norm(Z - alpha*tgt))
    prob = cvx.Problem(obj, cons)
    prob.solve()
    if m > 0:
        Qv = Q.value
        Kv = K.value
        Uv = getU(Qv, Kv, beta)
        tgt = tgt.value
    else:
        Qv = None
        Kv = None
        Uv = None
    if not no_W:
        Wv = W.value
    else:
        Wv = None
    if verbose:
        print(prob.status, prob.value)
        print('tgt', alpha*tgt)
        print('Z', Z.value)
        if m > 0:
            print('U', Uv)
            print('K', Kv)
            print('Q', Qv)
        if not no_W:
            print('W', Wv)

    
    return Z.value, Qv, Kv, Wv, Uv

def getU(Q, K, beta=None):
    if beta is None:
        beta = ones(Q.shape[1])
    bi = diag(1/beta)
    return (Q - K.T)@bi@(Q.T - K)

def getMinConditionW(Z, Q, K, da, db, alpha=1.0, c=None, verbose=False):
    n = Z.shape[0]

    # Find D - 2L + alpha(A + QBK)
    H = diag(diag(Z)) + 2*tril(Z,-1) + alpha*diag(da)
    if db is not None and len(db) > 0:
        H += alpha*Q@diag(db)@K
    V = inv(H)
    rV = sqrtm(V)

    c = alpha*(1-cos(pi/n))

    W = cvx.Variable((n,n), PSD=True)
    t = cvx.Variable(1, pos=True)
    s = cvx.Variable(1, pos=True)
    rvw = rV@W@rV.T
    cons = [cvx.sum(W, axis=0) == 0,
            cvx.lambda_sum_smallest(W, 2) >= c,
            # s*eye(n) << rvw + 10*ones((n,n)),
            cvx.lambda_sum_smallest(rvw, 2) >= s,
            rvw << t*eye(n),
            Z >> W]
    obj = cvx.Minimize(t/s)
    problem = cvx.Problem(obj, cons)
    problem.solve(qcp=True, solver=cvx.SCS)
    assert problem.is_dqcp()
    if verbose:
        print("Optimal value: ", problem.value)
        print("W", W.value)
        M = getMfromWCholesky(W.value)
        print("M", M)
        print("H = D - 2L + alpha (DA + Q DB K) ", H)
        print("H eigs", eigvals(H))
        print("M H^-1 M", M@V@M.T)
        print("M H^-1 M eigs", eigvals(M@V@M.T))
    return W.value

def getWmatch(Z, Q, K, da, db, alpha=1.0, c=None, verbose=False):
    n = Z.shape[0]

    # Find D - 2L + alpha(A + QBK)
    H = diag(diag(Z)) + 2*tril(Z,-1) + alpha*diag(da)
    if db is not None and len(db) > 0:
        H += alpha*Q@diag(db)@K

    V = inv(H)
    c = min(alpha, 0.5)*(1-cos(pi/n))

    W = cvx.Variable((n,n), PSD=True)
    cons = [cvx.sum(W, axis=0) == 0,
            cvx.lambda_sum_smallest(W, 2) >= c,
            Z >> W]
    obj = cvx.Minimize(cvx.norm(H-W))
    problem = cvx.Problem(obj, cons)
    problem.solve()
    if verbose:
        print("Optimal value: ", problem.value)
        print("W", W.value)
        M = getMfromWCholesky(W.value)
        print("H = D - 2L + alpha (DA + Q DB K) ", H)
        print("H eigs", eigvals(H))
        print("M H^-1 M", M@V@M.T)
        print("M H^-1 M eigs", eigvals(M@V@M.T))
    return W.value

def getMVM(Z, W, Q=None, K=None, da=None, db=None, A=None, alpha=1.0, verbose=False):
    if A is not None:
        AA = A
    else:
        if da is None:
            da = ones(Z.shape[0])
        AA = diag(da)
    M = getMfromWCholesky(W)
    H = diag(diag(Z)) + 2*tril(Z,-1) + alpha*AA
    if db is not None and len(db) > 0:
        H += alpha*Q@diag(db)@K
    V = inv(H)
    eigs = [norm(e) for e in eigvals(M@V@M.T)]
    if verbose:
        print("H = D - 2L + alpha (DA + Q DB K) ", H)
        print("H eigs", eigvals(H))
        print("M H^-1 M", M@V@M.T)

        print("M H^-1 M eigs", eigs)
    return eigs

    
def getWhalf(Z, Q, K, da, db, alpha=1.0, c=None, verbose=False):
    n = Z.shape[0]

    # Find D - 2L + alpha(A + QBK)
    H = diag(diag(Z)) + 2*tril(Z,-1) + alpha*diag(da)
    if db is not None and len(db) > 0:
        H += alpha*Q@diag(db)@K
    V = inv(H)
    rV = sqrtm(V)

    c = min(alpha, 0.5)*(1-cos(pi/n))

    W = cvx.Variable((n,n), PSD=True)
    t = cvx.Variable(1, pos=True)
    s = cvx.Variable(1, pos=True)
    rvw = rV@W@rV.T
    cons = [cvx.sum(W, axis=0) == 0,
            cvx.lambda_sum_smallest(W, 2) >= c,
            # s*eye(n) << rvw + 10*ones((n,n)),
            cvx.lambda_sum_smallest(rvw, 2) >= s,
            rvw << t*eye(n),
            Z >> W]
    obj = cvx.Minimize(cvx.norm(t - 0.5) + cvx.norm(s - 0.5))
    problem = cvx.Problem(obj, cons)
    problem.solve()
    if verbose:
        print('s,t', s.value, t.value)
    return W.value

