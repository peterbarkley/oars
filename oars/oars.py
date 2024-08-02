# Algorithm Design Functions
import numpy as np
import cvxpy as cvx

def getCore(n, fixed_Z={}, fixed_W={}, c=None, eps=0.0, gamma=1.0, adj=False, **kwargs):
    '''Get core variables and constraints for the algorithm
    inputs:
    n: number of nodes
    fixed_Z: dictionary of fixed Z values with keys as (i,j) tuples
    fixed_W: dictionary of fixed W values with keys as (i,j) tuples
    c: connectivity parameter
    eps: epsilon for Z[0,0] = 2 + eps constraint
    adj: whether to use the edge adjacency formulation
    outputs:
    Z: n x n cvxpy decision variable matrix for Z
    W: n x n cvxpy decision variable matrix for W
    cons: list of cvxpy constraints
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
    
    cons += [Z[i,i] == Z[0,0] for i in range(1,n)] # Z diagonal entries equal Z[0,0]

    # Set fixed L and W values
    if not adj:
        cons += [Z[idx] == val for idx,val in fixed_Z.items()]
        cons += [W[idx] == val for idx,val in fixed_W.items()]

    return Z, W, cons    