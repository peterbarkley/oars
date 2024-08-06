import numpy as np
import cvxpy as cvx

def opsplit_PEP(problem, L, M, operators, gamma=0.5, alpha=1, wrapper="cvxpy", solver=None, verbose=1):
    '''Compute the worst-case guarantee of the Operator Splitting method
    
    Inputs:
    problem is the PEPit problem object
    L is the L matrix
    M is the M matrix
    operators is the list of proximal operators
    gamma is the step size
    alpha is the proximal step size
    wrapper is the wrapper for the solver
    solver is the solver to use
    verbose is the verbosity level

    Returns:
    pepit_tau: the worst-case contraction rate
    problem: the PEPit problem object
    '''
    from PEPit import null_point
    from PEPit.primitive_steps import proximal_step
    # Define the starting points w0 and w1
    d, n = M.shape
    w0 = [problem.set_initial_point() for _ in range(d)]
    w1 = [problem.set_initial_point() for _ in range(d)]

    # Set the initial constraint that is the distance between w0 and w1
    problem.set_initial_condition(sum((w0[i] - w1[i]) ** 2 for i in range(d)) <= 1)

    # Compute one step of the Operator Splitting starting from w0
    x0 = []
    for i in range(n):
        Lx = sum((L[i, j]*x0[j] for j in range(i)), start=null_point)
        x0i, _, _ = proximal_step(-M.T[i,:]@w0 + Lx, operators[i], alpha)
        x0.append(x0i)

    z0 = [w0[r] + gamma * M[r,:]@x0 for r in range(d)]

    # Compute one step of the Operator Splitting starting from w1
    x1 = []
    for i in range(n):
        Lx = sum((L[i, j]*x1[j] for j in range(i)), start=null_point)
        x1i, _, _ = proximal_step(-M.T[i,:]@w1 + Lx, operators[i], alpha)
        x1.append(x1i)

    z1 = [w1[r] + gamma * M[r,:]@x1 for r in range(d)]

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric(sum((z0[i] - z1[i]) ** 2 for i in range(d)))

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method
    return pepit_tau, problem 

def operator_splitting_W(problem, L, W, operators, gamma=0.5, alpha=1, wrapper="cvxpy", solver=None, verbose=1):
    '''Compute the worst-case guarantee of the Operator Splitting iteration in dimension n ($v=v-\\gamma W x$)
    
    Inputs:
    problem is the PEPit problem object
    L is the n by n L matrix
    W is the n by n W matrix
    operators is the list of proximal operators
    gamma is the step size
    alpha is the proximal step size
    wrapper is the wrapper for the solver
    solver is the solver to use
    verbose is the verbosity level

    Returns:
    tau: the worst-case contraction rate
    '''
    from PEPit import null_point
    from PEPit.primitive_steps import proximal_step
    # Define the starting points v0, v1
    n = W.shape[0]
    v0 = []
    for r in range(n-1):
        v0.append(problem.set_initial_point())
    v0.append(-1*sum((v0[i] for i in range(n-1)), start=null_point))

    v1 = []
    for r in range(n-1):
        v1.append(problem.set_initial_point())
    v1.append(-1*sum((v1[i] for i in range(n-1)), start=null_point))


    # Set the initial constraint that is the distance between v0 and v1
    problem.set_initial_condition(sum((v0[i] - v1[i]) ** 2 for i in range(n)) <= 1)

    # Compute one step of the Operator Splitting starting from v0
    x0 = []
    for i in range(n):
        Lx = sum((L[i, j]*x0[j] for j in range(i)), start=null_point)
        x0i, _, _ = proximal_step(v0[i] + Lx, operators[i], alpha)
        x0.append(x0i)

    z0 = []
    for r in range(n):
        z0.append(v0[r] - gamma * W[r,:]@x0) 

    # Compute one step of the Operator Splitting starting from v1
    x1 = []
    for i in range(n):
        Lx = sum((L[i, j]*x1[j] for j in range(i)), start=null_point)
        x1i, _, _ = proximal_step(v1[i] + Lx, operators[i], alpha)
        x1.append(x1i)

    z1 = []
    for r in range(n):
        z1.append(v1[r] - gamma * W[r,:]@x1) 

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric(sum((z0[i] - z1[i]) ** 2 for i in range(n)))

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return tau 

def getReducedContractionFactor(Z, M, ls, mus, gamma=0.5, verbose=False):
    '''
    Get contraction factor for resolvent splitting via PEP

    Args:
        Z (ndarray): Z matrix n x n numpy array
        M (ndarray): M matrix d x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        gamma (float): step size
        verbose (bool): verbose output

    Returns:
        tau (float): contraction factor
        
    '''
    d, n = M.shape

    Ko, Ki, Kmu, Kl = getGrams(Z, M, ls=ls, mus=mus, gamma=gamma)

    # Build SDP
    G = cvx.Variable((d+n, d+n), PSD=True)

    constraints = [cvx.trace(Kmu[i] @ G) >= 0 for i in range(n)]
    constraints += [cvx.trace(Kl[i] @ G) >= 0 for i in range(len(Kl))]
    constraints += [cvx.trace(Ki @ G) == 1]

    objective = cvx.Maximize(cvx.trace(Ko @ G))

    prob = cvx.Problem(objective, constraints)
    prob.solve()
    if prob.status != 'optimal':
        print('Problem not solved')
        return None
    if verbose:
        print(prob.status)
        print(prob.value)
        print(G.value)

        # Duals
        for i in range(len(constraints)):
            print(constraints[i].dual_value)

    return prob.value

def getSubdiffPrimalTau(L, W, ls=None, mus=None, verbose=False):
    '''Get contraction factor for resolvent splitting via self made PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given L (and W)
    over a set of n subdifferentials of l_i smooth, mu_i strongly convex functions

    Inputs:
    L: n x n numpy array of resolvent multipliers
    W: n x n numpy array of consensus multipliers
    ls: size n numpy array of Lipschitz constants (default is 2)
    mus: size n numpy array of strong convexity parameters where mu[i] < l[i] (default is 1)

    Returns:
    tau: contraction factor
    '''
    
    n = L.shape[0]
    if mus is None and ls is None:
        mus = np.ones(n)
        ls = np.ones(n)*2
    Ko, K1, Ki, Ks = getSubdiffGrammians(L, W, ls=ls, mus=mus)
    G = cvx.Variable((2*n, 2*n), PSD=True)

    constraints = [cvx.trace(Ks[i] @ G) >= 0 for i in range(n)]
    constraints += [cvx.trace(Ki @ G) == 1]
    constraints += [cvx.trace(K1 @ G) == 0]

    objective = cvx.Maximize(cvx.trace(Ko @ G))

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print('Problem not solved')
        return None
    if verbose:
        print(prob.status)
        print(prob.value)
        print(G.value)

        # Duals
        for i in range(len(constraints)):
            print(constraints[i].dual_value)

    return prob.value

def getSubdiffW(L, ls=None, mus=None, Wref = None, W_fixed={}, verbose=False):
    '''Get W that minimizes the worst case contraction factor (and that factor)
    for resolvent splitting using L via self made PEP formulation on
    the iteration :math:`v = v - \\gamma W x`
    over a set of n subdifferentials of l_i smooth, mu_i strongly convex functions

    Inputs:
    L: n x n numpy array of resolvent multipliers
    ls: size n numpy array of Lipschitz constants (default is 2)
    mus: size n numpy array of strong convexity parameters where mu[i] < l[i] (default is 1)
    Wref: Base W if only gamma multiplier is desired
    W_fixed: Dict of values to set W[ij] equal to
    verbose: Bool for verbose output

    Returns:
    W: n x n numpy array of optimal consensus multipliers s.t. W1=0
    tau: contraction factor
    '''
    n = L.shape[0]
    _, K1, Ki, Ks = getSubdiffGrammians(L, ls=ls, mus=mus)
    lambda_s = cvx.Variable(n, nonneg=True)
    lambda_one = cvx.Variable(1)
    rho2 = cvx.Variable(1)
   
    if Wref is not None:
        gam = cvx.Variable(1)
        Wvar = gam*Wref
    else:
        Wvar = cvx.Variable(L.shape, symmetric=True)

    # Define the dual problem
    S = sum(lambda_s[i]*Ks[i] for i in range(n))
    obvec = cvx.vstack([np.eye(n), -Wvar])

    Stilde = cvx.bmat([[lambda_one*K1 + rho2*Ki-S, obvec], [obvec.T, np.eye(n)]])
    constraints = [Stilde >> 0, cvx.sum(Wvar, axis=1) == 0]

    # Fixed W
    for idx, val in W_fixed.items():
        constraints.append(Wvar[idx] == val)

    obj = cvx.Minimize(rho2)

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if verbose:
        print(prob.status)
        print('tau', rho2.value)
        print('W', Wvar.value)
        print('lambda s', lambda_s.value)
        print('lambda one', lambda_one.value)
        print('S', S.value)
        print('St', Stilde.value)

    return Wvar.value, prob.value

def getWGamma(L, l=None, mu=None, Wref=None, W_fixed={}, verbose=False):
    '''Use the dual W PEP to get the optimal W

    Inputs:
    L: n x n matrix of resolvent coefficients
    l: list of Lipschitz constants
    mu: list of strong convexity parameters
    Wref: Base W if only gamma is desired
    W_fixed: Dict of values to set W[ij] equal to
    verbose: Bool for verbose output

    Returns:
    W: optimal consensus matrix
    rho2: maximum contraction factor
    '''
    n = L.shape[0]
    Ko, Ki, Kmu, Kl, P = getWGrams(L, W=Wref, l=l, mu=mu)
    lmu = cvx.Variable(len(Kmu), nonneg=True)
    ll = cvx.Variable(len(Kl), nonneg=True)
    rho2 = cvx.Variable(1)
    if Wref is not None:
        gam = cvx.Variable(1)
        W = gam*Wref
    else:
        W = cvx.Variable(L.shape, symmetric=True)

    # Define the dual problem
    S = sum(lmu[i]*Kmu[i] for i in range(len(Kmu))) + sum(ll[i]*Kl[i] for i in range(len(Kl)))
    obvec = cvx.vstack([P, -W])

    Stilde = cvx.bmat([[rho2*Ki-S, obvec], [obvec.T, np.eye(n)]])
    constraints = [Stilde >> 0,
                cvx.sum(W, axis=1) == 0]

    # Fixed W
    for idx, val in W_fixed.items():
        constraints.append(W[idx] == val)

    obj = cvx.Minimize(rho2)

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if verbose:
        print(prob.status)
        print('tau', rho2.value)
        print('W', W.value)
        print('lambda mu', lmu.value)
        print('lambda l', ll.value)
        print('S', S.value)
        print('St',Stilde.value)

    return W.value, prob.value

def getGamma(Z, M, ls=None, mus=None, verbose=False):
    '''
    Use the dual PEP to get the optimal step size gamma

    Args:
        Z (ndarray): Z matrix n x n numpy array
        M (ndarray): M matrix d x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        verbose (bool): verbose output

    Returns:
        gamma (float): optimal step size
        tau (float): contraction factor
    '''
    Ko, Ki, Kmu, Kl = getGrams(Z, M, ls=ls, mus=mus)
    
    d = M.shape[0]
    # Define dual variables
    lmu = cvx.Variable(len(Kmu), nonneg=True)
    ll = cvx.Variable(len(Kl), nonneg=True)
    rho2 = cvx.Variable(1)
    gam = cvx.Variable(1)

    # Define the dual problem
    S = sum(lmu[i]*Kmu[i] for i in range(len(Kmu))) + sum(ll[i]*Kl[i] for i in range(len(Kl)))
    obvec = cvx.hstack([np.eye(d), cvx.multiply(gam, M)]) # gam*M
    
    Stilde = cvx.bmat([[rho2*Ki-S, obvec.T], [obvec, np.eye(d)]])
    constraints = [Stilde >> 0]

    obj = cvx.Minimize(rho2)

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if verbose:
        print(prob.status)
        print(prob.value)
        print('rho2', rho2.value)
        print('gam', gam.value)
        print(lmu.value)
        print(ll.value)
        print(S.value)
        print(Stilde.value)

    return gam.value[0], prob.value

def getSubdiffGrammians(L, W=None, ls=None, mus=None):
    '''Get the matrices for the PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given L (and W)
    over a set of n subdifferentials of l_i smooth, mu_i strongly convex function
    Inputs:
    L: n x n numpy array of resolvent multipliers
    W: n x n numpy array of consensus multipliers
    l: size n numpy array of Lipschitz constants (default is 2)
    mu: size n numpy array of strong convexity parameters where mu[i] < l[i] (default is 1)

    Returns:
    Ko: 2n x 2n matrix for the objective function (None if W is None)
    K1: 2n x 2n matrix for the constraint that v sums to 0 ([11^T, 0],[0, 0]])
    Ki: 2n x 2n matrix for the equality constraint ([[I, 0],[0,0]])
    Ks: list of n 2n x 2n matrices for the constraints
    '''
    n = L.shape[0]
    if mus is None or ls is None:
        mus = np.ones(n)
        ls = np.ones(n)*2
    Ks = []
    for i in range(n):
        xterm = np.zeros(2*n)
        xterm[n+i] = 1
        vLx_term = np.zeros(2*n)
        vLx_term[i] = 1
        vLx_term[n:] = L[i,:]
        vLx_term -= xterm
        ml_block = 0.5*np.outer(vLx_term, xterm)
        ml_sym = ml_block + ml_block.T

        l_sym = np.outer(vLx_term, vLx_term)
        mu_sym = np.outer(xterm, xterm)
        Ksi = ml_sym*(1+mus[i]/ls[i]) - (1/ls[i])*l_sym - mus[i]*mu_sym
        Ks.append(Ksi.copy())

    ones = np.ones((n,n))
    zero = np.zeros((n,n))
    Ki = np.block([[np.eye(n), zero],[zero, zero]])
    K1 = np.block([[ones, zero],[zero, zero]])
    if W is not None:
        Ko = np.block([[np.eye(n), -W], [-W, W.T@W]])
    else:
        Ko = None
    return Ko, K1, Ki, Ks

def getWGrams(L, W=None, l=None, mu=None, gamma=1, verbose=False):
    '''Get the matrices for the PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given $L$
    over a set of $n$ $l_i$-Lipschitz, $\\mu_i$-strongly convex operators
    Inputs:
    L: n x n numpy array of resolvent multipliers
    W: n x n numpy array of consensus multipliers
    l: size n numpy array of Lipschitz constants
    mu: size n numpy array of strong convexity parameters (mu[i] < l[i])
    gamma: step size (use 1 for the dual PEP matrices)
    verbose: boolean for verbose output

    Returns:
    Ko: (d+n) x (d+n) matrix for the objective function
    Ki: (d+n) x (d+n) matrix for the equality constraint
    Kmu: list of (d+n) x (d+n) matrices for the strong convexity constraints
    Kl: list of (d+n) x (d+n) matrices for the Lipschitz constraints
    P: d x n matrix for the projection
    '''
    n = L.shape[0]
    d = n-1
    
    if mu is None:
        mu = np.ones(n)
    if l is None:
        l = np.ones(n)*2

    P = np.zeros((d,n))

    # Set first d columns of P to be the identity matrix
    P[:d,:d] = np.eye(d)

    # Set the last column of P to be the sum of the first 5 columns
    P[:d,d] = -np.ones(d)
    E = np.eye(d)
    F = np.eye(n)

    # Ki
    Ki = cvx.bmat([[P@P.T, np.zeros((d, n))], [np.zeros((n, d)), np.zeros((n, n))]])

    # Ko
    if W is not None:
        Ko = cvx.bmat([[P@P.T, -gamma*P@W.T], [-gamma*W@P.T, gamma**2 * W.T @ W]])
    else:
        Ko = None
        
    # Kmu
    Kmu = []
    Kmu = [cvx.bmat([[np.zeros((d, d)), E[:, 0].reshape(-1, 1), np.zeros((d, n-1))],
                    [E[:, 0].reshape(1, -1), 2*np.diag([-1-mu[0]]), np.zeros((1, n-1))],
                    [np.zeros((n-1, d)), np.zeros((n-1, 1)), np.zeros((n-1, n-1))]])]
    for i in range(1,n-1):
        Kmu.append(cvx.bmat([[np.zeros((d, d)), np.zeros((d, i)), E[:, i].reshape(-1, 1), np.zeros((d, n-i-1))],
                [np.zeros((i, d)), np.zeros((i, i)), L[i,:i].reshape(-1, 1), np.zeros((i, n-i-1))],
                [E[:, i].reshape(1, -1), L[i,:i].reshape(1, -1), 2*np.diag([-1-mu[i]]), np.zeros((1,n-i-1))],
                [np.zeros((n-i-1, d)), np.zeros((n-i-1, i)), np.zeros((n-i-1, 1)), np.zeros((n-i-1, n-i-1))]]))

    Kmu.append(cvx.bmat([[np.zeros((d, d)), np.zeros((d, n-1)), -np.ones((d, 1))],
                        [np.zeros((n-1, d)), np.zeros((n-1, n-1)), L[n-1, :n-1].reshape(-1, 1)],
                        [-np.ones((1, d)), L[n-1, :n-1].reshape(1, -1), 2*np.diag([-1-mu[n-1]])]]))
    # Kl[i] = [-E[:,i]@E[:,i].T, -E[:,i]@(L[i,:] - F[i, :]).T; -(L[i,:] - F[i,:)@E[i,;], -(L[i,:] - e_i)@(L[i,:] - e_i).T + l_i * e_i @ e_i.T ]
    Kl = []
    for i in range(d):
        if l[i] != np.inf:
            Le = L[i,:] - F[i,:]
            Kl.append(cvx.bmat([
                [-np.outer(E[:,i],E[:,i]), -np.outer(E[:,i], Le)],
                [-np.outer(Le, E[:, i]), -np.outer(Le, Le) + l[i]**2 * np.outer(F[i,:], F[i,:])]]))

    if l[n-1] != np.inf:
        Le = L[n-1,:] - F[n-1,:]
        one = np.ones(d)
        Kl.append(cvx.bmat([
            [-np.outer(one,one), np.outer(one, Le)],
            [np.outer(Le, one), -np.outer(Le, Le) + l[n-1]**2 * np.outer(F[n-1,:], F[n-1,:])]]))

    if verbose:
        #print Ki, Ko, Kmu, Kl
        for i in range(n):
            print('kmu', i)
            print(Kmu[i].value)
        print('kmu_sum')
        print(sum(Kmu).value)
        Kmu_pred = cvx.bmat([[np.zeros((d, d)), P], [P.T, L + L.T - 2*np.eye(n) - 2*np.diag(mu)]])
        print(Kmu_pred.value)
        print('Diff from predicted:', sum((sum(Kmu) - Kmu_pred).value))
        for i in range(len(Kl)):
            print('kl', i)
            print(Kl[i].value)
        print('kl_sum')
        print(sum(Kl).value)
        Kl_pred = cvx.bmat([
                    [-P@P.T, -P@(L-np.eye(n))], 
                    [-(L-np.eye(n)).T@P.T, -(L-np.eye(n)).T@(L-np.eye(n)) + np.diag(l**2)]])
        print(Kl_pred.value)
        print('Diff from predicted:', sum((sum(Kl) - Kl_pred).value))
        print('Ko')
        print(Ko.value)
        print("Ki")
        print(Ki.value)

    return Ko, Ki, Kmu, Kl, P

def getGrams(Z, M, ls=None, mus=None, gamma=1, verbose=False):
    '''
    Get the matrices for the PEP formulation on
    the reduced iteration :math:`z = z + \\gamma M x` for a given :math:`Z, M`
    over a set of `n` `l_i`-Lipschitz, :math:`\\mu_i`-strongly convex operators

    Args:
        Z (ndarray): Z matrix n x n numpy array
        M (ndarray): M matrix d x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        gamma (float): step size (default is 1 for the dual PEP)
        verbose (bool): verbose output

    Returns:
    Ko: (d+n) x (d+n) matrix for the objective function
    Ko = [I, gamma*M; gamma*M^T, gamma^2 * M^T M]
    Ki: (d+n) x (d+n) matrix for the equality constraint
    Ki = [I, 0; 0, 0]
    Kmu: list of (d+n) x (d+n) matrices for the strong convexity constraints
    Kmu[i] = [0, 0, -M[:,i]/2, 0; 
              0, 0, L[1:i-1,i], 0; 
              -M[:,i].T/2, L[1:i-1,i].T, -1- mu_i, 0; 
              0, 0, 0, 0]
    Kl: list of (d+n) x (d+n) matrices for the Lipschitz constraints
    Kl[i] = [-M[:,i]@M[:,i].T, M[:,i]@(L[i,:] - e_i).T;
             (L[i,:] - e_i)@M[:,i].T, -(L[i,:] - e_i)@(L[i,:] - e_i).T + l_i * e_i @ e_i.T ]
    '''

    n = Z.shape[0]
    L = -np.tril(Z, -1)
    d = M.shape[0]
    Kl = []
    Kmu = []

    if mus is None:
        mu = np.ones(n)
    if ls is None:
        l = np.ones(n)*2

    # Set the values of the objective function matrix Ko
    zerodd = np.zeros((d, d))
    zerodn = np.zeros((d, n))
    zerond = np.zeros((n, d))
    zeronn = np.zeros((n, n))
    Ko = np.block([[np.eye(d), gamma*M], 
                   [gamma*M.T, gamma**2 * M.T @ M]])

    # Ki = [I, 0; 0, 0]
    Ki = np.block([[np.eye(d), zerodn], 
                   [zerond, zeronn]])

    # Kmu   
    Kmus = []
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        eiMi = np.outer(ei, M[:, i])
        eiLi = np.outer(ei, L[i, :])
        eiei = np.outer(ei, ei)
        mublock = np.block([[zerodd, zerodn],
                            [zerond, eiei*(-1-mus[i])]])
        k = np.block([[zerodd, zerodn],
                      [-eiMi, eiLi]])
        Kmus.append(0.5*(k + k.T) + mublock)

    # Kl
    Kls = []
    for i in range(n):
        if ls[i] != np.inf:
            ei = np.zeros(n)
            ei[i] = 1
            MiMi = np.outer(M[:, i], M[:, i])
            MiLi = np.outer(M[:, i], L[i, :] - ei)
            LiLi = np.outer(L[i, :] - ei, L[i, :] - ei)
            eiei = np.outer(ei, ei)
            k = np.block([[-MiMi, MiLi],
                        [MiLi.T, ls[i]**2 * eiei - LiLi]])
            Kls.append(k)
    if verbose:
        print(Ki)
        print(Ko)
        for i in range(n):
            print('kmu', i)
            print(Kmus[i])
        for i in range(n):
            print('kl', i)
            print(Kls[i])

    return Ko, Ki, Kmus, Kls

def getMatrixContraction(cases, names, Lc=2.0, mu=1.0, gamma=0.5, verbose=-1):
    '''
    Get contraction factors for matrix cases
    over n :math:`l`-smooth :math:`\\mu`-strongly convex functions

    Args:
        cases (list): list of cases with L and M
        names (list): list of names
        Lc (float): Lipschitz constant
        mu (float): strong convexity parameter
        gamma (float): step size
        verbose (int): verbosity level

    Returns:
        taus (dict): dictionary of contraction factors
        
    '''
    from PEPit import PEP
    from PEPit.functions import SmoothStronglyConvexFunction
    taus = {}
    for i, (L, M) in enumerate(cases):
        
        problem = PEP()
        ops = [problem.declare_function(SmoothStronglyConvexFunction, L=Lc, mu=mu) for _ in range(L.shape[0])]
        
        pepit_tau, problem = opsplit_PEP(problem=problem, gamma=gamma, L=L, M=M, operators=ops, wrapper="cvxpy", solver=None, verbose=verbose)
        print(names[i], pepit_tau)
        taus[names[i]] = pepit_tau

    return taus

