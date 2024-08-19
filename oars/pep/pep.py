import numpy as np
import cvxpy as cvx

class operator(object):
    """
    Class for the operator in the PEP formulation

    Attributes:
        name (str): name of the operator

    Methods:
        get_class_matrices(self)

    """

    def __init__(self, name=None):
        self.name = name

    def get_class_matrices(self):
        """
        Must be implemented in the child class
        """
        raise NotImplementedError("Must be implemented in the child class")
    
class LipschitzStronglyMonotoneOperator(operator):

    def __init__(self, L, mu, name=None):
        super().__init__(name)
        self.L = L
        self.mu = mu

    def get_class_matrices(self, i, Z, alpha=1):
        matrices = [getMonotoneMatrix(self.mu, i, Z, alpha)]
        if self.L != np.inf:
            matrices += [getLipschitzMatrix(self.L, i, Z, alpha)]
        
        return matrices

    def get_reduced_class_matrices(self, i, Z, M, alpha=1):
        matrices = [getRedMonotoneMatrix(self.mu, i, Z, M, alpha)]
        if self.L != np.inf:
            matrices += [getRedLipschitzMatrix(self.L, i, Z, M, alpha)]

        return matrices

def getLipschitzMatrix(lipschitz, i, Z, alpha=1):
    """
    Get the Lipschitz matrix for the operator
    implements the interpolation requirement given by:
    :math:`L_i^2 \\|x_i\\|^2 - \\frac{1}{\\alpha^2} \\|v_i  - \\sum_{j=1}^{i-1} Z_{ij} x_j - x_i\\|^2 \\geq 0`

    Args:
        lipschitz (float): Lipschitz constant
        i (int): index of the operator
        Z (ndarray): Z matrix
        alpha (float): proximal scaling parameter. Default is 1.

    Returns:
        ndarray: Lipschitz matrix for the operator
    """
    n = Z.shape[0]
    L = - np.tril(Z, -1)
    ei = np.zeros(n)
    ei[i] = 1
    a_sq = (1/alpha)**2
    eiei = np.outer(ei, ei)
    eili = np.outer(ei, L[i, :] - ei)
    lili = np.outer(L[i, :] - ei, L[i, :] - ei)

    return np.block([[-a_sq * eiei, -a_sq * eili],
                     [-a_sq * eili.T, -a_sq * lili + lipschitz**2 * eiei]])

def getRedLipschitzMatrix(lipschitz, i, Z, M, alpha=1):
    """
    Get the reduced Lipschitz matrix for the operator
    implements the interpolation requirement given by:
    :math:`l_i^2 \\|x_i\\|^2 - \\frac{1}{\\alpha^2} \\|\\sum_{j=1}^d -M_{ji} z_j - \\sum_{j=1}^{i-1} Z_{ij} x_j - x_i\\|^2 \\geq 0`

    Args:
        lipschitz (float): Lipschitz constant
        i (int): index of the operator
        Z (ndarray): Z matrix
        M (ndarray): M matrix
        alpha (float): proximal scaling parameter. Default is 1.

    Returns:
        ndarray: Lipschitz matrix for the operator
    """
    d,n = M.shape
    L = - np.tril(Z, -1)
    ei = np.zeros(n)
    ei[i] = 1
    a_sq = (1/alpha)**2
    eiei = np.outer(ei, ei)
    MiMi = np.outer(M[:, i], M[:, i])
    MiLi = np.outer(M[:, i], L[i, :] - ei)
    lili = np.outer(L[i, :] - ei, L[i, :] - ei)

    return np.block([[-a_sq * MiMi, a_sq * MiLi],
                     [a_sq * MiLi.T, -a_sq * lili + lipschitz**2 * eiei]])

def getMonotoneMatrix(mu, i, Z, alpha=1):
    """
    Get the Monotone matrix for the operator
    implements the interpolation requirement given by:
    :math:`\\frac{1}{\\alpha} \\langle x_i, v_i - \\sum_{j=1}^{i-1} Z_{ij} x_j \\rangle - (\\frac{1}{\\alpha}+\\mu_i) \\|x_i\\|^2 \\geq 0`

    Args:
        mu (float): strong convexity parameter
        i (int): index of the operator
        Z (ndarray): Z matrix
        alpha (float): proximal scaling parameter. Default is 1.

    Returns:
        ndarray: Monotone matrix for the operator
    """
    n = Z.shape[0]
    L = - np.tril(Z, -1)
    ei = np.zeros(n)
    ei[i] = 1
    eiei = np.outer(ei, ei)
    eili = np.outer(ei, L[i, :])
    xx = 0.5 * (1/alpha) * (eili + eili.T) - ((1/alpha) + mu) * eiei
    zero = np.zeros((n, n))
    return np.block([[zero, 0.5 * (1/alpha) * eiei],
                     [0.5 * (1/alpha) * eiei, xx]])

def getRedMonotoneMatrix(mu, i, Z, M, alpha=1):
    """
    Get the reduced Monotone matrix for the operator
    implements the interpolation requirement given by:
    :math:`\\frac{1}{\\alpha} \\langle x_i, \\sum_{j=1}^d -M_{ji} z_j - \\sum_{j=1}^{i-1} Z_{ij} x_j \\rangle - (\\frac{1}{\\alpha}+\\mu_i) \\|x_i\\|^2 \\geq 0`

    Args:
        mu (float): strong convexity parameter
        i (int): index of the operator
        Z (ndarray): Z matrix
        M (ndarray): M matrix

    Returns:
        ndarray: Monotone matrix for the operator
    """
    d,n = M.shape
    L = - np.tril(Z, -1)
    ei = np.zeros(n)
    ei[i] = 1
    eiei = np.outer(ei, ei)
    Miei = np.outer(M[:, i], ei)
    eili = np.outer(ei, L[i, :])
    xx = 0.5 * (1/alpha) * (eili + eili.T) - ((1/alpha) + mu) * eiei
    zero = np.zeros((d, d))
    return np.block([[zero, -0.5 * (1/alpha) * Miei],
                     [-0.5 * (1/alpha) * Miei.T, xx]])

class SmoothStronglyConvexFunction(operator):
    """
    Class for the smooth strongly convex subdifferential operator in the PEP formulation

    Attributes:
        L (float): Lipschitz constant
        mu (float): strong convexity parameter

    Methods:
        get_class_matrices(self)
        get_reduced_class_matrices(self)
    """

    def __init__(self, L, mu, name=None):
        super().__init__(name)
        self.L = L
        self.mu = mu

    def get_class_matrices(self, i, Z, alpha=1):
        return [getSmoothStrongMatrix(self.L, self.mu, i, Z, alpha)]

    def get_reduced_class_matrices(self, i, Z, M, alpha=1):
        return [getRedSmoothStrongMatrix(self.L, self.mu, i, Z, M, alpha)]

def getSmoothStrongMatrix(lipschitz, mu, i, Z, alpha=1):
    """
    Get the matrix for the PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given `Z` matrix
    over a set of n subdifferentials of l_i smooth, mu_i strongly convex function
    which implements the interpolation requirement given by:
    :math:`\\frac{1}{\\alpha}(1+\\frac{\\mu_i}{l_i})\\langle x_i, v_i - \\sum_{j=1}^{i-1} Z_{ij} x_j - x_i \\rangle - \\frac{1}{\\alpha^2 l_i}\\|v_i - \\sum_{j=1}^{i-1} Z_{ij} x_j - x_i\\|^2 - \\mu_i \\|x_i\\|^2 \\geq 0`

    Args:
        lipschitz (float): Lipschitz constant
        mu (float): strong convexity parameter
        i (int): index of the operator
        Z (ndarray): Z matrix
        alpha (float): proximal scaling parameter

    Returns:
        ndarray: interpolation matrix for the operator
    """
    n = Z.shape[0]
    L = - np.tril(Z, -1)
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
    K = ml_sym*(1+(mu/lipschitz))/alpha - (1/lipschitz)*l_sym/(alpha**2) - mu*mu_sym
    
    return K

def getRedSmoothStrongMatrix(lipschitz, mu, i, Z, M, alpha=1):
    """
    Get the reduced matrix for the PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given `Z` matrix
    over a set of n subdifferentials of l_i smooth, mu_i strongly convex function
    which implements the interpolation requirement given by:
    :math:`\\frac{1}{\\alpha}(1+\\frac{\\mu_i}{l_i})\\langle x_i, v_i - \\sum_{j=1}^{i-1} Z_{ij} x_j - x_i\\rangle - \\frac{1}{\\alpha^2 l_i}\\|v_i - \\sum_{j=1}^{i-1} Z_{ij} x_j - x_i \\|^2 - \\mu_i \\|x_i\\|^2 \\geq 0`

    Args:
        lipschitz (float): Lipschitz constant
        mu (float): strong convexity parameter
        i (int): index of the operator
        Z (ndarray): Z matrix
        M (ndarray): M matrix
        alpha (float): proximal scaling parameter

    Returns:
        ndarray: interpolation matrix for the operator
    """

    d,n = M.shape
    L = - np.tril(Z, -1)
    xterm = np.zeros(d+n)
    xterm[d+i] = 1
    MwLx_term = np.zeros(n+d)
    MwLx_term[:d] = -M[:,i].T
    MwLx_term[d:] = L[i,:]
    MwLx_term -= xterm
    ml_block = 0.5*np.outer(MwLx_term, xterm)
    ml_sym = ml_block + ml_block.T

    l_sym = np.outer(MwLx_term, MwLx_term)
    mu_sym = np.outer(xterm, xterm)
    K = ml_sym*(1+mu/lipschitz)/alpha - (1/lipschitz)*l_sym/(alpha**2) - mu*mu_sym

    return K

def getReducedContractionFactor(Z, M, ls=None, mus=None, operators=None, alpha=1, gamma=0.5, verbose=False):
    '''
    Get contraction factor for resolvent splitting via PEP

    Args:
        Z (ndarray): Z matrix n x n numpy array
        M (ndarray): M matrix d x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        operators (list): list of proximal operators
        alpha (float): proximal scaling parameter. Default is 1.
        gamma (float): step size. Default is 0.5.
        verbose (bool): verbose output. Default is False.

    Returns:
        tau (float): contraction factor
        
    '''
    d, n = M.shape

    #Ko, Ki, Kmu, Kl = getGramsOld(Z, M, ls=ls, mus=mus, alpha=alpha, gamma=gamma)
    Ko, Ki, Kp = getReducedConstraintMatrices(Z, M, ls=ls, mus=mus, operators=operators, alpha=alpha, gamma=gamma)

    # Build SDP
    G = cvx.Variable((d+n, d+n), PSD=True)

    #constraints = [cvx.trace(Kl[i] @ G) >= 0 for i in range(n)]
    #constraints += [cvx.trace(Kmu[i] @ G) >= 0 for i in range(n)]
    constraints = [cvx.trace(Kpi @ G) >= 0 for Kpi in Kp]
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

def getContractionFactor(Z, W, ls=None, mus=None, operators=None, alpha=1, gamma=0.5, verbose=False, **kwargs):
    """
    Get the contraction factor for the resolvent splitting method
    :math:`v = v - \\gamma W x` for given `Z` and `W` matrices

    Args:
        Z (ndarray): Z matrix n x n numpy array
        W (ndarray): W matrix n x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        operators (list): list of proximal operators
        alpha (float): proximal scaling parameter
        gamma (float): step size
        verbose (bool): verbose output
        kwargs: additional arguments for cvxpy solver

    Returns:
        tau (float): contraction factor
    """
    n = Z.shape[0]
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, ls=ls, mus=mus, operators=operators, alpha=alpha, gamma=gamma)

    G = cvx.Variable((2*n, 2*n), PSD=True)

    constraints = [cvx.trace(Kpi @ G) >= 0 for Kpi in Kp]
    constraints += [cvx.trace(Ki @ G) == 1]
    constraints += [cvx.trace(K1 @ G) == 0]

    objective = cvx.Maximize(cvx.trace(Ko @ G))

    prob = cvx.Problem(objective, constraints)
    prob.solve(verbose=verbose, **kwargs)
    if prob.status != 'optimal':
        print('Problem not solved', prob.status)
        return None
    if verbose:
        print(prob.status)
        print('tau', prob.value)
        print('G', G.value)

        # Duals
        print('Duals')
        for i in range(len(constraints)):
            print(constraints[i].dual_value)

    return prob.value

def getOptimalW(Z, ls=None, mus=None, operators=None, Wref=None, W_fixed={}, alpha=1, verbose=False):
    '''
    Use the dual PEP to get the optimal W

    Args:
        Z (ndarray): Z matrix n x n numpy array
        ls (list, optional): size n numpy array of Lipschitz constants
        mus (list, optional): size n numpy array of strong convexity parameters where mu[i] < l[i]
        operators (list, optional): list of proximal operators
        Wref (ndarray, optional): Base W if only gamma is desired
        W_fixed (dict, optional): Dict of fixed W values
        alpha (float, optional): proximal scaling parameter
        verbose (bool, optional): verbose output

    Returns:
        W (ndarray): optimal consensus matrix
        rho2 (float): minimal contraction factor
    '''
    n = Z.shape[0]
    _, K1, Ki, Ks = getConstraintMatrices(Z, ls=ls, mus=mus, operators=operators, alpha=alpha)
    lambda_s = cvx.Variable(len(Ks), nonneg=True)
    lambda_one = cvx.Variable(1)
    rho2 = cvx.Variable(1)
   
    if Wref is not None:
        gam = cvx.Variable(1)
        Wvar = gam*Wref
    else:
        Wvar = cvx.Variable(Z.shape, symmetric=True)

    # Define the dual problem
    S = sum(lambda_s[i]*Ks[i] for i in range(len(Ks)))
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

def getGamma(Z, W, ls=None, mus=None, operators=None, alpha=1, verbose=False):
    '''
    Use the dual PEP to get the optimal step size gamma
    for the resolvent splitting method
    :math:`v = v - \\gamma W x`

    Args:
        Z (ndarray): Z matrix n x n numpy array
        W (ndarray): W matrix n x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        operators (list): list of proximal operators
        alpha (float): proximal scaling parameter
        verbose (bool): verbose output

    Returns:
        gamma (float): optimal step size
        tau (float): contraction factor
    '''
    Wvar, tau = getOptimalW(Z, ls=ls, mus=mus, operators=operators, Wref=W, alpha=alpha, verbose=verbose)
    if Wvar is None:
        return None, None
    gamma = Wvar[0,0]/W[0,0]
    return gamma, tau

def getReducedGamma(Z, M, ls=None, mus=None, operators=None, alpha=1, verbose=False):
    '''
    Use the dual PEP to get the optimal step size gamma
    for the reduced resolvent splitting method
    :math:`z = z + \\gamma M x`

    Args:
        Z (ndarray): Z matrix n x n numpy array
        M (ndarray): M matrix d x n numpy array
        ls (list): size n numpy array of Lipschitz constants
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]
        operators (list): list of proximal operators
        alpha (float): proximal scaling parameter
        verbose (bool): verbose output

    Returns:
        gamma (float): optimal step size
        tau (float): contraction factor
    '''
    Ko, Ki, Kp = getReducedConstraintMatrices(Z, M, ls=ls, mus=mus, operators=operators, alpha=1)
    
    d = M.shape[0]

    # Define dual variables
    l = cvx.Variable(len(Kp), nonneg=True)
    rho2 = cvx.Variable(1)
    gam = cvx.Variable(1)

    # Define the dual problem
    S = sum(l[i]*Kp[i] for i in range(len(Kp)))
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

def getCoreMatrices(n, gamma, W):
    '''
    Get the matrices for the PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given $W$

    Args:
        n (int): number of operators
        gamma (float): step size
        W (ndarray): (n,n) W matrix

    Returns:
        tuple: tuple of matrices (Ko, Ki, K1)

            Ko (ndarray): 2n x 2n matrix for the objective function
                        :math:`K_o = [I, \\gamma*W; \\gamma*W^T, \\gamma^2 * W^T W]`
            Ki (ndarray): 2n x 2n matrix for the equality constraint
                        :math:`K_i = [I, 0; 0, 0]`
            K1 (ndarray): 2n x 2n matrix for the constraint that v sums to 0
                        :math:`K_1 = [11^T, 0; 0, 0]`
    '''
    ones = np.ones((n,n))
    zero = np.zeros((n,n))
    Ki = np.block([[np.eye(n), zero],[zero, zero]])
    K1 = np.block([[ones, zero],[zero, zero]])
    if W is not None:
        Ko = np.block([[np.eye(n), -gamma*W], [-gamma*W.T, gamma**2 * W.T @ W]])
    else:
        Ko = None
    return Ko, K1, Ki

def getReducedCoreMatrices(gamma, M):
    """
    Get the matrices for the PEP formulation on
    the reduced iteration :math:`z = z + \\gamma M x`

    Args:
        gamma (float): step size
        M (ndarray): d x n matrix

    Returns:
        tuple: tuple of matrices (Ko, Ki)

            Ko (ndarray): (d+n) x (d+n) matrix for the objective function
            Ki (ndarray): (d+n) x (d+n) matrix for the equality constraint
    """
    d, n = M.shape
    eye = np.eye(d)
    zerodn = np.zeros((d,n))
    zerond = np.zeros((n,d))
    zeronn = np.zeros((n,n))
    Ki = np.block([[eye, zerodn], [zerond, zeronn]])
    Ko = np.block([[eye, gamma*M], [gamma*M.T, gamma**2 * M.T @ M]])
    return Ko, Ki

def getConstraintMatrices(Z, W=None, ls=None, mus=None, operators=None, alpha=1, gamma=1, verbose=False):
    '''Get the matrices for the PEP formulation on
    the iteration :math:`v = v - \\gamma W x` for a given :math:`Z, W`
    over a set of `n` `l_i`-Lipschitz, :math:`\\mu_i`-strongly monotone operators
    or the set of provided operators

    Args:
        Z (ndarray): Z matrix n x n numpy array
        W (ndarray): W matrix n x n numpy array
        ls (list): size n numpy array of Lipschitz constants. Default is 2 if no operators are provided.
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]. Default is 1 if no operators are provided.
        operators (list): list of n operators. If None, LipschitzStronglyMonotoneOperator is used with provided ls and mus.
        gamma (float): step size (default is 1 for the dual PEP)
        verbose (bool): verbose output

    Returns:
        tuple: tuple of matrices (Ko, K1, Ki, Kp)

            Ko (ndarray): 2n x 2n matrix for the objective function
                        :math:`K_o = [I, \\gamma*W; \\gamma*W^T, \\gamma^2 * W^T W]`
            K1 (ndarray): 2n x 2n matrix for the constraint that v sums to 0
                        :math:`K_1 = [11^T, 0; 0, 0]`
            Ki (ndarray): 2n x 2n matrix for the equality constraint
                        :math:`K_i = [I, 0; 0, 0]`
            Kp (list): list of interpolation matrices
    '''
    n = Z.shape[0]
    if mus is None:
        mus = np.ones(n)
    if ls is None:
        ls = np.ones(n)*2
    if operators is None:
        operators = [LipschitzStronglyMonotoneOperator(L=ls[i], mu=mus[i], name=str(i)) for i in range(n)]

    # Define the matrices
    Ko, K1, Ki = getCoreMatrices(n, gamma, W)

    # Define the interpolation matrices
    Kp = []
    for i in range(n):
        Kp += operators[i].get_class_matrices(i, Z, alpha=alpha)

    if verbose:
        print(Ko)
        print(Ki)
        print(K1)
        for i in range(n):
            print(Kp[i])

    return Ko, K1, Ki, Kp

def getReducedConstraintMatrices(Z, M, ls=None, mus=None, operators=None, alpha=1, gamma=1, verbose=False):
    """
    Get the matrices for the PEP formulation on
    the reduced iteration :math:`z = z + \\gamma M x` for a given :math:`Z, M`
    over a set of `n` `l_i`-Lipschitz, :math:`\\mu_i`-strongly convex operators
    or the set of provided operators

    Args:
        Z (ndarray): Z matrix n x n numpy array
        M (ndarray): M matrix d x n numpy array
        ls (list): size n numpy array of Lipschitz constants. Default is 2 if no operators are provided.
        mus (list): size n numpy array of strong convexity parameters where mu[i] < l[i]. Default is 1 if no operators are provided.
        operators (list): list of n operators. If None, LipschitzStronglyMonotoneOperator is used with provided ls and mus
        gamma (float): step size (default is 1 for the dual PEP)
        verbose (bool): verbose output

    Returns:
        tuple: tuple of matrices (Ko, Ki, Kp)

            Ko (ndarray): (d+n) x (d+n) matrix for the objective function
            Ki (ndarray): (d+n) x (d+n) matrix for the equality constraint
            Kp (list): list of interpolation matrices
    """
    d, n = M.shape
    if mus is None:
        mus = np.ones(n)
    if ls is None:
        ls = np.ones(n)*2
    if operators is None:
        operators = [LipschitzStronglyMonotoneOperator(L=ls[i], mu=mus[i], name=str(i)) for i in range(n)]

    # Define the matrices
    Ko, Ki = getReducedCoreMatrices(gamma, M)

    # Define the interpolation matrices
    Kp = []
    for i in range(n):
        Kp += operators[i].get_reduced_class_matrices(i, Z, M, alpha=alpha)

    if verbose:
        print(Ko)
        print(Ki)
        for i in range(n):
            print(Kp[i])

    return Ko, Ki, Kp
