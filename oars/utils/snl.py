import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
from collections import defaultdict

def getData():
    # Define anchor points
    d = 2
    a = np.array([ [0, 1], [4, 2], [2, 0.5]])

    # Define sensor points
    x = np.array([ [1, 0], [2, 2.5], [3, 2], [1.5, 1.5]])

    # Define the squared distances
    da = {(i, k): np.linalg.norm(x[i] - a[k])**2 for i in range(len(x)) for k in range(len(a))}
    dx = {(i, j): np.linalg.norm(x[i] - x[j])**2 for i in range(len(x)) for j in range(len(x))}

    # Get da[i,k] - ||a[k]||^2
    aa = {(i, k): da[i, k] - np.linalg.norm(a[k])**2 for i in range(len(x)) for k in range(len(a))}

    # Define the matrices A_{ik}
    # A = {}
    # for i in range(len(x)):
    #     for k in range(len(a)):
    #         e_i = np.zeros(len(x))
    #         e_i[i] = 1
    #         ae = np.concatenate((a[k], e_i))
    #         A[i, k] = np.outer(ae, ae)

    return a, x, da, dx, aa

# Function to generate random data
def generateRandomData(n, m, d=2, rd=1, nf=0, cutoff=7, seed=0):
    """
    Generate randomize problem data for n points and m anchors inside [0, 1]^d

    Args:
        n (int): number of sensors
        m (int): number of anchors
        d (int): dimension
        rd (float): radius
        nf (float): noise factor
        cutoff (int): max number of neighbors to return
        seed (int): random seed

    Returns:
        a (ndarray): m x d array of anchor point locations
        x (ndarray): n x d array of sensor point locations
        da (dict): dictionary with the squared distances between sensors and anchors 
        where da[i,k] gives the squared distance b/t sensor i and anchor k.
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and sensor j.
        aa (dict): dictionary with the constant aa[i,k] = da[i,k] - ||a[k]||^2
        Ni (dict): dictionary with the neighbors within radius rd of sensor i, truncated to cutoff
        Na (dict): dictionary with the neighbors within radius rd of anchor k
    """
    np.random.seed(seed)
    a = np.random.rand(m, d)
    x = np.random.rand(n, d)

    da = {}
    Na = {}
    for i in range(len(x)):
        Na[i] = []
        for k in range(len(a)):
            dist = np.linalg.norm(x[i] - a[k])
            if dist < rd:
                dval = (dist*(1+nf*np.random.randn()))**2
                da[i, k] = dval
                Na[i].append(k)
    dx = {}
    counters = [0]*len(x)
    Ni = defaultdict(list)
    for i in range(len(x)):
        for j in range(i, len(x)):
            if i == j or counters[j] >= cutoff:
                continue

            dist = np.linalg.norm(x[i] - x[j])
            if dist < rd:
                perturbed_dist = (dist*(1+nf*np.random.randn()))**2
                dx[i, j] = perturbed_dist
                dx[j, i] = perturbed_dist
                counters[i] += 1
                counters[j] += 1
                Ni[i].append(j)
                Ni[j].append(i)
            if counters[i] >= cutoff:
                break
    
    aa = {(i, k): da[i, k] - np.linalg.norm(a[k])**2 for i in range(len(x)) for k in Na[i]}
    return a, x, da, dx, aa, Ni, Na



def getSensorMatrices(a, n, d):
    # Define the matrices A_{ik}
    A = {}
    for i in range(n):
        for k in range(len(a)):
            e_i = np.zeros(n)
            e_i[i] = 1
            ae = np.concatenate((-a[k], e_i))
            A[i, k] = np.outer(ae, ae)

    # Define the matrices L_{ij}
    L = {}
    for i in range(n):
        for j in range(n):
            eij = np.zeros(n+d)
            eij[i+d] = 1
            eij[j+d] = -1
            L[i, j] = np.outer(eij, eij)
    return A, L

# Functino to solve the snl with cvxpy
def solve_snl_cvxpy(a, n, da, dx, Ni, Na, d=2, timelimit=60, verbose=False):
    '''
    Solve the SNL problem using cvxpy

    Args:
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        n (int): number of sensors
        da (dict): dictionary with the squared distances between sensors and anchors 
        where da[i,k] gives the squared distance b/t sensor i and anchor k.
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        d (int): dimension

    Returns:
        x (ndarray): n x d array of sensor locations
    '''

    # Variables
    Z = cp.Variable((n+d, n+d), PSD=True)
    num_pairs = sum([len(Ni[i]) for i in range(n)])//2
    num_ref_pairs = sum([len(Na[i]) for i in range(n)])
    y = cp.Variable(num_pairs)
    b = cp.Parameter(num_pairs)
    z = cp.Variable(num_ref_pairs)
    c = cp.Parameter(num_ref_pairs)

    # Matrix Coefficients
    A, L = getSensorMatrices(a, n, d)
            
    # Objective
    # dx_sum = cp.sum([cp.abs(dx[i,j] - cp.trace(L[i,j]@Z)) for i,j in dx.keys() if i < j])
    # da_sum = cp.sum([cp.abs(da[i,k] - cp.trace(A[i,k]@Z)) for i,k in da.keys()])
    objective = cp.Minimize(cp.norm(y, 1) + cp.norm(z, 1))

    t = cp.vstack([cp.trace(L[i,j]@Z) for i in range(n) for j in Ni[i] if i < j])
    b.value = np.array([dx[i,j] for i in range(n) for j in Ni[i] if i < j])

    s = cp.vstack([cp.trace(A[i,k]@Z) for i in range(n) for k in Na[i]])
    c.value = np.array([da[i,k] for i in range(n) for k in Na[i]]) 

    # Constraints
    constraints = [Z[:d, :d] == np.eye(d), 
                    y == b - t[:,0],
                    z == c - s[:,0]]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose, mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':  timelimit,})

    # Extract the solution
    return Z.value

def solve_subproblem_prox_cvxpy(i, X, a, n, da, dx, Ni, d=2, psd=True, verbose=False):
    '''
    Solve the SNL subproblem using cvxpy

    Args:
        i (int): sensor index
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        n (int): number of sensors
        da (dict): dictionary with the squared distances between sensors and anchors 
        where da[i,k] gives the squared distance b/t sensor i and anchor k.
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        d (int): dimension

    Returns:
        x (ndarray): n x d array of sensor locations
    '''

    # Variables
    if psd:
        Z = cp.Variable((n+d, n+d), PSD=True)
    else:
        Z = cp.Variable((n+d, n+d), symmetric=True)
    # Matrix Coefficients
    A, L = getSensorMatrices(a, n, d)
            
    # Objective
    dx_sum = cp.sum([cp.abs(dx[i,j] - cp.trace(L[i,j]@Z)) for j in Ni])
    da_sum = cp.sum([cp.abs(da[i,k] - cp.trace(A[i,k]@Z)) for k in range(len(a))])
    objective = cp.Minimize(0.5*dx_sum + da_sum + 0.5*cp.norm(X - Z, 'fro')**2)

    # Constraints
    constraints = [Z[:d, :d] == np.eye(d)]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if verbose:
        print(prob.status)
        print('Optimal value: ', prob.value)
    # Extract the solution
    return Z.value

def plotPoints(a, x, xhat=None):
    
    plt.scatter(a[:,0], a[:,1], c='r', label='Anchor points')
    plt.scatter(x[:,0], x[:,1], c='b', label='Sensor points')
    # add labels for the sensor points
    for i in range(len(x)):
        plt.text(x[i,0], x[i,1], str(i))

    if xhat is not None:
        plt.scatter(xhat[:,0], xhat[:,1], c='g', label='Estimated Sensor points')
        # add labels for the estimated sensor points in a different relative position
        for i in range(len(x)):
            plt.text(xhat[i,0], xhat[i,1], str(i), ha='left', va='bottom')
        
    # Add legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def getSTS(d, ni):
    """
    Returns the matrix S^T S
    where S is the scaling matrix for the vectorized submatrix
    0.5||X||_F^2 => 0.5||Sx||^2
    for x = (x_ii, x_ij, x_jj, x_ij', x_j'j' ... x_i1, ..., x_id)
    the scaling for the diagonal elements is 1 and for the off-diagonal elements is 2
    """
    m = 1 + d + 2*len(ni)
    S = np.eye(m)*2
    for i in range(len(ni) + 1):
        S[i*2, i*2] = 1
    return S

class snl_node():
    """
    Prox class for the SNL node objective and top Z block identity constraint
    """
    def __init__(self, data):
        self.d = data['d']
        self.shape = data['shape']
        self.i = data['i']
        self.dx = data['dx']
        self.aa = data['aa']
        self.Ni = data['Ni']
        self.Na = data['Na']
        self.tol = data.get('tol', 1e-3)
        self.maxiter = data.get('maxiter', 100)
        self.iteration = 0
        self.a = data['a']
        self.B = getB(self.a, self.Ni, self.Na, self.d)
        self.rho = data.get('rho', 1)
        self.rhoSiBT = self.rho*np.linalg.inv(getSTS(self.d, self.Ni) + self.rho*self.B.T@self.B)@self.B.T
        self.n = self.B.shape[0]
        self.w = np.zeros(self.n)
        self.x = np.zeros(self.B.shape[1])
        self.y = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.xk = np.zeros(self.B.shape[1])
        self.ii = (self.i + self.d, self.i + self.d)
        self.Y = np.zeros(self.shape)

    def getTol(self):
        return max(1/(self.iteration**1.01), 1e-7)

    def firststep(self):
        """
        L2 prox

        min 0.5||Sx||^2 + rho/2||y + Bx - b + w||^2

        x = -rho*(S^T S + rho B^T B)^-1 B^T(y - b + w)
        
        Args:
            w (array): dual variable
            y (array): primal variable
            b (array): constant
            rhoSiBT (array): precomputed inverse -rho*(S^T S + rho B^T B)^-1 B^T

        Returns:
            array: primal variable x
        """
        return self.rhoSiBT@(-self.y + self.w + self.b)

    def secondstep(self, tau):
        """
        L1 prox

        """
        d = self.b - self.B@self.x + self.w
        y = np.sign(d)*np.maximum(0, np.abs(d) - tau/self.rho)

        return y

    def admm(self, xk, tau):
        # Update b using xk
        for jdx, j in enumerate(self.Ni):
            self.b[jdx] = 0.5*(self.dx[self.i,j] - xk[0] - xk[2*(jdx+1)] + 2*xk[2*jdx + 1])
        for kdx, k in enumerate(self.Na):
            self.b[kdx + len(self.Ni)] = self.aa[self.i,k] - xk[0] + 2*self.a[k]@xk[2*len(self.Ni) + 1:2*len(self.Ni) + 1 + self.d]

        # Main loop
        for itr in range(self.maxiter):
            self.x = self.firststep()
            self.y = self.secondstep(tau)
            delta = self.b - self.y - self.B@self.x
            self.w = self.w + delta
            dd = np.linalg.norm(delta)
            if dd < self.tol:
                # print(f'Converged in {itr} iterations with delta {dd}')
                break

        return self.x

    def prox(self, X, tau=1):
        """
        Proximal operator for the SNL node objective function
        still need to implement tau
        """
        self.iteration += 1
        self.xk[0] = X[self.ii]
        for jdx, j in enumerate(self.Ni):
            self.xk[2*jdx + 1] = X[self.i + self.d, j + self.d]
            self.xk[2*(jdx + 1)] = X[j + self.d, j + self.d]
        self.xk[2*len(self.Ni) + 1:] = X[self.i + self.d, :self.d]
        self.xk = self.admm(self.xk, tau)
        
        self.Y = X.copy()
        # Set the values of X
        self.Y[self.ii] += self.xk[0]
        for jdx, j in enumerate(self.Ni):
            self.Y[self.i + self.d, j + self.d] += self.xk[2*jdx + 1]
            self.Y[j + self.d, j + self.d] += self.xk[2*(jdx + 1)]
            self.Y[j + self.d, self.i + self.d] += self.xk[2*jdx + 1]
            
        self.Y[self.i + self.d, :self.d] += self.xk[2*len(self.Ni) + 1:]
        self.Y[:self.d, self.i + self.d] += self.xk[2*len(self.Ni) + 1:]

        # Update the diagonal elements in the reference block back to 1
        self.Y[:self.d, :self.d] = np.eye(self.d)
        return self.Y

class snl_node_cvx():

    def __init__(self, data):
        self.shape = data['shape']
        self.a = data['a']
        self.i = data['i']
        self.Ni = data['Ni']
        self.Na = data['Na']
        self.d = data['d']
        self.dx = data['dx']
        self.da = data['da'] 
        self.n = data['n']
        self.Z = cp.Variable(self.shape, symmetric=True)
        # Matrix Coefficients
        A, L = getSensorMatrices(self.a, self.n, self.d)
        self.X = cp.Parameter(self.shape, symmetric=True)
        self.X.value = np.zeros(self.shape)
        # Objective
        dx_sum = cp.sum([cp.abs(self.dx[self.i,j] - cp.trace(L[self.i,j]@self.Z)) for j in self.Ni])
        da_sum = cp.sum([cp.abs(self.da[self.i,k] - cp.trace(A[self.i,k]@self.Z)) for k in range(len(self.a))])
        objective = cp.Minimize(0.5*dx_sum + da_sum + 0.5*cp.norm(self.X - self.Z, 'fro')**2)

        # Constraints
        constraints = [self.Z[:self.d, :self.d] == np.eye(self.d)]

        # Solve the problem
        self.prob = cp.Problem(objective, constraints)

    def prox(self, X, tau=1):
        self.X.value = X
        self.prob.solve()
        return self.Z.value

class snl_point():
    def __init__(self, data):
        self.d = data['d']
        self.shape = data['shape']
        self.i = data['i']
        self.j = data['j']
        self.ref_dim = data['ref_dim']
        self.ii = (self.i + self.ref_dim, self.i + self.ref_dim)
        self.jj = (self.j + self.ref_dim, self.j + self.ref_dim)
        self.ij = (self.i + self.ref_dim, self.j + self.ref_dim)
        self.ji = (self.j + self.ref_dim, self.i + self.ref_dim)

    def prox(self, X, tau):
        x_ii, x_ij, x_jj = X[self.ii], X[self.ij], X[self.jj]
        z_ii, z_ij, z_jj = self.threshold(x_ii, x_ij, x_jj, tau)
        X[self.ii], X[self.ij], X[self.ji], X[self.jj] = z_ii, z_ij, z_ij, z_jj
        
        # Update the diagonal elements in the reference block back to 1
        X[:self.ref_dim, :self.ref_dim] = np.eye(self.ref_dim)
        return X

    def threshold(self, xii, xij, xjj, tau):
        '''
        Return the values of :math:`x_{ii}, x_{ij}, x_{jj}` that minimize the objective function
        :math:`|d_{ij} - x_{ii} - x_{jj} + 2x_{ij}||_1 + 0.5||x_{ii}||_2^2 + ||x_{ij}||_2^2 + 0.5||x_{jj}||_2^2` 
        '''
        d = self.d - xii - xjj + 2*xij
        v = np.sign(d)*min(1, np.abs(d)/(4*tau))
        z_ii = xii + tau*v
        z_jj = xjj + tau*v
        z_ij = xij - tau*v
        return z_ii, z_ij, z_jj

class snl_ref():
    def __init__(self, data):
        self.d = data['d'] # da[i,k] - ||a[k]||^2
        self.shape = data['shape']
        self.ak = data['ak']
        self.scale = 2*self.ak@self.ak + 1
        self.ref_dim = len(self.ak)
        self.i = data['i']
        self.ii = (self.i + self.ref_dim, self.i + self.ref_dim)

    def prox(self, X, tau):
        y = X[self.ref_dim + self.i, :self.ref_dim]
        xii = X[self.ii]
        z_ii, z = self.threshold(xii, y, tau)
        X[self.ref_dim + self.i, :self.ref_dim] = z
        X[:self.ref_dim, self.i + self.ref_dim] = z
        X[self.ii] = z_ii
        
        # Update the diagonal elements in the reference block back to 1
        X[:self.ref_dim, :self.ref_dim] = np.eye(self.ref_dim)
        return X

    def threshold(self, xii, y, tau):
        '''
        Return the values of :math:`x_{i1}, x_{i2}` that minimize the objective function
        :math:`||d_{ij} + 2*ak_1*x_{i1} + 2*ak_2*x_{i2}||_1 + ||x_{i1}||_2^2 + ||x_{i2}||_2^2` 
        '''
        d = self.d - xii + 2*self.ak@y
        lam = -np.sign(d)*min(1, np.abs(d)/(self.scale*tau))
        z_ii = xii - tau*lam
        z = y + tau*lam*self.ak
        return z_ii, z

class snl_node_psd():
    """
    Prox class for the SNL node PSD constraint
    """
    def __init__(self, data):
        self.shape = data['shape']
        self.i = data['i']
        self.Ni = data['Ni']
        self.ref_dim = data['ref_dim']
        self.Y = np.zeros((self.ref_dim + len(self.Ni) + 1, self.ref_dim + len(self.Ni) + 1))
        self.ii = (self.i + self.ref_dim, self.i + self.ref_dim)
        self.indices = list(range(self.ref_dim)) + [self.i+self.ref_dim] + [j + self.ref_dim for j in self.Ni]

    def prox(self, X, tau):
        self.Y = X[self.indices][:, self.indices]
        # X = np.zeros(self.shape)
        # PSD projection
        eig, eigv = np.linalg.eig(self.Y)
        eig[eig < 0] = 0
        self.Y = eigv @ np.diag(eig) @ eigv.T
        for i, idx in enumerate(self.indices):
            for j, jdx in enumerate(self.indices):
                X[idx, jdx] = self.Y[i, j]

        return X

def snl_point_prox(d):
    '''
    Return the values of :math:`x_{jj}` and :math:`x_{ij}` that minimize the objective function
    :math:`|d_{ij} - x_{jj} + 2x_{ij}||_1 + ||x_{ij}||_2^2 + 0.5||x_{jj}||_2^2` 
    '''
    v = np.sign(d)*min(1, np.abs(d)/3)
    return v, -v

def snl_ref_truncated_prox(d, ak):
    '''
    Return the values of :math:`x_{i1}` and :math:`x_{i2}` that minimize the objective function
    :math:`||b_{ik} + 2*ak_1*x_{i1} + 2*ak_2*x_{i2}||_1 + ||x_{i1}||_2^2 + ||x_{i2}||_2^2` 
    '''
    scale = 2*ak@ak
    lam = -np.sign(d)*min(1, np.abs(d)/scale)
    return lam*ak

def get_ref_abs_value(b, ak, x, z):
    return np.abs(b - x + 2*ak@z) + 0.5*np.linalg.norm(x)**2 + np.linalg.norm(z)**2

def get_point_abs_value(d, xii, xjj, xij, tau=1):
    return tau*np.abs(d - xii - xjj + 2*xij) + 0.5*np.linalg.norm(xii)**2 + 0.5*np.linalg.norm(xjj)**2 + np.linalg.norm(xij)**2

def cvx_ref_prox(aa, ak, i, X, y, d=2, tau=1, verbose=False):
    """
    Return the values of :math:`x^k_{ii} + x_{ii}, x^k_{i1} + x_{i1}, x^k_{i2} + x_{i2}` that minimize the objective function
    :math:`||b_{ik} - x_{ii} + 2*ak_1*x_{i1} + 2*ak_2*x_{i2}||_1 + 0.5||x_{ii}||^2 + ||x_{i1}||_2^2 + ||x_{i2}||_2^2` 
    where :math:`b_{ik} = aa - x_{ii}^k + 2*ak_1*x_{i1}^k + 2*ak_2*x_{i2}^k`
    and :math:`aa = d_{ik}^2 - ||a_k||^2
    """
    # Variables
    xii = cp.Variable()
    z = cp.Variable(2)

    # Objective
    b = aa - X[i, i] + 2*ak@y
    dx_sum = tau*cp.abs(b - xii + 2*ak@z)
    proxterm = 0.5*cp.norm(xii)**2 + cp.norm(z)**2
    objective = cp.Minimize(dx_sum + proxterm)

    # Solve the problem
    prob = cp.Problem(objective)
    prob.solve()

    if verbose:
        print(prob.status)
        print('Optimal value: ', prob.value)
        print('xii: ', xii.value)
        print('z: ', z.value)
        print('absval', dx_sum.value)

    # Extract the solution
    X[i, i] = xii.value + X[i, i]

    return X, z.value + y

def cvx_point_prox(d, X, i, j, tau=1, verbose=False):
    """
    Return the values of :math:`x_{ii}, x_{ij}, x_{jj}` that minimize the objective function
    :math:`|d_{ij} - x_{ii} - x_{jj} + 2x_{ij}||_1 + 0.5||x_{ii}||_2^2 + ||x_{ij}||_2^2 + 0.5||x_{jj}||_2^2` 
    """
    # Variables
    xii = cp.Variable()
    xij = cp.Variable()
    xjj = cp.Variable()

    # Objective
    dx = d - X[i, i] - X[j, j] + 2*X[i, j]
    dx_sum = tau*cp.abs(dx - xii - xjj + 2*xij)
    proxterm = 0.5*cp.norm(xii)**2 + cp.norm(xij)**2 + 0.5*cp.norm(xjj)**2
    objective = cp.Minimize(dx_sum + proxterm)

    # Solve the problem
    prob = cp.Problem(objective)
    prob.solve()

    if verbose:
        print(prob.status)
        print('Optimal value: ', prob.value)
        print('xii: ', xii.value)
        print('xij: ', xij.value)
        print('xjj: ', xjj.value)
        print('absval', dx_sum.value)

    # Extract the solution
    X[i, i] = xii.value + X[i, i]
    X[i, j] = xij.value + X[i, j]
    X[j, i] = xij.value + X[j, i]
    X[j, j] = xjj.value + X[j, j]

    return X

def getB(a, Ni, Na=None, d=2):
    if Na is None:
        Na = list(range(len(a)))
    m = len(Na) + len(Ni)

    A = np.zeros((m, 3 + 2*len(Ni)))
    A[:len(Ni),0] = 0.5*np.ones(len(Ni))
    A[len(Ni):,0] = np.ones(len(Na))
    for i in range(len(Ni)):
        A[i,2*i+1:2*i+3] = 0.5*np.array([-2, 1])
    for i in range(len(Ni), len(Ni) + len(Na)):
        k = Na[i - len(Ni)]
        # print(k, a[k, :], A[i,2*len(Ni)+1:2*len(Ni)+1+d])
        A[i,2*len(Ni)+1:2*len(Ni)+1+d] = -2*a[k,:]

    return A

# Test for snl_ref prox against cvxpy
def test_snl_ref():
    a, x, da, dx, aa = snl.getData()
    n = len(x)
    d = 2
    
    for tau in [0.5, 1, 2]:
        for yref in [1, 0]:            
            for xref in [np.zeros((n,n)), np.eye(n)]:
                for i in range(n):
                    for k in range(len(a)):
                        data = {'d': aa[i,k], 'shape': (n+d, n+d), 'ak': a[k], 'i': i}
                        snl_ref_obj = snl_ref(data)
                        A = np.block([[np.eye(d), yref*np.ones((d, n))], [yref*np.ones((n, d)), xref]])
                        X = A.copy()
                        Xk = xref
                        y = yref*np.ones(d)
                        Y, z = cvx_ref_prox(aa[i, k], a[k], i, Xk, y, tau=tau)
                        A[i+d, :d] = z
                        A[:d, i+d] = z
                        A[d:, d:] = Y
                        Z = snl_ref_obj.prox(X, tau)
                        assert np.allclose(A, Z, atol=1e-2)

def test_snl_point():
    a, x, da, dx, aa = snl.getData()
    n = len(x)
    d = 2
    
    for tau in [0.5, 1, 2]:
        for xref in [np.zeros((n,n)), np.eye(n)]:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # print(tau, i, j, dx[i,j])
                        data = {'d': dx[i,j], 'shape': (n+d, n+d), 'i': i, 'j': j, 'ref_dim': d}
                        snl_point_obj = snl_point(data)
                        A = np.block([[np.eye(d), np.zeros((d, n))], [np.zeros((n, d)), xref]])
                        X = A.copy()
                        Xk = xref.copy()
                        Y = cvx_point_prox(dx[i,j], Xk, i, j, tau=tau, verbose=False)
                        A[d:, d:] = Y
                        Z = snl_point_obj.prox(X, tau)
                        # print(snl.get_point_abs_value(dx[i,j], Z[d+i, d+i], Z[d+j, d+j],  Z[d+i, d+j], tau))
                        assert np.allclose(A, Z, atol=1e-2)
