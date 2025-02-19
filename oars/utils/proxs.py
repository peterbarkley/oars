import numpy as np
from time import time
import warnings
warnings.filterwarnings("error")


def _log(func):

    def wrapper(self, *args, **kwargs):
        if not self.logging:
            return func(self, *args, **kwargs)
        start = time()
        Y = func(self, *args, **kwargs)
        end = time()
        self.log.append((start, end) + tuple(Y))
        return Y

    return wrapper

class baseProx():
    '''
    Base class for proximal operators
    
    Attributes:
        logging (bool): Whether to log the time taken for each proximal operator call
        log (list): List of tuples containing the start and end times for each proximal operator call

    Methods:
        prox(X, t=1): Compute the proximal operator of the operator

    '''
    def __init__(self):
        self.logging = False
        self.log = []

    @_log
    def prox(self, X, t=None, tol=None):
        return X
    
class nullProx(baseProx):
    """
    Class for the null proximal operator
    """

    def __init__(self, dim):
        self.shape = dim
        super().__init__()

    @_log
    def prox(self, X, t=None, tol=None):
        """
        Compute the proximal operator of the null operator
        """
        return X
    
class traceEqualityIndicator(baseProx):
    """
    Class for the trace proximal operator
    projects into the space tr(A*X) = v
    where A is a real symmetric matrix and v is a scalar
    and X is a real symmetric matrix
    """

    def __init__(self, data):
        self.A = data['A'] # The coefficient matrix
        self.v = data['v'] # The value to match
        self.U = self.scale(self.A)
        self.shape = self.A.shape
        super().__init__()

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/np.linalg.norm(A)**2

    @_log
    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        
        ax = np.sum(self.A * X)
        if np.isclose(ax, self.v):
            return X
        Y = X - (ax - self.v)*self.U
        
        return Y


def solve_2x2(cross, rhs, det):
    x = np.array([(rhs[0] - cross * rhs[1]) / det,
                  (-cross * rhs[0] + rhs[1]) / det])
    return x  
    
class traceInequalityIndicator(baseProx):
    """
    Class for the trace proximal operator
    """

    def __init__(self, data):
        self.A = data['A']
        self.v = data['v']
        self.shape = self.A.shape
        self.scale(self.A)
        super().__init__()

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        self.U = A/np.linalg.norm(A)**2

    @_log
    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        
        ax = np.sum(self.A * X)
        
        if ax >= self.v:
            return X
        
        return X - (ax-self.v)*self.U


class traceInequalityIndicatorDouble(baseProx):
    """
    Class for the trace proximal operator over two constraints


    """

    def __init__(self, data):
        '''
        Args:
            data (dict): containing
                A (list): length 2 containing matrices
                v (list): length 2 containing rhs scalars
        '''
        self.A = data['A']
        self.v = data['v']
        self.shape = self.A[0].shape
        self.scale(self.A, self.v)
        self.cross = np.sum(self.Us[0] * self.Us[1])
        self.det = 1 - self.cross**2
        self.AX = np.zeros(len(self.A))
        self.satisfied = [False, False]
        super().__init__()

    def scale(self, A, v):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        self.A_norms = []
        self.Us = []
        self.rhs = np.zeros(len(self.A))
        for i in range(2):
            t = np.linalg.norm(A[i])
            self.A_norms.append(t**2)
            self.Us.append(A[i]/t)
            self.rhs[i] = v[i]/t


    @_log
    def prox(self, X, t=1, tol=None):
        """
        Compute the proximal operator of the trace norm
        """
        all_sat = True
        for i, Ui in enumerate(self.Us):
            self.AX[i] = np.sum(Ui * X)
        
            if self.AX[i] >= self.rhs[i]:
                self.satisfied[i] = True
            else:
                self.satisfied[i] = False
                all_sat = False
        
        if all_sat:
            return X
        if self.satisfied[1]:
            l0 = self.AX[0] - self.rhs[0]
            if self.cross*l0 <= self.AX[1] - self.rhs[1]:
                return X - l0*self.Us[0]
        elif self.satisfied[0]:
            l1 = self.AX[1] - self.rhs[1]
            if self.cross*l1 <= self.AX[0] - self.rhs[0]:
                return X - l1*self.Us[1]

        lambdas = solve_2x2(self.cross, self.AX - self.rhs, self.det)
        return X - sum(li*Ui for li, Ui in zip(lambdas, self.Us))
  
class traceHalfspaceIndicator(baseProx):
    """
    Class for the trace proximal operator
    """

    def __init__(self, A):
        self.A = A
        self.U = self.scale(A)
        self.shape = A.shape
        super().__init__()

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/np.linalg.norm(A)**2

    @_log
    def prox(self, X, t=None, tol=None):
        """
        Compute the proximal operator of the trace norm
        """
        
        ax = np.sum(self.A * X)
        
        if ax >= 0:
            return X
        
        return X - ax*self.U
    
class psdCone(baseProx):
    """
    Class for the PSD cone proximal operator
    """

    def __init__(self, dim):
        self.shape = dim
        super().__init__()

    @_log
    def prox(self, X, t=None, tol=None):
        """
        Compute the proximal operator of the PSD cone
        """
        try:
            eig, vec = np.linalg.eigh(X)
            eig[eig < 0] = 0
        except:
            print('Error in eigh')

        return vec @ np.diag(eig) @ vec.T


from scipy.sparse.linalg import lobpcg
class psdConeApprox(baseProx):
    """
    Class for the approximate PSD cone projection operator

    https://github.com/nrontsis/ApproximateCOSMO.jl/blob/master/src/lobpcg_projection.jl

    @article{rontsis2022efficient,
    title={Efficient semidefinite programming with approximate admm},
    author={Rontsis, Nikitas and Goulart, Paul and Nakatsukasa, Yuji},
    journal={Journal of Optimization Theory and Applications},
    pages={1--29},
    year={2022},
    publisher={Springer}
    }
    """

    def __init__(self, dim):
        self.shape = dim
        self.n = dim[0]
        self.tolconst = np.sqrt(dim[0])*10
        self.U = np.zeros((dim[0], 0))
        self.iteration = 0
        self.is_subspace_positive = False
        self.exact_projections = 0
        self.lobpcg_iterations = 0
        self.max_iter = 10
        self.max_subspace_dim = int(dim[0]/4)
        self.num_eigs = -1
        self.subspace_dim_history = []
        self.buffer_size = int((min(max(self.n / 50, 3), 20)))
        self.zero_tol = 1e-9
        self.check_iter = 500
        super().__init__()

    def get_tolerance(self):
        return max(self.tolconst/(self.iteration**1.01), 1e-7)

    def construct_projection(self, X, eig):
        """
        Construct the projection matrix

        Args:
            X: The matrix to project
            eig (np.array): The relevant eigenvalues of the matrix
                        (positive or negative depending on the subspace)
                        (if negative, they have been negated)
        """
        self.subspace_dim_history.append(self.U.shape[1])
        if self.is_subspace_positive:
            return self.U @ np.diag(np.maximum(eig, 0)) @ self.U.T
        else:
            return X + self.U @ np.diag(np.maximum(eig, 0)) @ self.U.T

    def project_exact(self, X):
        """
        Compute the proximal operator of the PSD cone
        """
        self.exact_projections += 1
        try:
            eig, vec = np.linalg.eigh(X)
            self.is_subspace_positive = sum(eig > self.zero_tol) < self.n/2
            if self.is_subspace_positive:
                start = max(0, sum(eig < -self.zero_tol) - self.buffer_size)
                stop = self.n
            else:
                start = 0
                stop = min( sum(eig < -self.zero_tol) + self.buffer_size, self.n)
                eig = -eig
            self.U = vec[:, start:stop]
            Y = self.construct_projection(X, eig[start:stop])
        except:
            print('Error in eigh')
            # print(X)
            raise(Exception)

        return Y

    @_log
    def prox(self, X, t=None, tol=None):
        
        Y = self.project(X, tol)
        return Y

    def project(self, X, tol):
        """
        Compute the proximal operator of the PSD cone
        """
        if tol is None:
            tol = self.get_tolerance()
        self.iteration += 1
        converged = False
        if (self.U.shape[1] < self.max_subspace_dim) and (self.iteration > 1):
            try:
                eig, vec = lobpcg(X, self.U, largest=self.is_subspace_positive, tol=tol, maxiter=self.max_iter)
                self.lobpcg_iterations += 1 # could add retResidualHistory and add len(retResidualHistory) to the iterations
                # test for at least 1 eigenvalue of opposite sign
                self.U = vec
                if not self.is_subspace_positive:
                    eig = -eig
                if sum(eig < self.zero_tol) > 0:
                    converged = True
            except UserWarning:
                converged = False
            except:
                raise(Exception)
                converged = False
            
        if not converged:
            return self.project_exact(X)

        Y = self.construct_projection(X, eig)
        return Y

class linearSubdiff(baseProx):
    """
    Class for the linear subdifferential
    """

    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.itr = 0
        super().__init__()

    @_log
    def prox(self, X, t=1):
        """
        Compute the proximal operator of the linear subdifferential
        """
        Y = X - t*self.A
        if self.itr%1000 == 0:
            print(np.sum(Y*self.A), flush=True)
        self.itr += 1

        return Y

class absprox(baseProx):
    '''L1 Norm Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        super().__init__()

    # Evaluates L1 norm
    def __call__(self, x):
        u = x - self.data
        return sum(abs(u))

    # Evaluates L1 norm resolvent
    @_log
    def prox(self, y, tau=1.0, tol=None):
        u = y - self.data
        r = np.maximum(np.abs(u)-tau, 0)*np.sign(u) + self.data
        # print(f"Data: {self.data}, y: {y}, u: {u}, r: {r}", flush=True)
        return r

    def __repr__(self):
        return "L1 norm resolvent"
        
class quadprox(baseProx):
    def __init__(self, data):
        self.data = data
        if hasattr(data, 'shape'):
            self.shape = data.shape
        else:
            self.shape = (1,)
        super().__init__()
    
    @_log
    def prox(self, y, tau=1.0, tol=None):
        return (y+tau*self.data)/(1+tau)

def conjgrad(A, b, x, tol=1e-8):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.
    Credit: https://gist.github.com/glederrey/7fe6e03bbc85c81ed60f3585eea2e073 
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix 
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """  
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    
    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < tol:
            print(i, end=' ')
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

class leastSquareConjprox(baseProx):
    '''
    Solve the problem min_x ||Ax-b||_2^2 + tau/2*||x-y||_2^2
    for a given A, b, and y
    '''
    def __init__(self, A, b, tau=1.0):
        self.A = A
        self.b = b
        self.tau = tau
        if hasattr(A, 'shape'):
            self.n = A.shape[1]
            self.shape = (self.n,)
        else:
            self.shape = (1,)
        self.AAI = A.T@A + tau*np.eye(self.n)
        self.Ab = A.T@b
        self.x = np.zeros(self.n)
        super().__init__()
    
    @_log
    def prox(self, y, tau=1.0, tol=1e-8):
        if tau != self.tau:
            self.tau = tau
            self.AAI = A.T@A + tau*np.eye(self.n)
        self.x = conjgrad(self.AAI, self.Ab + y, self.x, tol=tol)
        return self.x
        
class nullprox():
    def __init__(self, data):
        self.shape = data.shape
        self.logging = True
        self.log = []

    @_log
    def prox(self, y, tau=1.0):
        return y