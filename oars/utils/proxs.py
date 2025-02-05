import numpy as np
from time import time

class traceEqualityIndicator():
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
        self.log = []

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/np.linalg.norm(A, 'fro')**2

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        
        log = {}
        log['start'] = time()
        ax = np.trace(self.A @ X)
        if ax == self.v:
            log['end'] = time()
            log['time'] = log['end'] - log['start']
            self.log.append(log)
            return X
        Y = X - (ax - self.v)*self.U
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        self.log.append(log)
        return Y

class traceHalfspaceIndicator():
    """
    Class for the trace proximal operator
    """

    def __init__(self, A):
        self.A = A
        self.U = self.scale(A)
        self.shape = A.shape
        self.log = []

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/np.linalg.norm(A, 'fro')**2

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        log = {}
        log['start'] = time()
        ax = np.trace(self.A @ X)
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        self.log.append(log)
        if ax >= 0:
            return X
        
        return X - ax*self.U
    
class psdCone():
    """
    Class for the PSD cone proximal operator
    """

    def __init__(self, dim):
        self.shape = dim
        self.log = []

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the PSD cone
        """
        log = {}
        log['start'] = time()
        try:
            eig, vec = np.linalg.eigh(X)
            eig[eig < 0] = 0
        except:
            print('Error in eigh')
            print(X)

        Y = vec @ np.diag(eig) @ vec.T
        
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        self.log.append(log)
        return Y

from scipy.sparse.linalg import lobpcg
class psdConeApprox():
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
        self.log = []
        self.iteration = 0
        self.is_subspace_positive = False
        self.exact_projections = 0
        self.lobpcg_iterations = 0
        self.max_iter = 1000
        self.max_subspace_dim = int(dim[0]/4)
        self.subspace_dim_history = []
        self.buffer_size = int((min(max(self.n / 50, 3), 20)))
        self.zero_tol = 1e-9

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
            print(X)
            raise(Exception)

        return Y

    def prox(self, X, t=1):
        
        log = {}
        log['start'] = time()
        Y = self.project(X)
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        self.log.append(log)
        return Y

    def project(self, X):
        """
        Compute the proximal operator of the PSD cone
        """
        
        self.iteration += 1
        converged = False
        if self.U.shape[1] < self.max_subspace_dim and self.iteration > 1:
            try:
                eig, vec = lobpcg(X, self.U, largest=self.is_subspace_positive, tol=self.get_tolerance(), maxiter=self.max_iter)
                self.lobpcg_iterations += 1 # could add retResidualHistory and add len(retResidualHistory) to the iterations
                # test for at least 1 eigenvalue of opposite sign
                if not self.is_subspace_positive:
                    eig = -eig
                if sum(eig < self.zero_tol) > 0:
                    converged = True
            except:
                print('Error in lobpcg')
                print(X)
                raise(Exception)
                converged = False
            

        if not converged:
            return self.project_exact(X)



        Y = self.construct_projection(X, eig)
        return Y

class linearSubdiff():
    """
    Class for the linear subdifferential
    """

    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.log = []

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the linear subdifferential
        """
        log = {}
        log['start'] = time()
        Y = X - t*self.A
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        self.log.append(log)

        return Y

class absprox():
    '''L1 Norm Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    # Evaluates L1 norm
    def __call__(self, x):
        u = x - self.data
        return sum(abs(u))

    # Evaluates L1 norm resolvent
    def prox(self, y, tau=1.0):
        u = y - self.data
        r = max(abs(u)-tau, 0)*np.sign(u) + self.data
        # print(f"Data: {self.data}, y: {y}, u: {u}, r: {r}", flush=True)
        return r

    def __repr__(self):
        return "L1 norm resolvent"
        
class quadprox():
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
    
    def prox(self, y, tau=1.0):
        return (y+tau*self.data)/(1+tau)

class nullprox():
    def __init(self, data):
        self.shape = data.shape

    def prox(self, y, tau=1.0):
        return y