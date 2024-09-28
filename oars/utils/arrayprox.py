import numpy as np
from time import time
import warnings
warnings.filterwarnings("error")
from .proxs import baseProx, _log
from scipy.sparse.linalg import lobpcg

class traceArray(baseProx):
    """
    Class for the trace proximal operator
    projects into the space tr(A*X) = v
    where A is a real symmetric matrix and v is a scalar
    and X is a real symmetric matrix
    """

    def __init__(self, data):
        self.a = data['a'] # The row major linearized upper triangle of the coefficient matrix
        self.v = data['v'] # The value to match
        self.n = data['n'] # The size of the matrix
        self.shape = self.a.shape
        self.scale()
        self.build_trace_array()
        super().__init__()

    def scale(self):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        s = np.sum(2*self.a**2) - np.sum(self.a[self.diag_indices()]**2)
        self.U = self.a/s

    def diag_indices(self):
        """
        Get the indices of the diagonal elements
        """
        return np.arange(0, self.n)*(self.n+1) - np.cumsum(np.arange(0, self.n))

    def build_trace_array(self):
        """
        Build the array which can be used to compute the trace
        by doubling the array and resetting the diagonal elements
        """
        self.trace_array = 2*self.a
        self.trace_array[self.diag_indices()] = self.a[self.diag_indices()]

class traceEqualityIndicatorArray(traceArray):
    """
    Class for the trace proximal operator
    projects into the space tr(A*X) = v
    where A is a real symmetric matrix and v is a scalar
    and X is a real symmetric matrix
    """

    def __init__(self, data):
        super().__init__(data)

    @_log
    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        
        ax = np.dot(self.trace_array, X)
        if np.isclose(ax, self.v):
            return X
        Y = X - (ax - self.v)*self.U
        
        return Y

class traceInequalityIndicatorArray(traceArray):
    """
    Class for the trace proximal operator
    projects into the space tr(A*X) >= v
    where A is a real symmetric matrix and v is a scalar
    and X is a real symmetric matrix
    """

    def __init__(self, data):
        super().__init__(data)

    @_log
    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        
        ax = np.dot(self.trace_array, X)
        if ax >= self.v:
            return X
        Y = X - (ax - self.v)*self.U
        
        return Y

class psdConeApproxArray(baseProx):
    """
    Class for the approximate PSD cone projection operator
    using an n(n+1)/2 length array as the data structure

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
        self.n = dim #[0]
        self.shape = self.n*(self.n+1)//2
        self.tolconst = np.sqrt(self.n)*10
        self.X = np.zeros((self.n, self.n))
        self.U = np.zeros((self.n, 0))
        self.iteration = 0
        self.is_subspace_positive = False
        self.exact_projections = 0
        self.lobpcg_iterations = 0
        self.max_iter = 10
        self.max_subspace_dim = int(self.n/4)
        self.num_eigs = -1
        self.subspace_dim_history = []
        self.buffer_size = int((min(max(self.n / 50, 3), 20)))
        self.zero_tol = 1e-9
        self.check_iter = 500
        super().__init__()

    def get_tolerance(self):
        return max(self.tolconst/(self.iteration**1.01), 1e-7)

    def populate(self, X):
        """
        Populate the matrix X from the row major linearized upper triangle
        """
        self.X = np.zeros((self.n, self.n))
        self.X[np.triu_indices(self.n)] = X
        self.X = self.X + self.X.T - np.diag(np.diag(self.X))

    def construct_projection(self, eig):
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
            return self.rowmajor(self.U @ np.diag(np.maximum(eig, 0)) @ self.U.T)
        else:
            return self.rowmajor(self.X + self.U @ np.diag(np.maximum(eig, 0)) @ self.U.T)

    def rowmajor(self, X):
        return X[np.triu_indices(self.n)]

    def project_exact(self):
        """
        Compute the proximal operator of the PSD cone
        """
        self.exact_projections += 1
        try:
            eig, vec = np.linalg.eigh(self.X)
            self.is_subspace_positive = sum(eig > self.zero_tol) < self.n/2
            if self.is_subspace_positive:
                start = max(0, sum(eig <= self.zero_tol) - self.buffer_size)
                stop = self.n
            else:
                start = 0
                stop = min( sum(eig < -self.zero_tol) + self.buffer_size, self.n)
                eig = -eig
            self.U = vec[:, start:stop]
            Y = self.construct_projection(eig[start:stop])
        except:
            print('Error in eigh')
            # print(X)
            raise(Exception)

        return Y

    @_log
    def prox(self, X, t=1):
        self.populate(X)
        return self.project()

    def project(self):
        """
        Compute the proximal operator of the PSD cone
        """
        
        self.iteration += 1
        converged = False
        if self.U.shape[1] < self.max_subspace_dim and self.iteration > 1:
            try:
                eig, vec = lobpcg(self.X, self.U, largest=self.is_subspace_positive, tol=self.get_tolerance(), maxiter=self.max_iter)
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
            return self.project_exact()

        Y = self.construct_projection(eig)
        return Y

