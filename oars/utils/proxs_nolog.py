import numpy as np
from time import time
import warnings
warnings.filterwarnings("error")

class nullProx():
    """
    Class for the null proximal operator
    """

    def __init__(self, dim):
        self.shape = dim

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the null operator
        """
        return X
    
class ntraceEqualityIndicator():
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

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/np.linalg.norm(A, 'fro')**2

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        
        ax = np.trace(self.A @ X)
        if ax == self.v:
            return X
        Y = X - (ax - self.v)*self.U
        return Y

class ntraceHalfspaceIndicator():
    """
    Class for the trace proximal operator
    """

    def __init__(self, A):
        self.A = A
        self.U = self.scale(A)
        self.shape = A.shape

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/np.linalg.norm(A, 'fro')**2

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        ax = np.trace(self.A @ X)
        if ax >= 0:
            return X
        
        return X - ax*self.U
    
class npsdCone():
    """
    Class for the PSD cone proximal operator
    """

    def __init__(self, dim):
        self.shape = dim

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the PSD cone
        """
        try:
            eig, vec = np.linalg.eigh(X)
            eig[eig < 0] = 0
        except:
            print('Error in eigh')
            print(X)

        Y = vec @ np.diag(eig) @ vec.T
        
        return Y


from scipy.sparse.linalg import lobpcg
class npsdConeApprox():
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
        # self.log = []
        self.iteration = 0
        self.is_subspace_positive = False
        self.exact_projections = 0
        self.lobpcg_iterations = 0
        self.max_iter = 10#50000
        self.max_subspace_dim = int(dim[0]/4)
        self.num_eigs = -1
        self.subspace_dim_history = []
        self.buffer_size = int((min(max(self.n / 50, 3), 20)))
        self.zero_tol = 1e-7
        self.check_iter = 500

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
            self.num_eigs = sum(eig < -self.zero_tol)
            self.eigs = eig
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
        
        # log = {}
        # log['start'] = time()
        Y = self.project(X)
        # log['end'] = time()
        # log['time'] = log['end'] - log['start']
        # self.log.append(log)
        if self.iteration % self.check_iter == 0:
            print('Approx PSD', self.iteration, self.exact_projections, self.lobpcg_iterations, self.U.shape[1], self.max_subspace_dim)#, self.eigs)
        return Y

    def project(self, X):
        """
        Compute the proximal operator of the PSD cone
        """
        
        self.iteration += 1
        converged = False
        if (self.U.shape[1] < self.max_subspace_dim) and (self.iteration > 1):
            # print('make it in!')
            try:
                eig, vec = lobpcg(X, self.U, largest=self.is_subspace_positive, tol=self.get_tolerance(), maxiter=self.max_iter)#, retResidualNormsHistory=True)
                self.lobpcg_iterations += 1 #len(hist) # could add retResidualHistory and add len(retResidualHistory) to the iterations
                # test for at least 1 eigenvalue of opposite sign
                self.U = vec
                if not self.is_subspace_positive:
                    eig = -eig
                if sum(eig < self.zero_tol) > 0:
                    converged = True
            except UserWarning:
                # print('Warning in lobpcg, using exact projection')
                # print(X)
                converged = False
            except:
                # print('Error in lobpcg')
                # print(X)
                raise(Exception)
                converged = False
            

        if not converged:
            return self.project_exact(X)



        Y = self.construct_projection(X, eig)
        return Y


class nlinearSubdiff():
    """
    Class for the linear subdifferential
    """

    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.counter = 0

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the linear subdifferential
        """
        self.counter += 1
        Y = X - t*self.A
        if self.counter % 1000 == 0:
            print('Value', self.counter, -np.trace(self.A @ Y))
        return Y
