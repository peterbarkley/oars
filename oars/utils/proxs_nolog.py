import numpy as np
from time import time

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

class nlinearSubdiff():
    """
    Class for the linear subdifferential
    """

    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the linear subdifferential
        """
        Y = X - t*self.A
        return Y
