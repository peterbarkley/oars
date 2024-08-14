from scipy.sparse.linalg import norm #, eigsh
from scipy.sparse import csr_array
from numpy import diag
from numpy.linalg import eigh

def trace(A, X):
    """
    Compute the trace of A*X
    """
    return (A.multiply(X)).sum()

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

    def scale(self, A):
        """
        Scale the matrix A by the squared Frobenius norm
        """    
        return A/(norm(A, 'fro')**2)

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        ax = trace(self.A, X)
        if ax == self.v:
            return X

        return X - (ax - self.v)*self.U

class traceHalfspaceIndicator():
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
        return A/(norm(A, 'fro')**2)

    def prox(self, X, t=1):
        """
        Compute the proximal operator of the trace norm
        """
        ax = trace(self.A, X)
        if ax >= 0:
            return X
        
        return X - ax*self.U

class psdCone():
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
            # print('X', X)
            eig, vec = eigh(X.toarray())
            eig[eig < 0] = 0
        except:
            print('Error in eigh')
            print(X)

        # Use scipy sparse matrix multiplication
        #eig_sparse = csr_array(diag(eig))
        return csr_array(vec @ diag(eig) @ vec.T)

class linearSubdiff():
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
        return X - t*self.A

