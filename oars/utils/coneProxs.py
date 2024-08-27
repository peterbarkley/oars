from scipy.linalg import lapack
from numpy import eye
from time import time

class psdConeAlt():
    """
    Class for the PSD cone proximal operator
    """

    def __init__(self, dim):
        self.shape = dim
        self.log = []
        self.eye = eye(dim[0])
        


    def prox(self, X, t=1):
        """
        Compute the proximal operator of the PSD cone
        """
        log = {}
        log['start'] = time()
        smallest_eig = lapack.dsyevr(X, compute_v=0, range='I', il=1, iu=1, abstol=1e-6)[0][0]
        if smallest_eig >= 0:
            log['end'] = time()
            log['time'] = log['end'] - log['start']
            self.log.append(log)
            return X
        Y = X - smallest_eig*self.eye
        log['end'] = time()
        log['time'] = log['end'] - log['start']
        self.log.append(log)
        return Y