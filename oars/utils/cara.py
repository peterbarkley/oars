import numpy as np

from scipy.linalg import cho_factor, cho_solve


class quadProx():
    """
    warped prox of the function f(x) = 0.5 x^T Q x - P x
    """
    def __init__(self, Q, P, D, indices, varshapes, varlist, alpha=1.0):
        self.Q = Q # (n,n) array
        self.P = P # (n,) array
        self.D = D # (p,) array
        self.alpha = alpha
        self.aP = alpha*P
        self.shape = P.shape
        self.vars = varlist # length p list
        self.varshapes = varshapes # length p list where sum is n
        self.Dmat = np.diag([D[k] for k in range(len(D)) for _ in range(varshapes[k]) ])
        self.cho = cho_factor(self.Dmat + alpha*Q)
        self.indices = indices
        
    def prox(self, y, alpha=1.0, tol=None):
        if alpha != self.alpha:
            self.alpha = alpha
            self.cho = cho_factor(self.Dmat + alpha*self.Q)
            self.aP = alpha*self.P
        
        return cho_solve(self.cho, y+self.aP)