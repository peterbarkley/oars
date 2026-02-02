import numpy as np

from scipy.linalg import cho_factor, cho_solve


class quadProx():
    """
    warped prox of the function f(x) = 0.5 x^T Q x - P x
    """
    def __init__(self, Q, P, varshapes, varlist, indices=None, D=None, alpha=1.0):
        self.Q = Q # (n,n) array
        self.P = P # (n,) array
        self.alpha = alpha
        self.aP = alpha*P
        self.shape = P.shape
        self.vars = varlist # length p list
        self.varshapes = varshapes # length p list where sum is n
        if D is None:
            self.Dmat = np.eye(len(P)) # (p,) array
        else:
            self.Dmat = np.diag([D[k] for k in range(len(D)) for _ in range(varshapes[k]) ])
        self.cho = cho_factor(self.Dmat + alpha*Q)
        self.indices = indices
        
    def prox(self, y, alpha=1.0, tol=None):
        if alpha != self.alpha:
            self.alpha = alpha
            self.cho = cho_factor(self.Dmat + alpha*self.Q)
            self.aP = alpha*self.P
        
        return cho_solve(self.cho, y+self.aP)
    

class warpedBoxProj():
    '''Prox for box constraint indicator plus linear and diagonal quadratic at y
    returns x = argmin_z \\alpha*(i_0(z) + c^T z + 0.5 * z^T Q z) - z^T y + 0.5 z^T D z)
    where i_0 is the indicator function on the non-negative cone`
    so 0 = c - v + (Q+D)x + \\lambda(x) where \\lambda_i < 0 only if x_i = 0, and \\lambda_i = 0 if x_i > 0, and x >= 0
    the solution is x_i = 0 if (v_i - \\alpha*c_i)/(\\alpha*q_i + d_i) <= 0, else x_i = (v_i - \\alpha*c_i)/(\\alpha*q_i + d_i)
    '''
    def __init__(self, varlist, q=0., varshapes=None, indices=None, c=0., D=None, lower=0., upper=np.inf, **kwargs):
        '''
        Args:
            varlist (list): list of the variable indices
            q (ndarray): 1 dimensional array diagonal of quadratic term (optional, default zeros)
            varshapes (list): list of the lengths of the vectorized variables (optional, default ones)
            indices (list): list of the indices of the variables in varlist in the prox variable array.
            c (ndarray): array of the linear term (optional, default zeros)
            D (ndarray): 1 dimensional array diagonal for warping projection (optional, default ones)
            upper (ndarray): array of upper bounds (optional, default np.inf)
        '''
        if varshapes is None:
            shape = len(varlist)
        else:
            shape = sum(varshapes)
        if D is None:
            self.d = np.ones(shape)
        else:
            self.d = D 
        self.q = q
        self.c = c
        self.lower = lower
        self.upper = upper
        self.alpha = np.inf
        self.vars = varlist
        self.indices = indices

    def prox(self, y, alpha=1.0, D=None):
        if alpha != self.alpha:
            self.alphac = alpha*self.c
            self.daq = self.d + alpha*self.q

        return np.clip((y - self.alphac)/(self.daq), self.lower, self.upper, out=y)
