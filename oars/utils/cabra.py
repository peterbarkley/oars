import numpy as np
from time import time

def _log(func):

    def wrapper(self, *args, **kwargs):
        if not self.logging:
            return func(self, *args, **kwargs)
        start = time()
        func(self, *args, **kwargs)
        end = time()
        self.log.append((start, end))

    return wrapper

def _transform(func):

    def wrapper(self, y, *args, **kwargs):
        # Move from dictionary to prox data structure
        self.y = np.concatenate([y[v] for v in self.vars]).reshape(self.y.shape)
            
        y_out=func(self, *args, **kwargs).flatten()
        i=0
        for idx, v in enumerate(self.vars):
            y[v] = y_out[i:i+len(y[v])] 
            i += len(y[v])

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
    def __init__(self, logging=False):
        self.logging = logging
        self.log = []

    @_log
    def prox_step(self, X, alpha=None, D=None, tol=None):
        return X
    
def expand_inverse_diagonal(D, varshapes):
    # Ensure D is a numpy array (assume it's 2D square)
    D = np.asarray(D)
    assert D.shape[0] == D.shape[1], "D must be a square matrix"
    assert len(varshapes) == D.shape[0], "Length of varshapes must match size of D"

    # Extract the inverse of the diagonal elements
    inv_diag = 1.0 / np.diag(D)

    # Repeat each inverse value according to varshapes[i]
    result = np.concatenate([np.full(shape, inv_diag[i]) for i, shape in enumerate(varshapes)])

    return result

class halfspaceProjCabra():
    """
    Project onto the halfspace defined by
    c^T x >= v
    """
    def __init__(self, c=-np.ones(2), v=-1, shape=(2,), D=np.eye(2), varshapes=[(1,), (1,)], varlist=[0,1]):
        self.c = c
        self.Di = np.diag(expand_inverse_diagonal(D, varshapes))
        self.Dic = self.Di@c
        self.cDc = c@self.Dic
        self.u = self.Dic/np.linalg.norm(c)**2
        self.v = v
        self.vars = varlist
        self.varshapes = varshapes
        self.y = np.zeros(c.shape)
        self.itr = 0

    @_transform
    def prox_step(self, alpha=1.0, D=np.eye(2)):
        Diy = self.Di @ self.y
        t = self.c@Diy - self.v

        if t < 0.0:
            return Diy - t*(self.Dic)/self.cDc
        return Diy


class quadGradCabra():
    """
    grad of the function f(x) = 0.5 x^T Q x - P x
    """
    def __init__(self, Q, P, varshapes=None, varlist=None):
        self.Q = Q
        self.P = P
        self.shape = P.shape
        if varshapes is None:
            self.varshapes = [(1,) for _ in range(len(P))]
        else:
            self.varshapes = varshapes
        if varlist is None:
            self.vars = list(range(len(P)))
        else:
            self.vars = varlist
        self.y = np.zeros(P.shape)
    
    # @_log
    @_transform
    def grad(self):
        
        return self.Q@self.y - self.P

class zeroConeCabra(baseProx):
    """
    Project D^{-1}y into the simplex defined by x >= 0, 1^T x = 1 (or z, as defined)
    """
    def __init__(self, D_diag, varshapes=None, varlist=None, alpha=1.0, **kwargs):

        self.d = D_diag
        if varshapes is None:
            self.varshapes = [(1,) for _ in range(len(self.d))]
        else:
            self.varshapes = varshapes
        if varlist is None:
            self.vars = list(range(len(self.d)))
        else:
            self.vars = varlist
        self.y = np.zeros(self.d.shape)
        super().__init__(**kwargs)

    
    @_transform
    def prox_step(self, alpha=1.0, D=None, tol=None):
        return np.maximum(self.y/self.d, 0)
    

class cbLog:
    def __init__(self, n, m=0, verbose=False):
        self.xdata = [[] for _ in range(n)]  # Dictionary to store x values by iteration
        self.vdata = [[] for _ in range(n)]  # Dictionary to store x values by iteration
        self.bdata = [[] for _ in range(m)]  # Dictionary to store x values by iteration
        self.m = m
        self.verbose = verbose
        
    def __call__(self, itr, x, v, b):
        """
        Store x values for a given iteration.

        Args:
            itr: iteration number
            x: length n
            v: length n
            b: length m
        """
        for varlogi, xi in zip(self.xdata, x):
            varlogi.append(xi.copy())
        for varlogi, vi in zip(self.vdata, v):
            varlogi.append(vi.copy())
        if self.m != 0:
            for varlogi, bi in zip(self.bdata, b):
                varlogi.append(bi.copy())
        if self.verbose:
            print(itr, 'v', v)
            print(itr, 'x', x)
            
            if self.m != 0:print(itr, 'b', b)
