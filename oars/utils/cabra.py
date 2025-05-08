from time import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

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
        for idx, v in enumerate(self.vars):
            self.y[idx] = y[v][0]
            
        func(self, *args, **kwargs)
        for idx, v in enumerate(self.vars):
            y[v] = self.y[idx].reshape(self.varshapes[idx])

    return wrapper

def _transform_ret(func):

    def wrapper(self, y, *args, **kwargs):
        # Move from dictionary to prox data structure
        # for idx, v in enumerate(self.vars):
        self.y= np.array([y[v] for v in self.vars]).reshape(self.y.shape)
            
        y_out = func(self, *args, **kwargs)
        for idx, v in enumerate(self.vars):
            y[v] = np.array([y_out[idx]])

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
  

class quadprox(baseProx):
    """
    prox of the function f(x) = 0.5 x^T Q x - P x
    """
    def __init__(self, Q, P, D, varshapes, alpha=1.0, varlist=[0], **kwargs):
        self.Q = Q
        self.P = P
        self.D = D
        self.alpha = alpha
        self.aP = alpha*P
        self.shape = P.shape
        self.vars = varlist
        self.varshapes = varshapes
        self.cho = cho_factor(D + alpha*Q)
        self.y = np.zeros(P.shape)
        super().__init__(**kwargs)
    
    @_transform
    def prox_step(self, alpha=1.0, D=None, tol=None):
        if alpha != self.alpha:
            self.alpha = alpha
            self.cho = cho_factor(D + alpha*self.Q)
            self.aP = alpha*self.P
        
        self.y = cho_solve(self.cho, self.y+self.aP)

class quadgrad():
    """
    grad of the function f(x) = 0.5 x^T Q x - P x
    """
    def __init__(self, Q, P, varshapes=[(1,)], varlist=[0]):
        self.Q = Q
        self.P = P
        self.shape = P.shape
        self.vars = varlist
        self.varshapes = varshapes
        self.y = np.zeros(P.shape)
    
    # @_log
    @_transform
    def grad(self):
        
        self.y = self.Q@self.y - self.P

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

def weighted_projection_simplex(v, D, z=1, mid=0):
    """
    Compute x = argmin_{x >= 0, sum x = z} 1/2 * sum_i d_i * (x_i - v_i)^2.
    Inputs:
      v : array of shape (n,)
      d : array of shape (n,), positive weights (diagonal of D)
      z : target sum (default 1)
    Returns:
      x : the D‐weighted projection of v onto the simplex {x>=0, sum x = z}.
    """
    # We need to find theta s.t. sum_i max(v_i - theta/d_i, 0) = z.
    # A simple root‐finding (e.g. bisection) or sorting‐based approach works.
    
    # here’s a sorting‐based scheme:
    # 1. compute "adjusted" values v_i * d_i
    # 2. sort in descending order, accumulate weights, find rho
    # 3. compute theta and then x_i = max(v_i - theta/d_i, 0)
    
    # For brevity, one could do a generic bisection:
    d = np.diag(D)
    def f(theta):
        return np.sum(np.maximum(v - theta / d, 0)) - z

    # bracket theta:
    lo, hi = -1e5, 1e5
    for _ in range(50):
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        mid = 0.5 * (lo + hi)
    theta = 0.5 * (lo + hi)

    x = np.maximum(v - theta / d, 0)
    return x, theta

class warped_simplex(baseProx):
    """
    Project into the simplex defined by x >= 0, 1^T x = 1 (or z, as defined)
    """
    def __init__(self, varshapes, D=None, z=1, varlist=[0], alpha=1.0, **kwargs):

        self.vars = varlist
        self.varshapes = varshapes
        self.di = 1/np.diag(D)
        self.y = np.zeros(self.di.shape)
        self.z = z
        self.theta = 0
        super().__init__(**kwargs)

    
    @_transform_ret
    def prox_step(self, alpha=1.0, D=None, tol=None):
        y, self.theta = weighted_projection_simplex(self.y*self.di, D, self.z, self.theta)
        return y
    
def warped_L1_prox(v, adinv, data=None):


    if data is None:
        return np.sign(v)*np.maximum(np.abs(v) - adinv, 0)
    else:
        return np.sign(v)*np.maximum(np.abs(v-data) - adinv, 0) + data
    
class warped_L1(baseProx):
    """
    warped prox on ||y||_1
    """
    def __init__(self, varshapes, D, data=None, varlist=[0], alpha=1.0, **kwargs):

        self.vars = varlist
        self.varshapes = varshapes
        self.di = 1/np.diag(D)
        self.adinv = alpha/np.diag(D)
        self.y = np.zeros(self.adinv.shape)
        self.data = data
        super().__init__(**kwargs)

    
    @_transform_ret
    def prox_step(self, alpha=1.0, D=None, tol=None):

        y = warped_L1_prox(self.y*self.di, self.adinv, self.data)
        return y
    
class warped_L1_a(baseProx):
    """
    warped prox on ||diag(a)y - b||_1
    """
    def __init__(self, varshapes, D, a, data=None, varlist=[0], alpha=1.0, **kwargs):

        self.vars = varlist
        self.varshapes = varshapes
        self.a = a # same length as diag(D)
        self.di = 1/np.diag(D)
        self.adinv = alpha*self.a**2/np.diag(D)
        self.y = np.zeros(self.adinv.shape)
        self.data = data
        super().__init__(**kwargs)

    
    @_transform_ret
    def prox_step(self, alpha=1.0, D=None, tol=None):
        y = warped_L1_prox(self.y*self.di*self.a, self.adinv, self.data)/self.a
        return y
    
class zeroCone(baseProx):
    """
    Project D^{-1}y into the simplex defined by x >= 0, 1^T x = 1 (or z, as defined)
    """
    def __init__(self, varshapes, D, data=None, varlist=[0], alpha=1.0, **kwargs):

        self.vars = varlist
        self.varshapes = varshapes
        self.data = data
        self.d = np.diag(D)
        self.y = np.zeros(self.d.shape)
        super().__init__(**kwargs)

    
    @_transform_ret
    def prox_step(self, alpha=1.0, D=None, tol=None):
        return np.maximum(self.y/self.d, 0)

class halfspaceProjCabra():
    """
    Warped projection onto the halfspace defined by
    c^T x >= v

    if D^{-1}y is outside the halfspace, it returns
    D^{-1} y - (c^T D^{-1} y - v)* D^{-1} c / c^T D^{-1} c

    othewise it returns D^{-1}y
    """
    def __init__(self, c=-np.ones(2), v=-1, D=np.eye(2), alpha=1.0, varshapes=[(1,), (1,)], varlist=[0,1]):
        self.c = c
        self.Di = np.linalg.inv(D)
        self.Dic = self.Di@c
        self.cDc = c@self.Dic
        self.v = v
        self.vars = varlist
        self.varshapes = varshapes
        self.y = np.zeros(c.shape)
        self.itr = 0

    @_transform_ret
    def prox_step(self, alpha=1.0, D=np.eye(2)):
        Diy = self.Di @ self.y
        t = self.c@Diy - self.v
        if t < 0.0:
            return Diy - t*(self.Dic)/self.cDc
        return Diy


def warped_proj_ball(y, c, r, d, tol=1e-8):
    """
    Compute the warped resolvent J_{D^{-1}F}(y) for the Euclidean ball {||x-c|| <= r},
    under warp D = diag(d).
    """
    w = y - c
    # If already inside the Euclidean ball, no change
    if np.linalg.norm(w) <= r:
        return y

    # Define phi(lambda) = ||(D + λI)^{-1} D w|| - r
    def phi(lmbda):
        u = D_inv = d / (d + lmbda)  # vector of D / (D + λI) applied to w
        return np.linalg.norm(u * w) - r

    # Bisection on [0, L] where L is large enough
    lo, hi = 0.0, 1.0
    while phi(hi) > 0:
        hi *= 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0:
            lo = mid
        else:
            hi = mid
    lambda_star = 0.5 * (lo + hi)
    # print(hi-lo)
    # Compute u* = (D + λ* I)^{-1} D w
    u = (d / (d + lambda_star)) * w
    return c + u


class warped_ball():
    """
    Compute the warped resolvent J_{D^{-1}F}(y) for the Euclidean ball {||x-c|| <= r},
    under warp D = diag(d).
    """
    def __init__(self, c=np.zeros(2), r=1.0, D=np.eye(2), alpha=1.0, varshapes=[(1,), (1,)], varlist=[0,1]):
        self.c = c
        self.r = r
        self.d = np.diag(D)
        self.vars = varlist
        self.varshapes = varshapes
        self.y = np.zeros(c.shape)

    @_transform_ret
    def prox_step(self, alpha=1.0, D=np.eye(2)):
        
        return warped_proj_ball(self.y/self.d, self.c, self.r, self.d)