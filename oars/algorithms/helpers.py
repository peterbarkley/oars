import numpy as np
from collections import deque

class ConvergenceChecker():
    def __init__(self, vartol=None, objtol=None, earlyterm=None, detectcycle=0, counter=1, objective=None, data=None, x=None, f=None):
        self.vartol = vartol
        self.objtol = objtol
        self.earlyterm = earlyterm
        self.cycle = detectcycle
        self.objective = objective
        self.count = counter
        self.varcounter = counter
        self.objcounter = counter
        self.hashq = deque(maxlen=detectcycle)
        self.checkobj = self.objtol is not None and self.objective is not None
        if self.checkobj:
            self.last = None
            self.data = data[-1]
            if f is not None:
                self.last = f
            elif x is not None:
                self.last = self.objective(sum(x)/len(x)) #self.data, 

    def check(self, x, xbar=None, verbose=False):
        if self.vartol is not None:
            if xbar is None:
                xbar = sum(x)/len(x)
            deviation = x - xbar
            if np.linalg.norm(deviation) < self.vartol:
                self.varcounter -= 1
                if self.varcounter == 0:
                    return True
            else:
                self.varcounter = self.count
        if self.checkobj:
            if xbar is None:
                xbar = sum(x)/len(x)
            f = self.objective(xbar) #self.data, 
            if verbose:
                print("Objective value on mean", f)
            if abs(f - self.last) < self.objtol:
                self.objcounter -= 1
                if self.objcounter == 0:
                    return True
            else:
                self.objcounter = self.count
                self.last = f
        if self.earlyterm is not None:
            if xbar is None:
                xbar = sum(x)/len(x)
            xdev = sum(abs(x[i] - xbar) for i in range(len(x)))
            changed = np.sum(xdev != 0)
            if verbose:print("Vars with disagreement", changed)
            if changed < self.earlyterm:
                return True
        if self.cycle > 0:
            # xh = tuple(hash(xi.tobytes()) for xi in x)
            # if xh in self.hashq:
            #     return True
            # self.hashq.append(xh)
            for i in range(len(self.hashq)):
                if np.allclose(x, self.hashq[i]):
                    print("Cycling detected!")
                    return True
                self.hashq.append(x.copy())
        return False


# Splitting Algorithm Execution Functions
def getWarmPrimal(x, Z):
    '''
    Return an n x m ndarray of the primal warmstart input to v^0

    Args:
        x (ndarray): m ndarray of the estimated solution
        Z (ndarray): n x n ndarray of algorithm weights

    Returns:
        (ndarray): n x m ndarray of 1^T(I-L) otimes x
    '''
    IL = getIL(Z)
    return np.array([wt*x for wt in IL])
    # P = np.sum(np.tril(Z, -1), axis=1) + 1
    # return [i*x for i in P]

def getWarmDual(d):
    '''Returns the initial point v^0 in H^n
    
    Args:
        d (dict): dictionary with keys 'M' and 'u'
            M (ndarray): (n-1) by n ndarray for algorithm :math:`w^{k+1} = w^k + \\gamma M^T x^k`
            u (ndarray): (n-1) by x.shape ndarray of dual values

    Returns:
        -u@M (ndarray): n x x.shape ndarray of initial values for :math:`v^0`
    '''
    M = d['M']
    u = d['u']
    return -u@M

def getDuals(v, x, Z):
    """
    Get the dual values for the splitting algorithm
    using the formula :math:`u = v + (L-I)x`
    where :math:`L` is the lower triangular part of the -Z matrix

    Args:
        v (list): list of iterate values
        x (list): list of primal values
        Z (ndarray): Z matrix

    Returns:
        list: list of dual values
    """
    n = len(v)
    return np.array([v[i] - sum(Z[i,j]*x[j] for j in range(i)) - x[i] for i in range(n)])

def getDualsMean(v, xbar, Z):
    """
    Get the dual values for the splitting algorithm
    using the formula :math:`u = v + (L-I)x`
    where :math:`L` is the lower triangular part of the -Z matrix

    Args:
        v (list): list of iterate values
        xbar (list): primal value
        Z (ndarray): Z matrix

    Returns:
        list: list of dual values
    """
    n = len(v)
    IL = getIL(Z)
    return np.array([v[i] - IL[i]*xbar for i in range(n)])

def getIL(Z):
    return np.sum(np.tril(Z, -1), axis=1) + 1

def getGammaLimit(eta):
    a = 2*(eta-1)**2
    return 0.5*a/(a + 3*eta-1)

def cb(itr, all_x, all_v, xtol=1e-6, dualtol=1e-6):
    xbar = np.mean(all_x, axis=0)
    xdev = np.linalg.norm(all_x - xbar)
    dualsum = np.linalg.norm(sum(getDuals(all_v, all_x, Z)))
    if xdev <= xtol and dualsum <= dualtol:
        print("Converged at iteration", itr)
        return True

    return False