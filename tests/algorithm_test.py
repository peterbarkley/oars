import numpy as np
from oars import solveMT, solve
from time import time

class absprox():
    '''L1 Norm Resolvent function'''
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    # Evaluates L1 norm
    def __call__(self, x):
        u = x - self.data
        return sum(abs(u))

    # Evaluates L1 norm resolvent
    def prox(self, y, tau=1.0):
        u = y - self.data
        r = max(abs(u)-tau, 0)*np.sign(u) + self.data
        # print(f"Data: {self.data}, y: {y}, u: {u}, r: {r}", flush=True)
        return r

    def __repr__(self):
        return "L1 norm resolvent"
        

class quadprox():
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
    
    def prox(self, y, tau=1.0):
        return (y+tau*self.data)/(1+tau)

def testQuad(parallel=False, verbose=False):
    """
    Test the serial algorithm with a few simple problems
    including a quadratic and L1 norm

    """
    
    # Test quadratic     
    print("Testing quadratic")
    vals = np.array([0, 1, 3, 40]) #np.array([2, -3, -7, -8])
    n = len(vals)
    proxs = [quadprox]*n
    x, results = solveMT(n, vals, proxs, itrs=1000, parallel=parallel, vartol=1e-6, gamma=1.0,  verbose=verbose)
    assert(np.isclose(x,11))

    # Test L1
    print("Testing L1")
    proxs = [absprox]*n
    lx, lresults = solveMT(n, vals, proxs, alpha=0.8, itrs=1000, vartol=1e-6, parallel=parallel, verbose=verbose)
    assert(1 <= lx <= 3)
    # print("lx", lx)
    # print("lresults", lresults)

    # # Test huber resolvent
    # dres = [huber_resolvent]*n
    # dx, dresults = solveMT(n, ldata, dres, itrs=50, vartol=1e-2, verbose=True)
    # print("dx", dx)
    # print("dresults", dresults)



def testSDP(parallel=False):
    print("Testing SDP")
    from oars.matrices import getFull, getBlockMin
    from oars.pep import getConstraintMatrices
    import proxs
    n = 3
    Z, W = getFull(n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

    proxlist = [proxs.psdCone, proxs.traceEqualityIndicator, proxs.traceEqualityIndicator, proxs.linearSubdiff] + [proxs.traceHalfspaceIndicator for _ in Kp]
    data = [(2*n, 2*n), {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp
    dim = len(data)
    Wd, Zd = getBlockMin(dim, dim//2)
    x, results = solve(dim, data, proxlist, W=Wd, Z=Zd, parallel=parallel, itrs=100000, vartol=1e-5, gamma=0.8, checkperiod=10, verbose=False)
    # print(x)
    # print(results)
    print(np.trace(Ko @ x))
    from oars.pep import getContractionFactor
    print(getContractionFactor(Z, W))

if __name__ == "__main__":
    testQuad(parallel=False, verbose=False)
    testQuad(parallel=True, verbose=True)
    t = time()
    testSDP(parallel=False)
    print("Serial SDP Time:", time()-t)
    t = time()
    testSDP(parallel=True)
    print("Parallel SDP Time:", time()-t)