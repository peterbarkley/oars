import numpy as np
from oars import solveMT
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

def testSerial():
    """
    Test the serial algorithm with a few simple problems
    including a quadratic and L1 norm

    """
    
    # Test quadratic     
    vals = np.array([0, 1, 3, 40]) #np.array([2, -3, -7, -8])
    n = len(vals)
    proxs = [quadprox]*n
    x, results = solveMT(n, vals, proxs, itrs=1000, vartol=1e-6, gamma=1.0, checkperiod=1, verbose=False)
    assert(np.isclose(x,11))

    # Test L1
    proxs = [absprox]*n
    lx, lresults = solveMT(n, vals, proxs, alpha=0.8, itrs=1000, vartol=1e-6, parallel=False, verbose=False)
    assert(1 <= lx <= 3)
    # print("lx", lx)
    # print("lresults", lresults)

    # # Test huber resolvent
    # dres = [huber_resolvent]*n
    # dx, dresults = solveMT(n, ldata, dres, itrs=50, vartol=1e-2, verbose=True)
    # print("dx", dx)
    # print("dresults", dresults)

testSerial()