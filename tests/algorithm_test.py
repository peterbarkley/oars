import numpy as np
from oars import solveMT, solve
from time import time
from oars.utils.proxs import *
from oars.utils.proxs_nolog import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
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

def testSDP(tgt_n=3, parallel=False, verbose=False):
    print("Testing SDP")
    from oars.matrices import getFull, getBlockMin, getMT
    from oars.pep import getConstraintMatrices
    Z, W = getMT(tgt_n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

    proxlist = [ntraceEqualityIndicator, npsdConeApprox, ntraceEqualityIndicator, nlinearSubdiff] + [ntraceHalfspaceIndicator for _ in Kp]# + [npsdConeApprox]
    data = [ {'A':Ki, 'v':1}, (2*tgt_n, 2*tgt_n), {'A':K1, 'v':0}, -Ko] + Kp #+ [(2*tgt_n, 2*tgt_n)]
    dim = len(data)
    Zd, Wd = getMT(dim) #, dim//2)
    x, results = solve(dim, data, proxlist, W=Wd, Z=Zd, parallel=parallel, itrs=1000, gamma=0.8, checkperiod=10, verbose=verbose) #vartol=1e-5, 
    # print(x)
    # print(results)
    print(np.trace(Ko @ x))
    print(np.linalg.eigvalsh(x))
    L = -np.tril(Zd, -1)
    print(L)
    for i in range(dim):
        # print(np.linalg.eigvalsh(results[i]['v']))
        print((sum(L[i,:]) - 1))
        u = results[i]['v'] + (sum(L[i,:])-1)*x
        print(np.linalg.eigvalsh(u))
    from oars.pep import getContractionFactor
    print(getContractionFactor(Z, W))

if __name__ == "__main__":
    # testQuad(parallel=False, verbose=False)
    # testQuad(parallel=True, verbose=True)
    t = time()
    testSDP(tgt_n=30, parallel=False, verbose=False)
    print("Serial SDP Time:", time()-t)

    # t = time()
    # testSDP(parallel=True, verbose=True)
    # print("Parallel SDP Time:", time()-t)