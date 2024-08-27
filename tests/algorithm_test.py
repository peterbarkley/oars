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


def testSDP(tgt_n=3, parallel=False, verbose=False):
    print("Testing SDP")
    from oars.matrices import getFull, getBlockMin, getMT, getThreeBlockSimilar
    from oars.pep import getConstraintMatrices
    Z, W = getMT(tgt_n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)
    dim = 2*tgt_n + 4
    Zd, Wd = getThreeBlockSimilar(dim) 

    proxlist = [ntraceEqualityIndicator, ntraceEqualityIndicator, nlinearSubdiff] + [ntraceHalfspaceIndicator for _ in Kp]
    data = [ {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp 
    pos = 0
    proxlist.insert(pos, npsdConeApprox)
    data.insert(pos, (2*tgt_n, 2*tgt_n))
    x, results = solve(dim, data, proxlist, W=Wd, Z=Zd, parallel=parallel, itrs=1000, gamma=0.8, checkperiod=10, verbose=verbose) 

    print(np.trace(Ko @ x))

if __name__ == "__main__":
    testQuad(parallel=False, verbose=False)
    testQuad(parallel=True, verbose=True)
    testSDP(tgt_n=20, parallel=False, verbose=False)
    testSDP(parallel=True, verbose=True)