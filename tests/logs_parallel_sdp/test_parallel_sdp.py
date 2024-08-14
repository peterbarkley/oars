import numpy as np
from oars import solveMT, solve
from time import time
from oars.utils.proxs import *
import json

def testSDP(tgt_n=3, parallel=False, vartol=1e-4, verbose=False):
    print("Testing SDP")
    from oars.matrices import getFull, getTwoBlockSLEM
    from oars.pep import getConstraintMatrices
    Z, W = getFull(tgt_n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

    proxlist = [psdCone, traceEqualityIndicator, traceEqualityIndicator, linearSubdiff] + [traceHalfspaceIndicator for _ in Kp]
    data = [(2*tgt_n, 2*tgt_n), {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp
    dim = len(data)
    Zd, Wd = getTwoBlockSLEM(dim)
    x, results = solve(dim, data, proxlist, W=Wd, Z=Zd, parallel=parallel, itrs=500, gamma=0.8, checkperiod=10, vartol=vartol, verbose=verbose) #vartol=1e-5, 
    # print(x)
    # print(results)
    # Save results
    for i, log in enumerate(results):
        with open('parallel_logs_'+str(i)+'.json', 'w') as f:
            json.dump(log, f)
    print(np.trace(Ko @ x))
    from oars.pep import getContractionFactor
    print(getContractionFactor(Z, W))

if __name__ == "__main__":

    t = time()
    testSDP(tgt_n=16, parallel=True)
    print("Parallel SDP Time:", time()-t)
