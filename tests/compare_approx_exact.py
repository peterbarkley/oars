import numpy as np
from oars import solveMT, solve
from time import time
from oars.utils.proxs import *
from oars.utils.proxs_nolog import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
import warnings
warnings.filterwarnings("error")

def testSDP(tgt_n=3, parallel=False, verbose=False):
    print("Testing SDP")
    from oars.matrices import getFull, getBlockMin, getMT, getThreeBlockSimilar
    from oars.pep import getConstraintMatrices
    Z, W = getMT(tgt_n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)
    dim = 2*tgt_n + 4
    Zd, Wd = getThreeBlockSimilar(dim) #getBlockMin(dim, [dim//4, dim//2, dim-(dim//4 + dim//2)]) #, dim//2)
    L = -np.tril(Zd, -1)

    pos = 0
    itrs = 1000
    print("Position:", pos)
    proxlist = [ntraceEqualityIndicator, ntraceEqualityIndicator, nlinearSubdiff] + [ntraceHalfspaceIndicator for _ in Kp]# + [npsdConeApprox]
    data = [ {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp #+ [(2*tgt_n, 2*tgt_n)]
    
    proxlist.insert(pos, npsdConeApprox)
    data.insert(pos, (2*tgt_n, 2*tgt_n))
    # dim = len(data)
    t = time()
    x, results = solve(dim, data, proxlist, W=Wd, Z=Zd, parallel=parallel, itrs=itrs, gamma=0.8, checkperiod=10, verbose=verbose) #vartol=1e-5, 
    print("Approx Time taken", time()-t)
    proxlist = [ntraceEqualityIndicator, ntraceEqualityIndicator, nlinearSubdiff] + [ntraceHalfspaceIndicator for _ in Kp]# + [npsdConeApprox]
    data = [ {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp #+ [(2*tgt_n, 2*tgt_n)]
    
    proxlist.insert(pos, npsdCone)
    data.insert(pos, (2*tgt_n, 2*tgt_n))
    t = time()
    x, results = solve(dim, data, proxlist, W=Wd, Z=Zd, parallel=parallel, itrs=itrs, gamma=0.8, checkperiod=10, verbose=verbose) #vartol=1e-5, 
    print("Exact Time taken", time()-t)
    # print(x)
    # print(results)
    # print(np.trace(Ko @ x))
    # print(L)
    # for i in range(dim):
        # print(np.linalg.eigvalsh(results[i]['v']))
        # print((sum(L[i,:]) - 1))
    # v = results[pos]['v']
    # u = v + (sum(L[pos,:])-1)*x
    # input_vec = v + sum(L[pos,j]*results[j]['x'] for j in range(pos))
    # print('in', np.linalg.eigvalsh(input_vec))
    # print('v', np.linalg.eigvalsh(v))
    # print('u', np.linalg.eigvalsh(u))
    # from oars.pep import getContractionFactor
    # print(getContractionFactor(Z, W))

if __name__ == "__main__":
    # testQuad(parallel=False, verbose=False)
    # testQuad(parallel=True, verbose=True)
    t = time()
    testSDP(tgt_n=50, parallel=False, verbose=False)
    print("Serial SDP Time:", time()-t)

    # t = time()
    # testSDP(parallel=True, verbose=True)
    # print("Parallel SDP Time:", time()-t)