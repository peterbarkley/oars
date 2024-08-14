from scipy.sparse import csr_matrix
from oars.utils.sparseProxs import psdCone, traceHalfspaceIndicator, traceEqualityIndicator, linearSubdiff, trace
from oars.algorithms.sparse import serialSparseAlgorithm
from time import time

def testSDP(tgt_n=3, verbose=False):
    print("Testing SDP")
    from oars.matrices import getFull, getBlockMin
    from oars.pep import getConstraintMatrices
    Z, W = getFull(tgt_n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

    # Make sparse
    Ko = csr_matrix(Ko)
    K1 = csr_matrix(K1)
    Ki = csr_matrix(Ki)
    Kp = [csr_matrix(k) for k in Kp]

    proxlist = [psdCone, traceEqualityIndicator, traceEqualityIndicator, linearSubdiff] + [traceHalfspaceIndicator for _ in Kp]
    data = [(2*tgt_n, 2*tgt_n), {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp
    dim = len(data)
    Zd, Wd = getBlockMin(dim, dim//2)
    x, results = serialSparseAlgorithm(dim, data, proxlist, W=Wd, Z=Zd, itrs=2000, checkperiod=10, verbose=verbose) #vartol=1e-5, 
    # print(x)
    # print(results)
    print(trace(Ko, x))
    from oars.pep import getContractionFactor
    print(getContractionFactor(Z, W))

if __name__ == "__main__":
    t = time()
    testSDP(verbose=False)
    print("Serial SDP Time:", time()-t)
    # testSDP(parallel=True, verbose=True)