import numpy as np
from oars.algorithms import distributed_block_solve
from oars.matrices import getFull
from oars.pep import getConstraintMatrices
from oars.utils.proxs_nolog import *
from oars.utils.proxs import *
from time import time
np.set_printoptions(precision=3, suppress=True, linewidth=200)
VERBOSE = False

print("Testing SDP")
t = time()
tgt_n = 30
Z, W = getFull(tgt_n)
Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

# proxlist = [nlinearSubdiff, ntraceEqualityIndicator, ntraceEqualityIndicator, npsdCone] + [ntraceHalfspaceIndicator for _ in Kp]
proxlist = [nlinearSubdiff, ntraceEqualityIndicator, npsdConeApprox, ntraceEqualityIndicator] + [ntraceHalfspaceIndicator for _ in Kp]
# proxlist = [linearSubdiff, traceEqualityIndicator, traceEqualityIndicator, psdCone] + [traceHalfspaceIndicator for _ in Kp]
# proxlist = [linearSubdiff, traceEqualityIndicator, traceEqualityIndicator, psdConeApprox] + [traceHalfspaceIndicator for _ in Kp]
xdim = (2*tgt_n, 2*tgt_n)
data = [-Ko, {'A':Ki, 'v':1}, xdim, {'A':K1, 'v':0}] + Kp
dim = len(data)
warmstart = np.zeros(xdim) #sum(Kp + [Ko, -K1/(2*tgt_n), Ki/(tgt_n-1)])
x, results = distributed_block_solve(dim, data, proxlist, warmstartprimal=warmstart, itrs=50001, gamma=0.99, vartol=1e-5, verbose=True)
# Fiedler 0.157, 0.243 
# Similar 0, 4/dim (0.4)
# SLEM 2/dim, 2/dim (0.2)
# print(x)
# print(results)
# print(warmstart)
print(np.trace(Ko @ x))
print("Time taken", time()-t)
# from oars.pep import getContractionFactor
# print('Mosek')
# t = time()
# print(getContractionFactor(Z, W))
# print("Time taken", time()-t)
# print(np.linalg.eigvalsh(x))