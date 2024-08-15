import numpy as np
from oars.algorithms import distributed_block_solve
from oars.matrices import getFull
from oars.pep import getConstraintMatrices
from oars.utils.proxs import *
from time import time
np.set_printoptions(precision=3, suppress=True, linewidth=200)
VERBOSE = False

print("Testing SDP")
t = time()
tgt_n = 3
Z, W = getFull(tgt_n)
Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

proxlist = [linearSubdiff, traceEqualityIndicator, traceEqualityIndicator, psdCone] + [traceHalfspaceIndicator for _ in Kp]
xdim = (2*tgt_n, 2*tgt_n)
data = [-Ko, {'A':Ki, 'v':1}, {'A':K1, 'v':0}, xdim] + Kp
dim = len(data)
warmstart = np.zeros(xdim) #sum(Kp + [Ko, -K1/(2*tgt_n), Ki/(tgt_n-1)])
x, results = distributed_block_solve(dim, data, proxlist, warmstartprimal=warmstart, itrs=1000, gamma=0.99, vartol=1e-5)
warmdualfirst = []
warmdualsecond = []
for log in results:
    warmdualfirst.append(log[0]['first_v0'])
    warmdualsecond.append(log[0]['second_v0'])
warmdual = warmdualfirst + warmdualsecond
# print(results)
# print(warmstart)
print(np.trace(Ko @ x))
print("Time taken", time()-t)

proxlist = [linearSubdiff, traceEqualityIndicator, traceEqualityIndicator, psdCone] + [traceHalfspaceIndicator for _ in Kp]
x, results = distributed_block_solve(dim, data, proxlist, warmstartprimal=x, warmstartdual=warmdual, itrs=1000, gamma=0.99, vartol=1e-5)

print(np.trace(Ko @ x))
from oars.pep import getContractionFactor
print(getContractionFactor(Z, W))
print(np.linalg.eigvalsh(x))