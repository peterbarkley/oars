import numpy as np
from oars.algorithms import distributed_block_solve
from oars.matrices import getFull
from oars.pep import getConstraintMatrices
from oars.utils.proxs import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
VERBOSE = False

print("Testing SDP")
tgt_n = 3
Z, W = getFull(tgt_n)
Ko, K1, Ki, Kp = getConstraintMatrices(Z, W, gamma=0.5)

proxlist = [psdCone, traceEqualityIndicator, traceEqualityIndicator, linearSubdiff] + [traceHalfspaceIndicator for _ in Kp]
xdim = (2*tgt_n, 2*tgt_n)
data = [xdim, {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp
dim = len(data)
warmstart = np.zeros(xdim) #sum(Kp + [Ko, -K1/(2*tgt_n), Ki/(tgt_n-1)])
x, results = distributed_block_solve(dim, data, proxlist, warmstartprimal=warmstart, itrs=10000, gamma=0.99) #vartol=1e-5,
# Fiedler 0.157, 0.243 
# Similar 0, 4/dim (0.4)
# SLEM 2/dim, 2/dim (0.2)
# print(x)
# print(results)
print(warmstart)
print(np.trace(Ko @ x))
from oars.pep import getContractionFactor
print(getContractionFactor(Z, W))
print(np.linalg.eigvalsh(x))