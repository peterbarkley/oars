import numpy as np
import sys
from oars.algorithms import distributed_block_solve
from oars.utils.proxs import quadprox

VERBOSE = False

if __name__ == '__main__':
    if '-v' in sys.argv:
        VERBOSE = True
        sys.argv.remove('-v')
    print("Testing quadratic")
    vals = np.array([0, 1, 3, 40]) #np.array([2, -3, -7, -8])
    n = len(vals)
    proxs = [quadprox]*n
    warm = np.array(0)
    x, results = distributed_block_solve(n, vals, proxs, warmstartprimal=warm, itrs=10, gamma=1.0,  verbose=VERBOSE)
    for r in results:
        print(r)
    print(x)
    assert(np.isclose(x,11))