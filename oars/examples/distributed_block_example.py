from oars.utils.proxs import quadprox
from oars.algorithms import distributed_block_solve
import numpy as np
vals = np.array([0, 1, 3, 40])
n = len(vals)
proxs = [quadprox]*n
x, results = distributed_block_solve(n, vals, proxs, warmstartprimal=np.array(0))
print(x)
print(results)