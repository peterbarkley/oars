# -*- coding: utf-8 -*-
from oars.pep import getContractionFactor
from oars.matrices import getFull, getMT
from time import time
import cvxpy as cvx
import mosek

Z, W = getMT(30)
t = time()
print(getContractionFactor(Z, W, verbose=False, solver=cvx.MOSEK))
print(time() - t)