from oars.algorithms import *
from oars.utils.proxs import quadprox
from oars.matrices import getFull
import numpy as np

def testSerial():
    vals = [0, 1, 3, 40]
    data = [{'data':val} for val in vals]
    n = len(vals)
    proxs = [quadprox]*n
    Z, W = getFull(n)
    x, results = serialAlgorithm(n, data, proxs, W, Z, itrs=20, gamma=1.0, verbose=False)
    assert np.isclose(x[0],11.0)

testSerial()