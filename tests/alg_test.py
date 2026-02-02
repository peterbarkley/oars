from oars.algorithms.serial import serialAlgorithm
from oars.algorithms.reducedph import redPhAlgorithm
from oars.utils.proxs import quadProx
from oars.utils.cabra import quadProxCabra
from oars.algorithms.cabra import getPA
from oars.matrices.prebuilt import getFull
from oars.matrices.reducedph import getPH
from oars import solve
import numpy as np

def testSerial():
    vals = [0, 1, 3, 40]
    data = [{'P':np.array([val]), 'Q':np.array([[1]])} for val in vals]
    n = len(vals)
    proxs = [quadProx]*n
    Z, W = getFull(n)
    x, _, _, _ = serialAlgorithm(n, data, proxs, W, Z, itrs=20, gamma=1.0, verbose=False)
    assert np.isclose(x[0],11.0)

testSerial()

def getQP():
    d = 2
    n = 3
    Q = [np.eye(d)]*n
    P = [np.array([1,1]), np.array([2,3]), np.array([3,2])]
    return Q, P

def getTestData(p=None):
    n = 3
    if p is None:
        p = np.ones(n)
    Q, P = getQP()
    return [{'Q': p[i]*Q[i], 'P':p[i]*P[i]} for i in range(n)]

def getTestDataReducedPH():

    n = 3
    Q, P = getQP()
    return [{'Q': Q[i], 'P':P[i], 'varshapes':[2], 'varlist':[0]} for i in range(n)]

def testReduced():
    p = np.array([.1,.2,.7])
    n = 3
    scaled_data = getTestData(p)
    x, _, _, _ = solve(n, scaled_data, [quadProx for _ in range(n)], itrs=20)

    Z = getPH(p)
    varind = [[0]]*n
    data = getTestDataReducedPH()
    PA = getPA(varind, 1)
    A = [quadProxCabra for _ in range(n)]
    x_reduced, _, _, _ = redPhAlgorithm(p, data, A, [Z], [Z], PA, itrs=20)

    assert(np.isclose(x, x_reduced[0]).all())