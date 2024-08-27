from oars.pep import *
# from oars.pep.pep import operator_splitting_W
from oars.matrices import getMT, getIncidence
from oars.pepit import wc_frugal_resolvent_splitting, wc_reduced_frugal_resolvent_splitting
import numpy as np

# Verify correct matrices for lipschitz strongly monotone
def test_matrices_lipstrong():
    print("Testing matrices for lipschitz strongly monotone")
    n = 3
    Z, W = getMT(n)
    eye = np.eye(n)
    ones = np.ones((n,n))
    zeros = np.zeros((n,n))
    Ko, K1, Ki, Kp = getConstraintMatrices(Z, W)
    Ko_ref = np.block([[eye, -W], [-W, W@W]])
    assert(np.allclose(Ko, Ko_ref))
    K1_ref = np.block([[ones, zeros], [zeros, zeros]])
    assert(np.allclose(K1, K1_ref))
    Ki_ref = np.block([[eye, zeros], [zeros, zeros]])
    assert(np.allclose(Ki, Ki_ref))
    Kp_ref = [np.array([[ 0. ,  0. ,  0. ,  0.5,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0.5,  0. ,  0. , -2. , -0. , -0. ],
       [ 0. ,  0. ,  0. , -0. , -0. , -0. ],
       [ 0. ,  0. ,  0. , -0. , -0. , -0. ]]), 
       np.array([[-1., -0., -0.,  1.,  0.,  0.],
       [-0., -0., -0.,  0.,  0.,  0.],
       [-0., -0., -0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  3.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.]]), 
       np.array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.5,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0.5,  0. ],
       [ 0. ,  0.5,  0. ,  0.5, -2. , -0. ],
       [ 0. ,  0. ,  0. ,  0. , -0. , -0. ]]), 
       np.array([[-0., -0., -0., -0.,  0.,  0.],
       [-0., -1., -0., -1.,  1.,  0.],
       [-0., -0., -0., -0.,  0.,  0.],
       [-0., -1., -0., -1.,  1.,  0.],
       [ 0.,  1.,  0.,  1.,  3.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.]]), 
       np.array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.5],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.5],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.5],
       [ 0. ,  0. ,  0.5,  0.5,  0.5, -2. ]]), 
       np.array([[-0., -0., -0., -0., -0.,  0.],
       [-0., -0., -0., -0., -0.,  0.],
       [-0., -0., -1., -1., -1.,  1.],
       [-0., -0., -1., -1., -1.,  1.],
       [-0., -0., -1., -1., -1.,  1.],
       [ 0.,  0.,  1.,  1.,  1.,  3.]])]
    for i in range(2*n):
        assert(np.allclose(Kp[i], Kp_ref[i]))

test_matrices_lipstrong()

# Verify tau results with pepit for lipschitz strongly monotone
def test_tau_lipstrong(n=4, alpha=1, gamma=0.5):
    print("Testing tau for lipschitz strongly monotone")
    from PEPit import operators
    Z, W = getMT(n)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    
    tau = getContractionFactor(Z, W, ls=ls, mus=mus, alpha=alpha, gamma=gamma)
    pepit_tau = wc_frugal_resolvent_splitting(L, W, ls, mus, operator=operators.LipschitzStronglyMonotoneOperator, alpha=alpha, gamma=gamma, verbose=-1)
    # print(tau, pepit_tau)
    assert(np.allclose(tau, pepit_tau))

test_tau_lipstrong(alpha=2, gamma=0.5)

# Verify tau results with pepit for smooth strongly convex
def test_tau_smoothstrong(n=4, alpha=1, gamma=0.5):
    print("Testing tau for smooth strongly convex")
    Z, W = getMT(n)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    operators = [SmoothStronglyConvexSubdifferential(L=ls[i], mu=mus[i]) for i in range(n)]
    tau = getContractionFactor(Z, W, None, None, operators=operators, alpha=alpha, gamma=gamma)
    pepit_tau = wc_frugal_resolvent_splitting(L, W, ls, mus, alpha=alpha, gamma=gamma, verbose=-1)
    # print(tau, pepit_tau)
    assert(np.allclose(tau, pepit_tau))

test_tau_smoothstrong(alpha=1, gamma=0.5)

# Verify correct matrices for reduced lipschitz strongly monotone
def test_matrices_redlipstrong(gamma=0.5):
    print("Testing reduced matrices for lipschitz strongly monotone")
    n = 3
    Z, W = getMT(n)
    M = getIncidence(W)
    d = M.shape[0]
    Kon, Kin, Kpn = getReducedConstraintMatrices(Z, M)
    eye = np.eye(d)
    Ko_ref = np.block([[eye, M], [M.T, M.T@M]])
    assert(np.allclose(Ko_ref, Kon))
    Ki_ref = np.zeros((d+n, d+n))
    Ki_ref[:d, :d] = eye
    assert(np.allclose(Ki_ref, Kin))
    Kpn_ref = [np.array([[ 0. ,  0. ,  0.5,  0. ,  0. ],
                    [ 0. ,  0. , -0. , -0. , -0. ],
                    [ 0.5, -0. , -2. , -0. , -0. ],
                    [ 0. , -0. , -0. , -0. , -0. ],
                    [ 0. , -0. , -0. , -0. , -0. ]]),
                np.array([[-1.,  0.,  1.,  0.,  0.],
                    [ 0., -0., -0., -0., -0.],
                    [ 1., -0.,  3.,  0.,  0.],
                    [ 0., -0.,  0.,  0.,  0.],
                    [ 0., -0.,  0.,  0.,  0.]]),
            
            np.array([[ 0. ,  0. , -0. , -0.5, -0. ],
                    [ 0. ,  0. ,  0. ,  0.5,  0. ],
                    [-0. ,  0. ,  0. ,  0.5,  0. ],
                    [-0.5,  0.5,  0.5, -2. , -0. ],
                    [-0. ,  0. ,  0. , -0. , -0. ]]),
            np.array([[-1.,  1.,  1., -1., -0.],
                    [ 1., -1., -1.,  1.,  0.],
                    [ 1., -1., -1.,  1.,  0.],
                    [-1.,  1.,  1.,  3.,  0.],
                    [-0.,  0.,  0.,  0.,  0.]]),
            np.array([[ 0. ,  0. , -0. , -0. , -0. ],
                    [ 0. ,  0. , -0. , -0. , -0.5],
                    [-0. , -0. ,  0. ,  0. ,  0.5],
                    [-0. , -0. ,  0. ,  0. ,  0.5],
                    [-0. , -0.5,  0.5,  0.5, -2. ]]),
            np.array([[-0., -0.,  0.,  0., -0.],
                    [-0., -1.,  1.,  1., -1.],
                    [ 0.,  1., -1., -1.,  1.],
                    [ 0.,  1., -1., -1.,  1.],
                    [-0., -1.,  1.,  1.,  3.]]),]
    for i in range(2*n):
        assert(np.allclose(Kpn[i], Kpn_ref[i]))

test_matrices_redlipstrong(gamma=0.5)

# Verify tau results with pepit for reduced lipschitz strongly monotone
def test_tau_redlipstrong(n=4, alpha=1, gamma=0.5):
    from PEPit.operators import LipschitzStronglyMonotoneOperator
    print("Testing tau for reduced lipschitz strongly monotone")
    Z, W = getMT(n)
    M = getIncidence(W)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    tau = getReducedContractionFactor(Z, M, ls=ls, mus=mus, alpha=alpha, gamma=gamma)
    pepit_tau = wc_reduced_frugal_resolvent_splitting(L, M, ls, mus, LipschitzStronglyMonotoneOperator, alpha=alpha, gamma=gamma, verbose=-1)
    # print(tau, pepit_tau)
    assert(np.allclose(tau, pepit_tau))

test_tau_redlipstrong(alpha=1, gamma=0.5)

# Verify tau results with pepit for reduced smooth strongly convex
def test_tau_redsmoothstrong(n=4, alpha=1, gamma=0.5):
    print("Testing tau for reduced smooth strongly convex")
    Z, W = getMT(n)
    M = getIncidence(W)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    operators = [SmoothStronglyConvexSubdifferential(L=ls[i], mu=mus[i]) for i in range(n)]
    tau = getReducedContractionFactor(Z, M, None, None, operators=operators, alpha=alpha, gamma=gamma)
    pepit_tau = wc_reduced_frugal_resolvent_splitting(L, M, ls, mus, alpha=alpha, gamma=gamma, verbose=-1)
    # print(tau, pepit_tau)
    assert(np.allclose(tau, pepit_tau))

test_tau_redsmoothstrong(alpha=1, gamma=0.5)

# Verify tau resuts with pepit for mixed case

# Verify dual results with pepit for lipschitz strongly monotone
def test_dual_lipstrong(n, alpha=1):
    print("Testing dual results for lipschitz strongly monotone")
    from PEPit import operators
    Z, W = getMT(n)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    tau = getContractionFactor(Z, W, ls=ls, mus=mus, alpha=alpha, gamma=0.5)
    gam, tau_gam = getGamma(Z, W, ls=ls, mus=mus, alpha=alpha)
    Wo, tau_opt = getOptimalW(Z, ls=ls, mus=mus, alpha=alpha)
    pepit_tau_gam = wc_frugal_resolvent_splitting(L, W, ls, mus, operators.LipschitzStronglyMonotoneOperator, alpha=alpha, gamma=gam, verbose=-1)
    petit_tau_W = wc_frugal_resolvent_splitting(L, Wo, ls, mus, operators.LipschitzStronglyMonotoneOperator, alpha=alpha, gamma=1, verbose=-1)
    # print(tau, tau_gam, pepit_tau_gam, tau_opt, petit_tau_W)
    assert(np.isclose(tau_gam, pepit_tau_gam))
    assert(np.isclose(tau_opt, petit_tau_W))
    assert(tau_opt <= tau_gam)
    assert(tau_gam <= tau)

test_dual_lipstrong(4, alpha=1)

# Verify dual results with pepit for smooth strongly convex
def test_dual_smoothstrong(n, alpha=1):
    print("Testing dual results for smooth strongly convex")
    Z, W = getMT(n)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    operators = [SmoothStronglyConvexSubdifferential(L=ls[i], mu=mus[i]) for i in range(n)]
    tau = getContractionFactor(Z, W, None, None, operators=operators, alpha=alpha, gamma=0.5)
    Wo, tau_opt = getOptimalW(Z, operators=operators, alpha=alpha)
    pepit_tau = wc_frugal_resolvent_splitting(L, Wo, ls, mus, alpha=alpha, gamma=1, verbose=-1)
    # print(tau, pepit_tau, tau_opt)
    assert(np.isclose(tau_opt, pepit_tau, atol=1e-3))
    assert(tau_opt <= tau)

test_dual_smoothstrong(4, alpha=1)

# Verify dual results with pepit for reduced lipschitz strongly monotone
def test_dual_redlipstrong():
    print("Testing dual results for reduced lipschitz strongly monotone")
    from PEPit import operators
    n = 4
    Z, W = getMT(n)
    M = getIncidence(W)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    tau = getReducedContractionFactor(Z, M, ls=ls, mus=mus, alpha=1, gamma=0.5)
    gam, tau_opt = getReducedGamma(Z, M, ls=ls, mus=mus)
    pepit_tau = wc_reduced_frugal_resolvent_splitting(L, M, ls, mus, operators.LipschitzStronglyMonotoneOperator, alpha=1, gamma=gam, verbose=-1)
    # print(tau, pepit_tau, tau_opt, gam)
    assert(np.allclose(tau_opt, pepit_tau))
    assert(tau_opt <= tau)

test_dual_redlipstrong()

# Verify dual results with pepit for reduced smooth strongly convex
def test_dual_redsmoothstrong(n, alpha=1):
    print("Testing dual results for reduced smooth strongly convex")
    Z, W = getMT(n)
    M = getIncidence(W)
    L = - np.tril(Z, -1)
    ls = np.ones(n)*2
    ls = np.cumsum(ls)
    mus = np.ones(n)
    mus = np.cumsum(mus)
    ops = [SmoothStronglyConvexSubdifferential(L=ls[i], mu=mus[i]) for i in range(n)]
    tau = getReducedContractionFactor(Z, M, operators=ops, alpha=alpha, gamma=0.5)
    gam, tau_opt = getReducedGamma(Z, M, operators=ops, alpha=alpha)
    pepit_tau = wc_reduced_frugal_resolvent_splitting(L, M, ls, mus, alpha=1, gamma=gam, verbose=-1)
    # print(tau, pepit_tau, tau_opt, gam)
    assert(np.allclose(tau_opt, pepit_tau))
    assert(tau_opt <= tau)

test_dual_redsmoothstrong(4, alpha=1)