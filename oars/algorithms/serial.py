import numpy as np
from oars.algorithms.helpers import getWarmPrimal, getDualsMean
from time import time
from datetime import datetime


def serialAlgorithm(n, data, resolvents, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, verbose=False, callback=None):
    """
    Run the frugal resolvent splitting algorithm defined by Z and W in serial

    Args:
        n (int): the number of resolvents
        data (list): list containing the problem data for each resolvent
        resolvents (list): list of :math:`n` resolvent classes
        W (ndarray): size (n, n) ndarray for the :math:`W` matrix
        Z (ndarray): size (n, n) ndarray for the :math:`Z` matrix
        warmstartprimal (ndarray, optional): resolvent.shape ndarray for :math:`x` in v^0
        warmstartdual (ndarray, optional): n x resolvent.shape ndarray for :math:`u` which sums to 0 in v^0
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (ndarray): resolvent.shape ndarray of the mean over the node solutions at termination
        results (list): list of dictionaries with the results for each resolvent

    Examples:
        >>> from oars.utils.proxs import quadprox
        >>> from oars.algorithms import serialAlgorithm
        >>> from oars.matrices import getFull
        >>> import numpy as np
        >>> vals = np.array([0, 1, 3, 40])
        >>> n = len(vals)
        >>> proxs = [quadprox]*n
        >>> Z, W = getFull(n)
        >>> x, results = serialAlgorithm(n, vals, proxs, W, Z, itrs=1000, vartol=1e-6, gamma=1.0)
        Converged in objective value, iteration 13
        >>> x
        10.999999999990674
        >>> results
        [{'x': 10.999999999906539, 'v': 22.00000000003744}, {'x': 11.000000000103075, 'v': 13.66666666663539}, {'x': 10.999999999962412, 'v': 4.333333333327117}, {'x': 10.999999999990674, 'v': -40.0}]

    """
    
    # Initialize the resolvents and variables
    for i in range(n):
        resolvents[i] = resolvents[i](data[i])
        if i == 0:
            m = resolvents[0].shape
            all_x = np.zeros((n,) + m)

    if warmstartprimal is None:
        all_v = np.zeros((n,) + m)
    else:
        all_v = getWarmPrimal(warmstartprimal, Z)
    if warmstartdual is not None:
        all_v += warmstartdual
    gammaW = gamma*W

    # Run the algorithm
    if verbose: 
        print('date time itr ||x-bar(x)|| ||dual|| ||v||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            if i == 0:
                y = all_v[0]
            else:
                y = all_v[i] - np.einsum('i,i...->...', Z[i,:i], all_x[:i])
            all_x[i] = resolvents[i].prox(y, alpha)
            
        if callback is not None and callback(itr, all_x, all_v): break

        if verbose and itr % checkperiod == 0:
            xbar = np.mean(all_x, axis=0)
            ynorm = np.linalg.norm(getDualsMean(all_v, xbar, Z))
            print(datetime.now(), itr, np.linalg.norm(all_x - xbar), ynorm, np.linalg.norm(all_v))

        all_v -= np.einsum('nl,l...->n...', gammaW, all_x)

        
    x = np.mean(all_x, axis=0)
    
    # Build results list
    results = []
    for i in range(n):
        resultdict = {'x':all_x[i], 'v':all_v[i]}
        if hasattr(resolvents[i], 'log'):
            resultdict['log'] = resolvents[i].log
        results.append(resultdict)

    return x, results
