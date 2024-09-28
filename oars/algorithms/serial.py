import numpy as np
from oars.algorithms.helpers import ConvergenceChecker, getWarmPrimal, getDuals
from time import time
from datetime import datetime

def serialAlgorithm(n, data, resolvents, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None, objtol=None, objective=None, checkperiod=10, verbose=False, debug=False):
    """
    Run the frugal resolvent splitting algorithm defined by Z and W in serial

    Args:
        n (int): the number of resolvents
        data (list): list containing the problem data for each resolvent
        resolvents (list): list of :math:`n` resolvent classes
        W (ndarray): size (n, n) ndarray for the :math:`W` matrix
        Z (ndarray): size (n, n) ndarray for the :math:`Z` matrix
        warmstartprimal (ndarray, optional): resolvent.shape ndarray for :math:`x` in v^0
        warmstartdual (list, optional): is a list of n ndarrays for :math:`u` which sums to 0 in v^0
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        vartol (float, optional): is the variable tolerance
        objtol (float, optional): is the objective tolerance
        objective (function, optional): the objective function
        checkperiod (int, optional): the period to check for convergence
        verbose (bool, optional): True for verbose output

    Returns:
        x (ndarray): the solution
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
    all_x = []
    for i in range(n):
        resolvents[i] = resolvents[i](data[i])
        if i == 0:
            m = resolvents[0].shape
        x = np.zeros(m)
        all_x.append(x)
    if warmstartprimal is not None:
        all_v = getWarmPrimal(warmstartprimal, Z)
        if debug:print('warmstartprimal', all_v)
    else:
        all_v = [np.zeros(m) for _ in range(n)]
    if warmstartdual is not None:
        all_v = [all_v[i] + warmstartdual[i] for i in range(n)]
        if debug:print('warmstart final', all_v)

    # Run the algorithm
    if verbose:
        print('Starting Serial Algorithm')
        diffs = [ 0 ]*n
        start_time = time()
    convergence = ConvergenceChecker(vartol, objtol, counter=n, objective=objective, data=data, x=all_x) 
    itr_period = itrs//10
    counter = checkperiod
    xresults = []
    vresults = []
    wx = [np.zeros(m) for _ in range(n)]
    for itr in range(itrs):
        for i in range(n):
            resolvent = resolvents[i]
            y = all_v[i] - sum(Z[i,j]*all_x[j] for j in range(i))
            x = resolvent.prox(y, alpha)
            if debug: 
                diffs[i] = np.linalg.norm(x - all_x[i])
                print("B/t iteration difference norm for", i, ":", diffs[i])
            all_x[i] = x #resolvent.prox(y, alpha)

        for i in range(n):     
            wx[i] = sum(W[i,j]*all_x[j] for j in range(n))       
            all_v[i] = all_v[i] - gamma*wx[i]
        if verbose and itr % itr_period == 0:
            timedelta = (time()-start_time)
            delta = gamma*np.linalg.norm(wx)
            xbar = sum(all_x)/n
            sum_diff = sum(np.linalg.norm(all_x[i] - xbar) for i in range(n))
            u = getDuals(all_v, all_x, Z)
            sum_zero_diff = np.linalg.norm(sum(u))
            print(datetime.now(), 'Iteration', itr, 'time', timedelta, 'Delta v', delta, 'Sum diff', sum_diff, 'Sum zero diff', sum_zero_diff)
            if debug:
                for i in range(n):    
                    print("Difference across x", i, i-1, ":", np.linalg.norm(all_x[i]-all_x[i-1]))
                for i in range(n):
                    print('x', i, all_x[i])
                    print('v', i, all_v[i])
                xresults.append(all_x.copy())
                vresults.append(all_v.copy())
        if convergence.check(all_x, verbose=verbose):
            print('Converged in value, iteration', itr+1)
            break
        
    if verbose:
        print('Serial Algorithm Loop Time:', time()-start_time)
    x = sum(all_x)/n
    
    # Build results list
    results = []
    for i in range(n):
        if hasattr(resolvents[i], 'log'):
            results.append({'x':all_x[i], 'v':all_v[i], 'log':resolvents[i].log})
        else:
            results.append({'x':all_x[i], 'v':all_v[i]})
    if verbose:
        print('results')
        for i in range(n):
            print(f'x {i}:', all_x[i])
            print(f'v {i}:', all_v[i])
        results.append({'xresults':xresults, 'vresults':vresults})
    return x, results


