import numpy as np
from oars.algorithms.helpers import getWarmPrimal, getDualsMean, getDuals
from datetime import datetime

def inertialAlgorithm(n, data, resolvents, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, eta=0.0, verbose=False, callback=None):
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
        eta (float, optional): inertial parameter
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (ndarray): resolvent.shape ndarray of the mean over the node solutions at termination
        results (list): list of dictionaries with the results for each resolvent

    Examples:
        >>> from oars.utils.proxs import quadprox
        >>> from oars.algorithms import inertialAlgorithm
        >>> from oars.matrices import getFull
        >>> import numpy as np
        >>> vals = np.array([0, 1, 3, 40])
        >>> n = len(vals)
        >>> proxs = [quadprox]*n
        >>> Z, W = getFull(n)
        >>> x, results = inertialAlgorithm(n, vals, proxs, W, Z, itrs=1000, gamma=0.85, eta=0.1)

    """
    
    # Initialize the resolvents and variables
    for i in range(n):
        resolvents[i] = resolvents[i](**data[i])
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

    old_v = all_v.copy()
    hat_v = all_v.copy()
    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual at x||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            if i == 0:
                y = hat_v[0]
            else:
                y = hat_v[i] - np.einsum('i,i...->...', Z[i,:i], all_x[:i])
            all_x[i] = resolvents[i].prox(y, alpha)
            
        if callback is not None and callback(itr, all_x, all_v): break

        if verbose and itr % checkperiod == 0:
            xbar = np.mean(all_x, axis=0)
            dualsum = np.linalg.norm(sum(getDuals(all_v, all_x, Z)))
            print(f"{datetime.now()}\t{itr}\t{np.linalg.norm(all_x - xbar):.3e}\t{dualsum:.3e}")

        old_v = all_v.copy()
        all_v = hat_v - np.einsum('nl,l...->n...', gammaW, all_x)
        hat_v = all_v + eta*(all_v-old_v)

        
    x = np.mean(all_x, axis=0)
    
    # Build results list
    results = []
    for i in range(n):
        resultdict = {'x':all_x[i], 'v':all_v[i]}
        if hasattr(resolvents[i], 'log'):
            resultdict['log'] = resolvents[i].log
        results.append(resultdict)

    return x, results

def inertialErrorAlgorithm(n, data, resolvents, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, eta=0.0, sigma=0.0, verbose=False, callback=None):
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
        eta (float, optional): inertial parameter
        sigma (float, optional): error tolerance parameter
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function with signature (itr, x, v)

    Returns:
        x (ndarray): resolvent.shape ndarray of the mean over the node solutions at termination
        results (list): list of dictionaries with the results for each resolvent

    Examples:
        >>> from oars.utils.proxs import quadprox
        >>> from oars.algorithms import inertialErrorAlgorithm
        >>> from oars.matrices import getFull
        >>> import numpy as np
        >>> vals = np.array([0, 1, 3, 40])
        >>> n = len(vals)
        >>> proxs = [quadprox]*n
        >>> Z, W = getFull(n)
        >>> x, results = inertialErrorAlgorithm(n, vals, proxs, W, Z, gamma=0.8, eta=0.05, sigma=0.05)

    """
    
    # Initialize the resolvents and variables
    for i in range(n):
        resolvents[i] = resolvents[i](**data[i])
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
    if sigma > 0.0:
        tol_multiplier = np.linalg.eigvalsh(W)[1] * sigma / 4
        tol = tol_multiplier
    else:
        tol_multiplier = 0.0

    old_v = all_v.copy()
    hat_v = all_v.copy()
    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual at x||\ttol')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            if i == 0:
                y = hat_v[0]
            else:
                y = hat_v[i] - np.einsum('i,i...->...', Z[i,:i], all_x[:i])
            all_x[i] = resolvents[i].prox(y, alpha, tol=tol)
            
        if callback is not None and callback(itr, all_x, all_v): break

        if verbose and itr % checkperiod == 0:
            xbar = np.mean(all_x, axis=0)
            ynorm = np.linalg.norm(sum(getDualsMean(all_v, xbar, Z)))
            dualsum = np.linalg.norm(sum(getDuals(all_v, all_x, Z)))
            print(f"{datetime.now()}\t{itr}\t{np.linalg.norm(all_x - xbar):.3e}\t{dualsum:.3e}\t\t{tol:.3e}")

        old_v = all_v.copy()
        update = np.einsum('nl,l...->n...', gammaW, all_x)
        tol = tol_multiplier*np.linalg.norm(update)/gamma
        all_v = hat_v - update
        hat_v = all_v + eta*(all_v-old_v)

        
    x = np.mean(all_x, axis=0)
    
    # Build results list
    results = []
    for i in range(n):
        resultdict = {'x':all_x[i], 'v':all_v[i]}
        if hasattr(resolvents[i], 'log'):
            resultdict['log'] = resolvents[i].log
        results.append(resultdict)

    return x, results
