import numpy as np
from oars.algorithms.helpers import getWarmPrimal, getDualsMean, getDuals
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
        warmstartdual (ndarray, optional): n x resolvent.shape ndarray for dual values :math:`u` which sums to 0 in v^0
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (ndarray): resolvent.shape ndarray of the mean over the node solutions at termination
        logs (list): list of n logs for the operators
        all_x (ndarray): n x resolvent.shape ndarray of the node solution
        all_v (ndarray): n x resolvent.shape ndarray of the consensus iterates at solution

    Examples:
        >>> from oars.utils.proxs import quadprox
        >>> from oars.algorithms import serialAlgorithm
        >>> from oars.matrices import getFull
        >>> import numpy as np
        >>> vals = [0, 1, 3, 40]
        >>> data = [{'data':val} for val in vals]
        >>> n = len(vals)
        >>> proxs = [quadprox]*n
        >>> Z, W = getFull(n)
        >>> x, results = serialAlgorithm(n, data, proxs, W, Z, itrs=20, gamma=1.0, verbose=True)
        >>> x
        [11.]
        >>> results
        [{'x': array([11.]), 'v': array([22.]), 'log': []}, {'x': array([11.]), 'v': array([13.66666667]), 'log': []}, {'x': array([11.]), 'v': array([4.33333333]), 'log': []}, {'x': array([11.]), 'v': array([-40.]), 'log': []}]

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

    all_y = np.zeros((n,) + m)
    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual at x||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            if i == 0:
                all_y[i] = all_v[0]
            else:
                all_y[i] = all_v[i] - np.einsum('i,i...->...', Z[i,:i], all_x[:i])
            all_x[i] = resolvents[i].prox(all_y[i], alpha)
            
        if callback is not None and callback(itr=itr, x=all_x, v=all_v, y=all_y): break

        if verbose and itr % checkperiod == 0:
            xbar = np.mean(all_x, axis=0)
            dualsum = np.linalg.norm(sum(getDuals(all_v, all_x, Z)))
            print(f"{datetime.now()}\t{itr}\t{np.linalg.norm(all_x - xbar):.3e}\t{dualsum:.3e}")

        all_v -= np.einsum('nl,l...->n...', gammaW, all_x)

        
    x = np.mean(all_x, axis=0)
    
    # Build logs list
    logs = []
    for i in range(n):
        if hasattr(resolvents[i], 'log'):
            logs.append(resolvents[i].log)
        else:
            logs.append([])

    return x, logs, all_x, all_v
