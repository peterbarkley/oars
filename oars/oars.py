# Algorithm Design Functions
# import numpy as np
from oars.matrices.prebuilt import getMT, getFull, getTwoBlockSimilar

def solve(n, data, resolvents, W=None, Z=None, parallel=False, **kwargs):
    '''
    
    Solve the problem with a given W and Z matrix

    Args:
        n (int): the number of nodes
        data (list): list of dictionaries containing the problem data
        resolvents (list): list of uninitialized resolvent classes
        W (ndarray): W matrix
        Z (ndarray): Z matrix
        parallel (bool): whether to run the algorithm in parallel
        kwargs: additional keyword arguments for the algorithm

                - itrs (int): the number of iterations
                - gamma (float): the consensus parameter
                - alpha (float): the resolvent scaling parameter
                - verbose (bool): whether to print verbose output

    Returns:
        x, results (ndarray, list): tuple with the solution and a list of dictionaries with the results for each resolvent

    Examples:
        >>> from oars.utils.proxs import quadprox
        >>> from oars import solve
        >>> from oars.matrices import getFull
        >>> import numpy as np
        >>> vals = np.array([0, 1, 3, 40])
        >>> n = len(vals)
        >>> proxs = [quadprox]*n
        >>> Z, W = getFull(n)
        >>> x, results = solve(n, vals, proxs, W, Z, itrs=1000, vartol=1e-6, gamma=1.0)
        Converged in objective value, iteration 13
        >>> x
        10.999999999990674
        >>> results
        [{'x': 10.999999999906539, 'v': 22.00000000003744}, {'x': 11.000000000103075, 'v': 13.66666666663539}, {'x': 10.999999999962412, 'v': 4.333333333327117}, {'x': 10.999999999990674, 'v': -40.0}]
        '''

    if parallel:
        from oars.algorithms.parallel import parallelAlgorithm
        alg = parallelAlgorithm
        if Z is None or W is None:
            Z, W = getTwoBlockSimilar(n)
    else:
        from oars.algorithms.serial import serialAlgorithm
        alg = serialAlgorithm
        if Z is None or W is None:
            Z, W = getFull(n)
        
    return alg(n, data, resolvents, W, Z, **kwargs)

def solveMT(n, data, resolvents, **kwargs):
    '''
    Solve the problem with the Malitsky-Tam W and Z matrices

    Args:
        n (int): the number of nodes
        data (list): list of dictionaries containing the problem data
        resolvents (list): list of uninitialized resolvent classes
        kwargs: additional keyword arguments for the algorithm

                - itrs (int): the number of iterations
                - gamma (float): the consensus parameter
                - alpha (float): the resolvent scaling parameter
                - verbose (bool): whether to print verbose output

    Returns:
        x, results (ndarray, list): tuple with the solution and a list of dictionaries with the results for each resolvent

    Examples:
        >>> from oars.utils.proxs import quadprox
        >>> from oars import solveMT
        >>> import numpy as np
        >>> vals = np.array([0, 1, 3, 40])
        >>> n = len(vals)
        >>> proxs = [quadprox]*n
        >>> x, results = solveMT(n, vals, proxs, itrs=1000, vartol=1e-6, gamma=1.0)
        Converged in objective value, iteration 69
        >>> x
        10.999999857565648
        >>> results
        [{'x': 10.999999565156383, 'v': 21.99999932717702}, {'x': 10.999999762020636, 'v': 9.99999996819179}, {'x': 10.99999996819179, 'v': 8.00000013489378}, {'x': 11.000000134893778, 'v': -39.999999430262605}]
    '''

    Z, W = getMT(n)
    return solve(n, data, resolvents, W, Z, **kwargs)


