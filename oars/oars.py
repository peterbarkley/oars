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
        x (ndarray): resolvent.shape ndarray of the mean over the node solutions at termination
        logs (list): list of n logs for the operators
        all_x (ndarray): n x resolvent.shape ndarray of the node solution
        all_v (ndarray): n x resolvent.shape ndarray of the consensus iterates at solution

    Examples:
        >>> from oars.utils.proxs import quadProx
        >>> from oars import solve
        >>> import numpy as np
        >>> d = 2
        >>> n = 3
        >>> Q = [np.eye(d)]*n
        >>> P = [np.array([1,1]), np.array([2,3]), np.array([3,2])]
        >>> x, _, _, _ = solve(n, [{'Q': Q[i], 'P':P[i]} for i in range(n)], [quadProx for _ in range(n)])
        >>> x
        array([2., 2.])
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


