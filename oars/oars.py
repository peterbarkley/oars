# Algorithm Design Functions
import numpy as np
from oars.matrices import getMT

def solve(n, data, resolvents, W, Z, parallel=False, **kwargs):
    '''
    
    Solve the problem with a given W and L matrix

    Args:
        n (int): the number of nodes
        data (list): list of dictionaries containing the problem data
        resolvents (list): list of resolvent functions
        W (ndarray): W matrix
        Z (ndarray): Z matrix
        parallel (bool): whether to run the algorithm in parallel
        kwargs: additional keyword arguments for the algorithm

                itrs (int): the number of iterations
                gamma (float): the consensus parameter
                alpha (float): the prox scaling parameter
                verbose (bool): whether to print verbose output

    Returns:
        x, results (ndarray, list): tuple with the solution and a list of dictionaries with the results for each resolvent
    '''

    if parallel:
        from oars.algorithms.parallel import parallelAlgorithm
        alg = parallelAlgorithm
    else:
        from oars.algorithms.serial import serialAlgorithm
        alg = serialAlgorithm
        
    x, results = alg(n, data, resolvents, W, Z, **kwargs)
    return x, results

def solveMT(n, data, resolvents, **kwargs):
    # Solve the problem with the Malitsky-Tam W and L matrices
    Z, W = getMT(n)
    return solve(n, data, resolvents, W, Z, **kwargs)


