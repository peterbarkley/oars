from numpy import isclose, zeros, mean, tril, einsum, array
from numpy import sum as npsum
from numpy.linalg import norm
from datetime import datetime

def getBafter(K, n, m):
    bafter = [[] for _ in range(n)]
    for j in range(m):
        i = n-1
        while isclose(K[j,i], 0.0):
            i -=1
        bafter[i].append(j)

    return bafter


def abraAlgorithm(data, A, B, W, Z, K=None, Q=None, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, verbose=False, callback=None):
    """
    Run the frugal resolvent splitting algorithm defined by Z and W in serial

    Args:
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
    """
    
    # Initialize the resolvents and variables
    n = len(A)
    m = len(B)
    for i in range(n):
        A[i] = A[i](**data[0][i])
    for j in range(m):
        B[j] = B[j](**data[1][j])
    shape = A[0].shape
    all_x = zeros((n,) + shape)
    all_y = zeros((n,) + shape)
    all_b = zeros((m,) + shape)

    if warmstartprimal is None:
        all_v = zeros((n,) + shape)
    else:
        all_v = array([(Z[i,i] + 2.0*sum(Z[i,j] for j in range(i)))*warmstartprimal for i in range(n)])
    if warmstartdual is not None:
        all_v += warmstartdual

    gammaW = gamma*W
    bafter = getBafter(K, n, m)
    L = -2.0*tril(Z, -1)

    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        all_b = zeros((m,) + shape)
        for i in range(n):
            if i == 0:
                all_y[i] = all_v[0]
            else:
                all_y[i] = all_v[i] + einsum('i,i...->...', L[i,:i], all_x[:i]) - alpha*sum(Q[i,j]*all_b[j] for j in range(m))
            # print('y', i, y)
            all_y[i] /= Z[i,i]
            all_x[i] = A[i].prox(all_y[i], alpha/Z[i, i])
            for j in bafter[i]:
                # y = einsum('j,i...->...', K[j,:i+1], all_x[:i+1])
                y = sum(K[j, d]*all_x[d] for d in range(i+1))
                # print('b', i, j, y)
                all_b[j] = B[j].grad(y)
            
        if callback is not None and callback(itr=itr, x=all_x, v=all_v, b=all_b, y=all_y): break

        if verbose and itr % checkperiod == 0:
            xbar = mean(all_x, axis=0)
            
            subgs = array([Z[i,i]*(all_y[i] - all_x[i]) for i in range(n)])
            subg_sum_norm = norm(npsum(subgs, axis=0))
            print(f"{datetime.now()}\t{itr}\t{norm(all_x - xbar):.3e}\t{subg_sum_norm:.3e}") 

        all_v -= einsum('nl,l...->n...', gammaW, all_x)

        
    x = mean(all_x, axis=0)
    
    # Build logs list
    logs = []
    for i in range(n):
        if hasattr(A[i], 'log'):
            logs.append(A[i].log)
        else:
            logs.append([])
    for j in range(m):
        if hasattr(B[j], 'log'):
            logs.append(B[j].log)
        else:
            logs.append([])

    return x, logs, all_x, all_v, all_b

