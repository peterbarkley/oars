import numpy as np
from oars.algorithms.reducedph import getVar, getFullVariable
from datetime import datetime

def dadmmAlgorithm(data, resolvents, neighbors, warmstartprimal=None, alpha=1.0, itrs=1001, verbose=False, callback=None):

    n = len(resolvents)
    for i in range(n):
        resolvents[i] = resolvents[i](**data[i])

    if warmstartprimal is not None:
        V = np.array([warmstartprimal.copy() for i in range(n)])
    else:
        V = np.zeros((n, ) + resolvents[0].shape) # Initialize Z, the consensus variable

    U = V.copy() # Initialize U, the variable for storing the resolvents
    R = V.copy() # Initialize R, the variable for storing the average of the neighbors
    UR_old = U + R
    
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||') 
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(n):
            U[i] = resolvents[i].prox(V[i], alpha/len(neighbors[i]))

        for i in range(n):
            R[i] = (np.sum([U[j] for j in neighbors[i]], axis=0))/len(neighbors[i])
            V[i] = V[i] + R[i] - 0.5*UR_old[i]
            UR_old[i] = R[i] + U[i]

        if callback is not None and callback(itr=itr, x=U, y=V): break

        if verbose and itr % checkperiod == 0:
            xbar = np.mean(U, axis=0)
            print(f"{datetime.now()}\t{itr}\t{np.linalg.norm(U - xbar):.3e}")

    x = np.mean(U, axis=0)
    
    # Build logs list
    logs = []
    for i in range(n):
        if hasattr(resolvents[i], 'log'):
            logs.append(resolvents[i].log)
        else:
            logs.append([])

    return x, logs, U

def pg_extra(data, A, B, W, bar_W=None, itrs=1001, alpha=1.0, warmstartprimal=None, verbose=False, callback=None):
    """
    Run the PG-EXTRA algorithm for decentralized composite optimization.

    Args:
        n (int): the number of agents
        data (list): list of lists containing the problem data for each operator, with prox data in data[0] and grad data in data[1]
        A (list): list of prox operator classes
        B (list): list of grad operator classes
        W (ndarray): mixing matrix (n, n)
        bar_W (ndarray): another mixing matrix (n, n), often (W + I)/2
        itrs (int, optional): the number of iterations
        alpha (float, optional): the step size
        warmstartprimal (ndarray, optional): initial point for all agents, shape p
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (ndarray): mean of the agents' solutions at termination
        logs (list): list of logs for the operators
        all_x (ndarray): solutions of all agents

    """

    if bar_W is None:
        bar_W = 0.5*(np.eye(W.shape[0]) + W)

    assert len(A) == len(B)
    n = len(A)
    for i in range(n):
        A[i] = A[i](**data[0][i])
    for j in range(n):
        B[j] = B[j](**data[1][j])
    shape = A[0].shape
    if warmstartprimal is not None:
        x_zero = np.array([warmstartprimal for _ in range(n)])
    else:
        x_zero = np.zeros((n,) + shape)

    x_one = np.zeros((n,) + shape)
    half_x = np.zeros((n,) + shape)
    grad_zero = np.zeros((n,) + shape)
    grad_one = np.zeros((n,) + shape)

    # Initialization
    for i in range(n):
        grad_zero[i] = B[i].grad(x_zero[i])
        half_x[i] = sum(W[i,j]*x_zero[j] for j in range(n)) - alpha*grad_zero[i]
        x_one[i] = A[i].prox(half_x[i], alpha)

    if verbose:
        print('date\t\ttime\t\titr\t||x-bar(x)||')
        checkperiod = max(itrs // 10, 1)

    # Main loop
    for itr in range(itrs//2):
        for i in range(n):
            grad_one[i] = B[i].grad(x_one[i])
            half_x[i] += sum(W[i,j]*x_one[j] - bar_W[i,j]*x_zero[j] for j in range(n)) - alpha*(grad_one[i] - grad_zero[i])
            x_zero[i] = A[i].prox(half_x[i], alpha)
            
            grad_zero[i] = B[i].grad(x_zero[i])
            half_x[i] += sum(W[i,j]*x_zero[j] - bar_W[i,j]*x_one[j] for j in range(n)) - alpha*(grad_zero[i] - grad_one[i])
            x_one[i] = A[i].prox(half_x[i], alpha)

        if callback is not None and callback(itr=itr*2, x=x_one):
            break
        
        if verbose and itr % checkperiod == 0:
            xbar = np.mean(x_one, axis=0)
            consensus_error = np.linalg.norm(x_one - xbar)
            print(f"{datetime.now()}\t{itr}\t{consensus_error:.3e}")
    
    x = np.mean(x_one, axis=0)
    
    # Collect logs
    logs = []
    for op in A + B:
        if hasattr(op, 'log'):
            logs.append(op.log)
        else:
            logs.append([])
    
    return x, logs, x_one

def p_extra(data, A, W, bar_W=None, itrs=1001, alpha=1.0, warmstartprimal=None, verbose=False, callback=None):
    """
    Run the P-EXTRA algorithm for decentralized composite optimization.

    Args:
        n (int): the number of agents
        data (list): list containing the problem data for each operator, with prox data in data[0] and grad data in data[1]
        A (list): list of prox operator classes
        W (ndarray): mixing matrix (n, n)
        bar_W (ndarray): another mixing matrix (n, n), often (W + I)/2
        itrs (int, optional): the number of iterations
        alpha (float, optional): the step size
        warmstartprimal (ndarray, optional): initial point for all agents, shape p
        verbose (bool, optional): True for verbose output
        callback (function, optional): callback function

    Returns:
        x (ndarray): mean of the agents' solutions at termination
        logs (list): list of logs for the operators
        all_x (ndarray): solutions of all agents

    """

    if bar_W is None:
        bar_W = 0.5*(np.eye(W.shape[0]) + W)

    n = len(A)
    for i in range(n):
        A[i] = A[i](**data[i])
    shape = A[0].shape
    if warmstartprimal is not None:
        x_zero = np.array([warmstartprimal for _ in range(n)])
    else:
        x_zero = np.zeros((n,) + shape)

    x_one = np.zeros((n,) + shape)
    half_x = np.zeros((n,) + shape)

    # Initialization
    for i in range(n):
        half_x[i] = sum(W[i,j]*x_zero[j] for j in range(n))
        x_one[i] = A[i].prox(half_x[i], alpha)

    if verbose:
        print('date\t\ttime\t\titr\t||x-bar(x)||')
        checkperiod = max(itrs // 10, 1)

    # Main loop
    for itr in range(itrs//2):
        for i in range(n):
            half_x[i] += sum(W[i,j]*x_one[j] - bar_W[i,j]*x_zero[j] for j in range(n))
            x_zero[i] = A[i].prox(half_x[i], alpha)
            
            half_x[i] += sum(W[i,j]*x_zero[j] - bar_W[i,j]*x_one[j] for j in range(n))
            x_one[i] = A[i].prox(half_x[i], alpha)

        if callback is not None and callback(itr=itr*2, x=x_one):
            break
        
        if verbose and (itr*2) % checkperiod == 0:
            xbar = np.mean(x_one, axis=0)
            consensus_error = np.linalg.norm(x_one - xbar)
            print(f"{datetime.now()}\t{itr*2}\t{consensus_error:.3e}")
    
    x = np.mean(x_one, axis=0)
    
    # Collect logs
    logs = []
    for op in A:
        if hasattr(op, 'log'):
            logs.append(op.log)
        else:
            logs.append([])
    
    return x, logs, x_one


def progressiveHedgingAlgorithm(p, data, A, I, varshapes, alpha=1.0, itrs=1001, verbose=False, callback=None):
    num_subvectors = len(p)
    num_functs = len(data)

    all_x = [getVar(data[i]) for i in range(num_functs)]
    all_v = [all_x[i].copy() for i in range(num_functs)]
    xbar = [np.zeros(shape) for shape in varshapes]

    if verbose or callback is not None:
        all_y = [all_x[i].copy() for i in range(num_functs)]

    for i in range(num_functs):
        A[i] = A[i](**data[i])

    
    # Run the algorithm
    if verbose: 
        print('date\t\ttime\t\titr\t||x-bar(x)||\t||sum dual||')
        checkperiod = max(itrs//10,1)
    for itr in range(itrs):
        for i in range(num_functs):
            np.copyto(all_x[i],all_v[i])
            all_x[i] *= -alpha
            for k in A[i].vars:
                all_x[i][A[i].indices[k]] += xbar[k]
            if verbose or callback is not None:
                np.copyto(all_y[i],all_x[i])
            all_x[i] = A[i].prox(all_x[i], alpha)
            
        if callback is not None and callback(itr, all_x, all_v, all_y, xbar, A): break

        if verbose and itr % checkperiod == 0:
            ysqdiff = 0.0
            for k in range(num_subvectors):
                if len(I[k]) > 1:
                    ybar = np.mean([all_x[i][A[i].indices[k]] for i in I[k]], axis=0)
                    ysqdiff += sum(np.linalg.norm(all_x[i][A[i].indices[k]] - ybar)**2 for i in I[k])
            subg_sum_norm = sum([np.linalg.norm(sum([(p[k][I[k].index(i)]/alpha)*(all_y[i][A[i].indices[k]]-all_x[i][A[i].indices[k]]) for i in I[k]]))**2 for k in range(num_subvectors)])**0.5
            print(f"{datetime.now()}\t{itr}\t{ysqdiff**0.5:.3e}\t{subg_sum_norm:.3e}")

        for k in range(num_subvectors):
            xbar[k] = sum(p[k][I[k].index(i)]*all_x[i][A[i].indices[k]] for i in I[k])

        for i in range(num_functs):
            for k in A[i].vars:
                all_v[i][A[i].indices[k]] += (1/alpha)*(all_x[i][A[i].indices[k]] - xbar[k])

        
    ybar = getFullVariable(all_x, A, I)
    
    # Build logs list
    logs = []
    for i in range(num_functs):
        if hasattr(A[i], 'log'):
            logs.append(A[i].log)
        else:
            logs.append([])

    return ybar, logs, all_x, all_v
