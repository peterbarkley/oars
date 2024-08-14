import numpy as np
import multiprocessing as mp
from oars.algorithms.helpers import ConvergenceChecker, getWarmPrimal
from time import time

def parallelAlgorithm(n, data, resolvents, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None,  checkperiod=1, verbose=False):
    """Run the parallel algorithm
    Args:
        n (int): the number of resolvents
        data (list): list containing the problem data for each resolvent
        resolvents (list): list of :math:`n` resolvent functions
        W (ndarray): size (n, n) ndarray for the :math:`W` matrix
        Z (ndarray): size (n, n) ndarray for the :math:`Z` matrix
        warmstartprimal (ndarray, optional): resolvent.shape ndarray for :math:`x` in v^0
        warmstartdual (list, optional): is a list of n ndarrays for :math:`u` which sums to 0 in v^0
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        vartol (float, optional): is the variable tolerance
        earlyterm (int, optional): the number of variables that must agree to terminate early and solve explicitly for the remaining variables
        detectcycle (int, optional): the number of iterations to check for cycling
        verbose (bool, optional): True for verbose output

    Returns:
        xbar (ndarray): the solution
        results (list): list of dictionaries with the results for each node
    """
    L = -np.tril(Z, -1)

    # Create the queues
    man = mp.Manager()
    Queue_Array, Comms_Data = requiredQueues(man, W, L)
    if vartol is not None:
        terminate = man.Value('i',0) #man.Event()
        Queue_Array['terminate'] = [man.Queue() for _ in range(n)]

        # Create evaluation process
        evalProcess = mp.Process(target=evaluate, args=(n, Queue_Array['terminate'], terminate, vartol, itrs, checkperiod, verbose))
        evalProcess.start()
    else:
        terminate = None
    
    # Set v0
    if warmstartprimal is not None:
        all_v = getWarmPrimal(warmstartprimal, L)
        if verbose:print('warmstartprimal', all_v)
    else:
        all_v = [0 for _ in range(n)]
    if warmstartdual is not None:
        all_v = [all_v[i] + warmstartdual[i] for i in range(n)]
        if verbose:print('warmstart final', all_v)

    # Run subproblems in parallel
    if verbose:
        print('Starting Parallel Algorithm')
        t = time()
    with mp.Pool(processes=n) as p:
        params = [(i, data[i], resolvents[i], all_v[i], W, L, Comms_Data[i], Queue_Array, gamma, alpha, itrs, terminate, verbose) for i in range(n)]
        results = p.starmap(subproblem, params)
    if verbose:
        alg_time = time()-t
        print('Parallel Algorithm Loop Time:', alg_time)

    xbar = np.mean([results[i]['x'] for i in range(n)], axis=0)
    # Join the evaluation process
    if terminate is not None:        
        evalProcess.join()
        xdev = sum(abs(results[i]['x'] - xbar) for i in range(n))


    if verbose:
        results[0]['alg_time'] = alg_time
    return xbar, results


def requiredQueues(man, W, L):
    '''
    Returns the queues for the given W and L matrices

    Args:
        man (multiprocessing manager): the manager
        W (ndarray): is the n x n W matrix
        L (ndarray): is the n x nL matrix

    Returns:
        Queue_Array (dict): is the dictionary of the queues with keys (i,j) for the queues from i to j
        Comms_Data (list): is a list of the required comms data for each node
                            The comms data entry for node i is a dictionary with the following keys:

                            - WQ: nodes which feed only W data into node i
                            - up_LQ: nodes which feed only L data into node i
                            - down_LQ: nodes which receive only L data from node i
                            - up_BQ: nodes which feed both W and L data into node i, and node i feeds W back to
                            - down_BQ: nodes which receive W and L data from node i
    '''
    # Get the number of nodes
    n = W.shape[0]

    Queue_Array = {} # Queues required by non-zero off diagonal elements of W
    Comms_Data = []
    for i in range(n):
        WQ = []
        up_LQ = []
        down_LQ = []
        up_BQ = []
        down_BQ = []
        Comms_Data.append({'WQ':WQ, 'up_LQ':up_LQ, 'down_LQ':down_LQ, 'up_BQ':up_BQ, 'down_BQ':down_BQ})

    for i in range(n):
        comms_i = Comms_Data[i]
        for j in range(i):
            comms_j = Comms_Data[j]
            if not np.isclose(W[i,j],0.0):
                if (i,j) not in Queue_Array:
                    queue_ij = man.Queue()
                    Queue_Array[i,j] = queue_ij
                if (j,i) not in Queue_Array:
                    queue_ji = man.Queue()
                    Queue_Array[j,i] = queue_ji
                if not np.isclose(L[i,j],0.0):
                    comms_i['up_BQ'].append(j)
                    comms_j['down_BQ'].append(i)
                else:
                    comms_j['WQ'].append(i)
                    comms_i['WQ'].append(j)
            elif not np.isclose(L[i,j],0.0):
                if (j,i) not in Queue_Array:
                    queue_ji = man.Queue()
                    Queue_Array[j,i] = queue_ji
                comms_i['up_LQ'].append(j)
                comms_j['down_LQ'].append(i)

    return Queue_Array, Comms_Data

def subproblem(i, data, problem_builder, v0, W, L, comms_data, queue, gamma=0.5, alpha=1.0, itrs=501, terminate=None, verbose=False):
    '''
    Solves the parallel subproblem for node i

    Args:
        i (int): is the node number
        data (dict): is a dictionary containing arguments for the problem
        problem_builder (class): is a prox class for the problem
        W (ndarray): is the n x n W matrix
        L (ndarray): is the n x n L matrix
        comms_data (dict): is a dictionary with the following keys

                            - WQ: nodes which feed only W data into node i
                            - up_LQ: nodes which feed only L data into node i
                            - down_LQ: nodes which receive only L data from node i
                            - up_BQ: nodes which feed both W and L data into node i, and node i feeds W back to
                            - down_BQ: nodes which receive W and L data from node i
        queue (dict): is the array of queues
        gamma (float): is the consensus parameter
        itrs (int): is the number of iterations
        terminate (multiprocessing value): is the termination value
        verbose (bool): is a boolean for verbose output

    Returns:
        tuple (x, results): 
            x (ndarray): the solution
            results (dict): is a dictionary with the following keys

                            - x: the solution
                            - v: the consensus variable
                            - log: the log of the problem (if available)
    '''

    # Create the problem
    resolvent = problem_builder(data)
    m = resolvent.shape
    v_temp = np.zeros(m)
    local_v = v0 # + np.zeros(m)
    local_r = np.zeros(m)
    w_value = np.zeros(m)

    # Iterate over the problem
    itr = 0
    itr_period = itrs//10
    while itr < itrs:
        if terminate is not None and terminate.value != 0:
            if verbose:
                print('Node', i, 'received terminate value', terminate.value, 'on iteration', itr)
            if terminate.value < itr:
                break
                #terminate.value = itr + 1
            itrs = terminate.value
        if itr % itr_period == 0:
            print(f'Node {i} iteration {itr}')

        # Get data from upstream L queue
        for k in comms_data['up_LQ']:
            local_r += L[i,k]*queue[k,i].get()
            
        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            temp = queue[k,i].get()
            local_r += L[i,k]*temp
            v_temp += W[i,k]*temp

        # Solve the problem
        w_value = resolvent.prox(local_v + local_r, alpha)

            

        # Put data in downstream queues
        for k in comms_data['down_LQ']:
            queue[i,k].put(w_value)
        for k in comms_data['down_BQ']:
            queue[i,k].put(w_value)

        # Put data in upstream W queues
        for k in comms_data['WQ']:
            queue[i,k].put(w_value)
        for k in comms_data['up_BQ']:
            queue[i,k].put(w_value)

        # Update v from all W queues
        for k in comms_data['WQ']:         
            v_temp += W[i,k]*queue[k,i].get() 
            
        # Update v from all B queues
        for k in comms_data['down_BQ']:
            v_temp += W[i,k]*queue[k,i].get()
        
        v_update = gamma*(W[i,i]*w_value+v_temp)

        local_v = local_v - v_update
        
        # Terminate if needed
        if terminate is not None:
            queue['terminate'][i].put(v_update)

        # Zero out v_temp without reallocating memory
        v_temp.fill(0)
        local_r.fill(0)
        itr += 1

    if hasattr(resolvent, 'log'):
        return {'x':w_value, 'v':local_v, 'log':resolvent.log}
    return {'x':w_value, 'v':local_v}

def evaluate(n, terminateQueue, terminate, vartol, itrs, checkperiod=1, verbose=False):
    """
    Evaluate the termination conditions and set the terminate value if needed
    The terminate value is set a number of iterations ahead of the convergence iteration
    
    Args:
        n (int): the number of nodes
        terminateQueue (list): the list of queues for termination
        terminate (multiprocessing value): the termination value
        vartol (float): the variable tolerance
        itrs (int): the number of iterations
        checkperiod (int): the number of iterations between checks (not implemented)
        verbose (bool): True for verbose output
        
    """
    v = []
    for i in range(n):
        v.append(terminateQueue[i].get())
    #n = len(x) # x is just from node 0
    varcounter = 0
    itr = 0
    itrs -= n
    while itr < itrs:
        if verbose:print('iteration', itr+1)
        prev_v = v.copy()
        for i in range(n):
            v[i] = terminateQueue[i].get()
        delta = sum(np.linalg.norm(v[i], 'fro') for i in range(n))
        if verbose:print("vartol check delta", delta)
        if delta < vartol:
            varcounter += 1
            if varcounter >= n:
                terminate.value = itr + 2*n
                if verbose:
                    print('Converged on vartol on iteration', itr)
                break
        else:
            varcounter = 0

        itr += 1
  