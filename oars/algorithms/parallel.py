import numpy as np
import multiprocessing as mp
from oars.algorithms.helpers import ConvergenceChecker, getWarmPrimal

def parallelAlgorithm(n, data, resolvents, W, L, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None, objtol=None, earlyterm=None, detectcycle=0, objective=None, verbose=False):
    """Run the parallel algorithm
    Args:
        n (int): the number of nodes
        data (list): list containing the problem data for each node
        resolvents (list): list of :math:`n` resolvent functions
        W (ndarray): size (n, n) ndarray for the :math:`W` matrix
        L (ndarray): size (n, n) ndarray for the :math:`L` matrix
        warmstartprimal (ndarray, optional): resolvent.shape ndarray for :math:`x` in v^0
        warmstartdual (list, optional): is a list of n ndarrays for :math:`u` which sums to 0 in v^0
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        vartol (float, optional): is the variable tolerance
        objtol (float, optional): is the objective tolerance
        earlyterm (int, optional): the number of variables that must agree to terminate early and solve explicitly for the remaining variables
        detectcycle (int, optional): the number of iterations to check for cycling
        objective (function, optional): the objective function
        verbose (bool, optional): True for verbose output

    Returns:
        xbar (ndarray): the solution
        results (list): list of dictionaries with the results for each node
    """
    
    # Create the queues
    man = mp.Manager()
    Queue_Array, Comms_Data = requiredQueues(man, W, L)
    if vartol is not None or objtol is not None:
        terminate = man.Value('i',0) #man.Event()
        Queue_Array['terminate'] = man.Queue()
        # Create evaluation process
        if len(data) > n:
            d = data[n]
        else:
            d = None
        evalProcess = mp.Process(target=evaluate, args=(Queue_Array['terminate'], terminate, vartol, objtol, d, objective, itrs, verbose))
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
        if objective is not None and earlyterm is not None and (0 < np.sum(xdev != 0) < earlyterm):
            it = np.nditer(xbar, flags=['multi_index'])
            fixed = {tuple(it.multi_index):xbar[it.multi_index] for i in it if xdev[it.multi_index] == 0 }
            # Solve fixed problem
            xbar = objective(fixdict=fixed, verbose=verbose)

    if verbose:
        results[0]['alg_time'] = alg_time
    return xbar, results


def requiredQueues(man, W, L):
    '''
    Returns the queues for the given W and L matrices
    Inputs:
    man is the multiprocessing manager
    W is the W matrix
    L is the L matrix

    Returns:
    Queue_Array is the dictionary of the queues with keys (i,j) for the queues from i to j
    Comms_Data is a list of the required comms data for each node
    The comms data entry for node i is a dictionary with the following keys:
    WQ: nodes which feed only W data into node i
    up_LQ: nodes which feed only L data into node i
    down_LQ: nodes which receive only L data from node i
    up_BQ: nodes which feed both W and L data into node i, and node i feeds W back to
    down_BQ: nodes which receive W and L data from node i
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
    '''Solves the parallel subproblem for node i
    Inputs:
    i is the node number
    data is a dictionary containing arguments for the problem
    problem_builder is a pyprox class for the problem
    W is the W matrix
    L is the L matrix
    comms_data is a dictionary with the following keys:
    WQ: nodes which feed only W data into node i
    up_LQ: nodes which feed only L data into node i
    down_LQ: nodes which receive only L data from node i
    up_BQ: nodes which feed both W and L data into node i, and node i feeds W back to
    down_BQ: nodes which receive W and L data from node i
    queue is the array of queues
    gamma is the consensus parameter
    itrs is the number of iterations

    Returns:
    Dictionary with the following
    w: the solution
    v: the consensus variable
    log: the log of the problem (if available)
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
    while itr < itrs:
        if terminate is not None and terminate.value != 0:
            print('Node', i, 'received terminate value', terminate.value, 'on iteration', itr)
            if terminate.value < itr:
                break
                #terminate.value = itr + 1
            itrs = terminate.value
        # if verbose and itr % 1000 == 999:
        #     print(f'Node {i} iteration {itr}')

        # Get data from upstream L queue
        for k in comms_data['up_LQ']:
            local_r += L[i,k]*queue[k,i].get()
            
        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            temp = queue[k,i].get()
            local_r += L[i,k]*temp
            v_temp += W[i,k]*temp

        # Solve the problem
        w_temp = resolvent.prox(local_v + local_r, alpha)
        if verbose:
            delta = np.linalg.norm(w_temp - w_value, 'fro')
            print("Node", i, "Itr", itr, "Var Difference", delta)
        w_value = w_temp

        # Terminate if needed
        if i==0 and terminate is not None:
            queue['terminate'].put(w_value)
            

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
            #temp = queue[k,i].get()            
            v_temp += W[i,k]*queue[k,i].get() 
            
        # Update v from all B queues
        for k in comms_data['down_BQ']:
            v_temp += W[i,k]*queue[k,i].get()
        #v_temp += sum([W[i,k]*queue[k,i].get() for k in comms_data['down_BQ']])
        
        v_update = gamma*(W[i,i]*w_value+v_temp)
        if verbose:
            delta = np.linalg.norm(v_update, 'fro')
            print("Node", i, "Itr", itr, "Consensus Difference", delta)
        local_v = local_v - v_update
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)
        local_r.fill(0)
        itr += 1
        # Log the value -- needs to be done in another process
        # if i == 0:
        #     prob_val = fullValue(data, w_value)
        #     log_val.append(prob_val)

    # Return the solution
    # if i == 0:
    #     return {'w':w_value, 'v':local_v, 'log':log_val}
    if hasattr(resolvent, 'log'):
        return {'x':w_value, 'v':local_v, 'log':resolvent.log}
    return {'x':w_value, 'v':local_v}

def evaluate(terminateQueue, terminate, vartol, objtol, data, objective, itrs, earlyterm=None, detectcycle=0, verbose=False):
    """Evaluate the termination conditions and set the terminate value if needed
    The terminate value is set a number of iterations ahead of the convergence iteration
    
    Inputs:
    terminateQueue is the queue for termination
    terminate is the multiprocessing value for termination
    vartol is the variable tolerance
    objtol is the objective tolerance
    data is the data for the objective function
    objective is the objective function
    itrs is the number of iterations
    verbose is a boolean for verbose output


    """
    x = terminateQueue.get()
    #n = len(x) # x is just from node 0
    varcounter = 0
    objcounter = 0
    itr = 0
    itrs -= 10
    while itr < itrs:
        if verbose:print('iteration', itr+1)
        prev_x = x
        x = terminateQueue.get()        
        if vartol is not None:
            delta = np.linalg.norm(x-prev_x)
            print("vartol check delta", delta)
            if delta < vartol:
                varcounter += 1
                if varcounter >= 10:
                    terminate.value = itr + 10
                    if verbose:
                        print('Converged on vartol on iteration', itr)
                    break
            else:
                varcounter = 0
        if objtol is not None:
            if abs(objective(x)-objective(prev_x)) < objtol:
                objcounter += 1
                if objcounter >= 10:
                    terminate.value = itr + 10
                    if verbose:
                        print('Converged on objtol on iteration', itr)
                    break
            else:
                objcounter = 0
        if earlyterm is not None:
            xbar = np.mean(x, axis=0)
            xdev = sum(abs(x[i] - xbar) for i in range(len(x)))
            if np.sum(xdev != 0) < earlyterm:
                terminate.value = itr + 10
                if verbose:
                    print('Converged on earlyterm on iteration', itr)
                it = np.nditer(xbar, flags=['multi_index'])
                fixed = {tuple(it.multi_index):xbar[it.multi_index] for i in it if xdev[it.multi_index] == 0 }
                # Solve fixed problem
                x = objective(fixdict=fixed)
                # Return solution
        itr += 1
  