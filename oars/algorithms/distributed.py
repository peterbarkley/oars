#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np

def initialize(comm, n, data, resolvents, W, Z, gamma=0.8, alpha=1.0):
    """
    Initialize the resolvents and variables

    Args:
        n (int): the number of resolvents
        data (list): list of dictionaries containing the problem data
        resolvents (list): list of uninitialized resolvent classes
        W (ndarray, optional): size (n, n) ndarray for the W matrix
        Z (ndarray, optional): size (n, n) ndarray for the Z matrix
        gamma (float, optional): the consensus parameter
        alpha (float, optional): the resolvent scaling parameter

    """

    Comms_Data = requiredComms(Z, W)

    # Distribute the data
    for j in range(1, n):
        #print("Node 0 sending data to node", j, flush=True)

        comm.send(data[j], dest=j, tag=44) # Data
        comm.send(resolvents[j], dest=j, tag=17) # Resolvent
        comm.send(Comms_Data[j], dest=j, tag=33) # Comms data

    return Comms_Data[0]

# Does not work yet
# Distributed algorithm
def distributedAlgorithm(n, data, resolvents, W, Z, warmstartprimal=None, warmstartdual=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None, verbose=False):
    """
    Distributed algorithm for frugal resolvent splitting

    Args:
        n (int): the number of resolvents
        resolvents (list): list of resolvent classes
        W (ndarray): size (n, n) ndarray for the W matrix
        Z (ndarray): size (n, n) ndarray for the Z matrix
        data (list): list containing the problem data for each resolvent
        warmstartprimal (ndarray, optional): resolvent.shape ndarray for x in v^0
        warmstartdual (list, optional): is a list of n ndarrays for u which sums to 0 in v^0
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in v^{k+1} = v^k - \gamma W x^k
        alpha (float, optional): the resolvent step size in x^{k+1} = J_{\alpha F^i}(y^k)
        vartol (float, optional): is the variable tolerance
        objtol (float, optional): is the objective tolerance
        earlyterm (int, optional): the number of iterations to run before checking for termination
        detectcycle (int, optional): the number of iterations to check for a cycle
        objective (function, optional): the objective function
        verbose (bool, optional): True for verbose output

    Returns:
        tuple (x, results): the solution and a list of dictionaries with the results for each resolvent

            x (ndarray): the solution
            results (list): list of dictionaries with the results for each resolvent
    """

    # nodes = L.shape[0]
    comm = MPI.COMM_WORLD
    i = comm.Get_rank()
    n_size = comm.Get_size()
    if n > n_size - 1:
        raise ValueError("Number of nodes is greater than the number of processes")
    # nodes = n-1

    if i == 0:
        initialization(n, data, resolvents, W, Z, gamma, alpha)
        
        # Run subproblems
        print("Node 0 running subproblem", flush=True)
        #print("Comms data 0", Comms_Data[0], flush=True)
        t = time()
        x, log = subproblem(i, data[i], resolvents[i], W, Z, Comms_Data[i], comm, gamma, itrs, vartol=1e-5, verbose=True)
        print("Time", time() - t)
        
        #timestamp = time()
        # with open('logs'+str(i)+'_'+title+'.json', 'w') as f:
        #     json.dump(log, f)
        #w = np.array(m)
        results = []
        results.append({'x':x})
        x_i = np.zeros(x.shape)
        for k in range(1, n-1):
            comm.Recv(x_i, source=k, tag=0)
            results.append({'x':x_i})
            x += x_i
        xbar = (1/n)*x
        #print(w, proj_w, flush=True)
        # print("alg val", fullValue(fulldata[-1], proj_w))
        #t = time()
        return xbar, results
    elif i < n-1:
        # Receive L and W
        #print(f"Node {i} receiving L and W", flush=True)
        #L = np.zeros((n-1,n-1))
        #W = np.zeros((n-1,n-1))
        #comm.Bcast(L, root=0)
        #comm.Bcast(W, root=0)
        #print(f"Node {i} received L and W", flush=True)
        # Receive the data
        #data = np.array(m)
        data = comm.recv(source=0, tag=44)
        res = comm.recv(source=0, tag=17)
        comms = comm.recv(source=0, tag=33)
        # Run the subproblem
        #print(f"Node {i} running subproblem", flush=True)
        x, log = subproblem(i, data, res, W, Z, comms, comm, gamma, itrs, vartol=1e-2, verbose=True)
        #timestamp = time()
        # with open('logs_wta'+str(i)+'_'+title+'.json', 'w') as f:
        #     json.dump(log, f)
        #w = np.array(i)
        comm.Send(x, dest=0, tag=0)
    elif i == n-1:
        #L = np.zeros((n-1,n-1))
        #W = np.zeros((n-1,n-1))
        #comm.Bcast(L, root=0)
        #comm.Bcast(W, root=0)
        evaluate(m, comm, vartol=1e-5, itrs=itrs) 

def requiredComms(Z, W):
    '''
    Returns a dictionary of the communications required by the given W and L matrices

    Args:
        Z (ndarray): the Z matrix
        W (ndarray): the W matrix

    Returns:
        Comms_Data (list): a list of dictionaries with the required comms data for each node
            WQ (list): nodes which feed only W data into node i
            up_LQ (list): nodes which feed only Z data into node i
            down_LQ (list): nodes which receive only Z data from node i
            up_BQ (list): nodes which feed both W and Z data into node i, and node i feeds W back to
            down_BQ (list): nodes which receive W and Z data from node i
    '''

    # Get the number of nodes
    n = W.shape[0]

    Comms_Data = []
    for i in range(n):
        Comms_Data.append({'WQ':[], 'up_ZQ':[], 'down_ZQ':[], 'up_BQ':[], 'down_BQ':[]})

    for i in range(n):
        comms_i = Comms_Data[i]
        for j in range(i):
            comms_j = Comms_Data[j]
            if not np.isclose(W[i,j], 0, atol=1e-3):
                if not np.isclose(Z[i,j], 0, atol=1e-3):
                    comms_i['up_BQ'].append(j)
                    comms_j['down_BQ'].append(i)
                else:
                    comms_j['WQ'].append(i)
                    comms_i['WQ'].append(j)
            elif not np.isclose(Z[i,j], 0, atol=1e-3):
                comms_i['up_ZQ'].append(j)
                comms_j['down_ZQ'].append(i)

    return Comms_Data

#def solve(s, itrs=100, gamma=0.5, verbose=False, terminate=None):
def subproblem(i, data, resolvents, W, Z, comms_data, comm, gamma=0.5, itrs=100, vartol=None, verbose=False):
    
    # comm = MPI.COMM_WORLD
    # i = comm.Get_rank()
    # size = comm.Get_size()

    # L, W = oars.getMT(size)
    # comms_data_all = requiredComms(L, W)
    # comms_data = comms_data_all[i]
    #s = 10
    resolvent = resolvents(data)
    s = resolvent.shape
    #s = data.shape
    buffer = np.ones(s, dtype=np.float64)
    local_v = np.zeros(s, dtype=np.float64)
    local_r = np.zeros(s, dtype=np.float64)
    v_temp = np.zeros(s, dtype=np.float64)
    n = W.shape[0]
    itr = 0
    t_itr = np.array(itrs, 'i')
    terminated = False
    while itr < itrs:
        if vartol is not None and comm.Iprobe(source=n, tag=0):
            itrs = comm.recv(source=n, tag=0)
            terminated = True
        if verbose and itr % 500 == 0:
            print(f'Node {i} iteration {itr}', flush=True)

        # Get data from upstream L queue
        for k in comms_data['up_ZQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            local_r -= Z[i,k]*buffer

        # Pull from the B queues, update r and v_temp
        for k in comms_data['up_BQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            local_r -= Z[i,k]*buffer
            v_temp += W[i,k]*buffer

        # Solve the problem
        w_value = resolvent.prox(local_v + local_r)

        # Terminate if needed
        if i==0 and vartol is not None and not terminated:
             #print(f'Node {i} w_value sending for eval: {w_value}', flush=True)
             comm.Send(w_value, dest=n, tag=itr)

        # Put data in downstream queues
        for k in comms_data['down_ZQ']:
            comm.Isend(w_value, dest=k, tag=itr)
        for k in comms_data['down_BQ']:
            comm.Isend(w_value, dest=k, tag=itr)

        # Put data in upstream W queues
        for k in comms_data['WQ']:
            comm.Isend(w_value, dest=k, tag=itr)
        for k in comms_data['up_BQ']:
            comm.Isend(w_value, dest=k, tag=itr)

        # Update v from all W queues
        for k in comms_data['WQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            v_temp += W[i,k]*buffer
            
        # Update v from all B queues
        for k in comms_data['down_BQ']:
            req = comm.Irecv(buffer, source=k, tag=itr)
            req.Wait()
            v_temp += W[i,k]*buffer
        #v_temp += sum([W[i,k]*queue[k,i].get() for k in comms_data['down_BQ']])
        

        local_v = local_v - gamma*(W[i,i]*w_value+v_temp)
        
        # Zero out v_temp without reallocating memory
        v_temp.fill(0)
        local_r.fill(0)
        itr += 1

    #print(f'Node {i} w_value: {w_value}', flush=True)
    # return w_value and log if it is in the resolvent
    if hasattr(resolvent, 'log'):
        log = resolvent.log
    else:
        log = None
    return w_value, log

def evaluate(n, shape, comm, vartol=1e-7, itrs=100):
    """
    Evaluate the convergence of the algorithm and terminate if needed

    Args:
        s (tuple): the shape of the data
        comm (MPI communicator): the MPI communicator
        itrs (int): the number of iterations to run
    
    """
    last = np.zeros(shape, dtype=np.float64)
    buffer = np.zeros(shape, dtype=np.float64)
    counter = 0
    itr = 0
    while counter < n and itr < itrs:
        comm.Recv(buffer, source=0, tag=itr)
        w = buffer.copy()
        # Print last and buffer
        #print(f'Counter: {counter}, Last: {last}, Buffer: {w}', flush=True)
        if np.linalg.norm(w - last) < vartol:
            counter += 1
        else:
            counter = 0
        last = w
        itr += 1
    # print counter, last and buff
    #print(f'Counter: {counter}, Last: {last}, Buffer: {w}', flush=True)
    print(f'Reached termination criteria on Iteration {itr}', flush=True)

    # Terminate the other processes
    advance = 50
    terminate_itr = itr + advance
    if itr < itrs - advance:
        # t_itr = np.array(terminate_itr, 'i')
        # comm.Bcast([t_itr, MPI.INT], root=n)
        for i in range(n):
            #print(f'Sending termination criteria {terminate_itr} to {i}', flush=True)
            comm.send(terminate_itr, dest=i, tag=0)