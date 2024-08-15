from mpi4py import MPI
from oars.matrices import getTwoBlockSimilar
import json
import numpy as np
from time import time

comm = MPI.COMM_WORLD
i = comm.Get_rank()
mpi_size = comm.Get_size()
tgt_n = 32
n = 4 + 2*tgt_n
check_vartol = 0
log_file = 'logs1/distributed_logs_'+str(i)+'.json'
if n > (mpi_size - check_vartol):
    print("Problem size is larger than the number of processors")
    exit()
    

title = "SDP_Test"
gamma = 0.8
itrs = 7000
vartol = None #1e-5
shape = (2*tgt_n, 2*tgt_n)
Z, W = getTwoBlockSimilar(n)

# Initialize the resolvents and variables
if i == 0:
    t = time()
    print("Testing SDP")
    from oars.utils.proxs import psdCone, traceEqualityIndicator, traceHalfspaceIndicator, linearSubdiff
    from oars.matrices import getFull, getBlockMin, getMT
    from oars.pep import getConstraintMatrices    
    from oars.algorithms.distributed import subproblem, initialize

    Zp, Wp = getMT(tgt_n)
    Ko, K1, Ki, Kp = getConstraintMatrices(Zp, Wp, gamma=0.5)

    proxlist = [psdCone, traceEqualityIndicator, traceEqualityIndicator, linearSubdiff] + [traceHalfspaceIndicator for _ in Kp]
    data = [(2*tgt_n, 2*tgt_n), {'A':Ki, 'v':1}, {'A':K1, 'v':0}, -Ko] + Kp


    comms_data = initialize(comm, n, data, proxlist, W, Z)
    
    print("Node 0 running subproblem", flush=True)
        #print("Comms data 0", Comms_Data[0], flush=True)
    d = data[i]
    res = proxlist[i]
    x, log = subproblem(i, d, res, W, Z, comms_data, comm, gamma, itrs, vartol=vartol, verbose=True)

    
    x_i = np.zeros(x.shape)
    for k in range(1, n):
        comm.Recv(x_i, source=k, tag=0)
        x += x_i
    xbar = (1/n)*x
    print(xbar, flush=True)

    print(np.trace(Ko @ xbar), flush=True)
    print("Time", time() - t)
    # Save xbar to file
    np.save('logs/distributed_results_'+title+'.npy', xbar)

    # Save log to file
    if log is not None:
        timestamp = time()
        with open(log_file, 'w') as f:
            json.dump(log, f)

elif i < n:
    
    from oars.algorithms.distributed import subproblem
    data = comm.recv(source=0, tag=44)
    res = comm.recv(source=0, tag=17)
    comms = comm.recv(source=0, tag=33)

    # Run the subproblem
    #print(f"Node {i} running subproblem", flush=True)
    x, log = subproblem(i, data, res, W, Z, comms, comm, gamma, itrs, vartol=vartol, verbose=True)

    if log is not None:
        timestamp = time()
        with open(log_file, 'w') as f:
            json.dump(log, f)
    
    comm.Send(x, dest=0, tag=0)

elif i == n and vartol is not None:
    from oars.algorithms.distributed import evaluate
    evaluate(n, shape, comm, vartol, itrs=itrs)