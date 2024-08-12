from mpi4py import MPI
from oars import solveMT, solve
from oars.algorithms.distributed import subproblem, initialize, evaluate
from oars.matrices import getBlockMin, getMinResist
from time import time
from proxs import *

comm = MPI.COMM_WORLD
i = comm.Get_rank()
mpi_size = comm.Get_size()
n = 4

if n > mpi_size - 1:
    print("Problem size is larger than the number of processors")
    exit()
    

title = "Quad_Test"
gamma = 0.8
itrs = 1000
vartol = 1e-5
shape = ()
Z, W = getBlockMin(n, n//2, objective=getMinResist)

# Initialize the resolvents and variables
if i == 0:
    vals = np.array([0, 1, 3, 40]) #np.array([2, -3, -7, -8])
    proxs = [quadprox]*n

    comms_data = initialize(comm, n, vals, proxs, W, Z)
    
    print("Node 0 running subproblem", flush=True)
        #print("Comms data 0", Comms_Data[0], flush=True)
    t = time()
    data = vals[i]
    res = proxs[i]
    x, log = subproblem(i, data, res, W, Z, comms_data, comm, gamma, itrs, vartol=vartol, verbose=True)
    print("Time", time() - t)
    
    x_i = np.zeros(x.shape)
    for k in range(1, n):
        comm.Recv(x_i, source=k, tag=0)
        x += x_i
    xbar = (1/n)*x
    print(xbar, flush=True)

    # Save xbar to file
    np.save('distributed_results_'+title+'.npy', xbar)

    # Save log to file
    if log is not None:
        timestamp = time()
        with open('distributed_logs'+str(i)+'_'+title+'.json', 'w') as f:
            json.dump(log, f)

elif i < n:
    data = comm.recv(source=0, tag=44)
    res = comm.recv(source=0, tag=17)
    comms = comm.recv(source=0, tag=33)

    # Run the subproblem
    #print(f"Node {i} running subproblem", flush=True)
    x, log = subproblem(i, data, res, W, Z, comms, comm, gamma, itrs, vartol=vartol, verbose=True)

    if log is not None:
        timestamp = time()
        with open('distributed_logs'+str(i)+'_'+title+'.json', 'w') as f:
            json.dump(log, f)
    
    comm.Send(x, dest=0, tag=0)

elif i == n and vartol is not None:
    evaluate(n, shape, comm, vartol, itrs=itrs)