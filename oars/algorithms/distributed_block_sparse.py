
import sys
from mpi4py import MPI
import numpy as np
from time import time
from datetime import datetime
from oars.matrices import getTwoBlockSimilar

def distributed_block_sparse_solve(n, data, resolvents, warmstartprimal, warmstartdual=None, W=None, Z=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=1e-8, logging=False, verbose=False):
    '''
    Solve the 2-Block resolvent splitting algorithm in parallel using MPI

    Args:
        n (int): Number of resolvents
        data (list): List of data for each resolvent
        resolvents (list): List of resolvents
        warmstartprimal (ndarray): Warm start for the primal variables
        warmstartdual (ndarray): Warm start for the dual variables
        W (ndarray): Weight matrix
        Z (ndarray): Weight matrix
        itrs (int): Number of iterations
        gamma (float): Step size
        alpha (float): Proximal parameter
        vartol (float): Tolerance for convergence
        verbose (bool): Print verbose output

    Returns:
        x_bar (ndarray): The average of the primal variables
        results (list): List of dictionaries with the results for each resolvent

    Examples:
        >>> mpiexec -n 1 python distributed_block_example.py
        2024-08-27 11:39:50.517313 Worker 1 started
        2024-08-27 11:39:50.517313 Worker 0 started
        2024-08-27 11:39:50.517313 Iteration 0 time 0.0 Delta v 45.304966615151585 Sum diff 16.93553955443995 Sum zero diff 21.500000000000004
        Converged at iteration 10 Delta v 4.530496464705242e-09 Sum diff 1.693553905459803e-09 Sum zero diff 2.1499886315723415e-09
        2024-08-27 11:39:50.532951 Worker 0 finished, time 0.01563882827758789
        10.999999999462496
        [[{'first_x': 10.999999998899996, 'second_x': 10.999999999074998, 'first_v0': 21.999999999779995, 'second_v0': -2.999999999970001}], [{'first_x': 10.999999998949997, 'second_x': 11.000000000924997, 'first_v0': 20.999999999789992, 'second_v0': -39.9999999996}]]
    '''
    assert(MPI.COMM_WORLD.Get_size() == 1)
    assert(n % 2 == 0)
    icomm = MPI.COMM_WORLD.Spawn(command=sys.executable,
                                 args=[__file__, 'child'],
                                 maxprocs=n//2)

    # Set v0
    shape = warmstartprimal.shape
    v0 = [warmstartprimal]*(n//2) + [-warmstartprimal]*(n//2)
    if warmstartdual is not None:
        v0 = [v0[i] + warmstartdual[i] for i in range(n)]

    # Send data to workers
    if Z is None:
        Zo, Wo = getTwoBlockSimilar(n)
        Z = Zo
    if W is None:
        W = Wo
    
    icomm.bcast((n, Z, W, gamma, alpha, itrs, logging, vartol), root=MPI.ROOT)
    
    for i in range(n//2):
        icomm.send((data[i], data[i+n//2], resolvents[i], resolvents[i+n//2], v0[i], v0[i+n//2]), dest=i)

    if verbose:print(datetime.now(), 'Data sent to workers', flush=True)
        
    x_bar = icomm.recv(source=0)
    results = icomm.recv(source=0)

    icomm.Disconnect()
    return x_bar, results

def main_child():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker(icomm)
    icomm.Disconnect()

def xbar_diff(my_x, my_y, sum_x, sum_y, n, comm):
    xbar = (sum_x + sum_y)/n
    diff = np.linalg.norm(xbar - my_x)**2 + np.linalg.norm(xbar - my_y)**2
    sum_diff = np.array(0.0)
    comm.Allreduce([diff, MPI.DOUBLE], [sum_diff, MPI.DOUBLE], op=MPI.SUM)
    return sum_diff**0.5

def zero_diff(v, comm):
    sum_v = np.zeros(v[0].shape)
    v_total = sum(v)
    comm.Allreduce([v_total, MPI.DOUBLE], [sum_v, MPI.DOUBLE], op=MPI.SUM)
    return np.linalg.norm(sum_v)

def buildComms(icomm, myrank, Z, W, zerotol=1e-4):
    # my_coms is a list of 2 communicators
    # my_comms[0] is the communicator for the first required reduction
    # it gives \\sum_{j=1}^n -Z_{n+i,j}x_j when used with a reduction
    # it includes myrank as the root node and the ranks for the the nonzero entries from 0->n in row myrank+n
    n = Z.shape[0]//2
    my_comms = []
    Ni = [[] for _ in range(n)]
    Nj = [[] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if r != c and not np.isclose(Z[n+r][c],0.0, atol=zerotol):
                Ni[r].append(c)
                Nj[c].append(r)

    igroups = [icomm.group.Incl([i] + Ni[i]) for i in range(n)]
    jgroups = [icomm.group.Incl([i] + Nj[i]) for i in range(n)]

    leftcomms = [icomm.Create_group(group) for group in igroups]
    rightcomms = [icomm.Create_group(group) for group in jgroups]

    # myleftdeps = leftcomms[Nj[myrank]]
    myleftdeps = [(leftcomms[j], -Z[n+j, myrank]) for j in Nj[myrank]]
    # myrightdeps = rightcomms[Ni[myrank]]
    myrightdeps = [(rightcomms[j], -Z[j, myrank+n]) for j in Ni[myrank]]

    return leftcomms[myrank], rightcomms[myrank], myleftdeps, myrightdeps


def worker(icomm):
    myrank = icomm.Get_rank()

    # Build intracommunicator
    comm = MPI.COMM_WORLD

    # Receive data from parent
    n, Z, W, gamma, alpha, itrs, logging, vartol = icomm.bcast((), root=0)
    first_data, second_data, first_resolvent, second_resolvent, first_v0, second_v0 = icomm.recv(source=0)

    my_left_comm, my_right_comm, my_left_deps, my_right_deps = buildComms(comm, myrank, Z, W)

    v = [first_v0, second_v0]

    res = [first_resolvent(first_data), second_resolvent(second_data)]
    if logging:
        res[0].logging = True
        res[1].logging = True
    shape = first_v0.shape
    sum_x = np.zeros(shape)
    sum_y = np.zeros(shape)
    old_delta = np.array(-1.0)
    delta = np.array(0.0)
    itr_period = itrs // 10
    check_period = itr_period
    t = time()
    myrankshift = n//2 + myrank
    if myrank == 0 or myrank == n//2 - 1:
        print(datetime.now(), 'Worker', myrank, 'started', flush=True)
    for itr in range(itrs):

        # First block
        my_x = res[0].prox(v[0], alpha)
        # comm.Allreduce([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        for comm, wt in my_left_deps:
            comm.Ireduce([my_x*wt, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        req = my_left_comm.Ireduce([-my_x*Z[myrankshift, myrank], MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        # Wait for req
        req.Wait()

        # Second block
        my_y = res[1].prox(v[1]+z*sum_x, alpha)
        # comm.Allreduce([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
        for comm, wt in my_right_deps:
            comm.Ireduce([my_y*wt, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
        req = my_left_comm.Ireduce([-my_y*Z[myrank, myrankshift], MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
        update_1 = gamma*(2*my_y - sum_x)
        v[1] = v[1] - update_1

        # Wait for sum_y
        req.Wait()

        update_0 = gamma*(2*my_x - sum_y)
        v[0] = v[0] - update_0
        

        # if itr % itr_period == 0:
        #     v_sq = np.linalg.norm(update_0)**2 + np.linalg.norm(update_1)**2
        #     comm.Allreduce([v_sq, MPI.DOUBLE], [delta, MPI.DOUBLE], op=MPI.SUM)
        #     delta_rt = np.sqrt(delta)
        #     # change = max(np.abs(delta_rt - old_delta), vartol, 1e-8)
        #     # check_period = max(1, min(itr_period, int(delta_rt/change)))
        #     sum_diff = xbar_diff(my_x, my_y, sum_x, sum_y, n, comm)
        #     u0 = v[0] - my_x
        #     u1 = v[1] + my_y
        #     sum_zero_diff = zero_diff([u0,u1], comm)
        #     if delta_rt < vartol:
        #         if myrank == 0:
        #             print('Converged at iteration', itr, 'Delta v', delta_rt, 'Sum diff', sum_diff, 'Sum zero diff', sum_zero_diff, flush=True)
        #         break
        #     old_delta = delta_rt.copy()
        
        # if myrank == 0 and itr % itr_period == 0:
        #     timedelta = (time()-t)
        #     print(datetime.now(), 'Iteration', itr, 'time', timedelta, 'Delta v', old_delta, 'Sum diff', sum_diff, 'Sum zero diff', sum_zero_diff, flush=True)

    if myrank == 0:
        print(datetime.now(), 'Worker', myrank, 'finished, time', time()-t, flush=True)
    result = [{'first_x': my_x, 'second_x': my_y, 'first_v0': v[0], 'second_v0': v[1]}]

    if logging:
        import json
        for i in [0, 1]:
            log = res[i].log
            idx = myrank + i*(n//2)
            with open(str(idx) + '_dist_log.json', 'w') as f:
                json.dump(log, f)

    results = comm.gather(result, root=0)
    xbar = np.zeros(my_x.shape)
    comm.Reduce([my_y + my_x, MPI.DOUBLE], [xbar, MPI.DOUBLE], op=MPI.SUM)
    if myrank == 0:
        xbar = xbar/n
        icomm.send(xbar, dest=0)
        icomm.send(results, dest=0)
    comm.Barrier()
    for comm in my_left_deps:
        comm.Disconnect()
    for comm in my_right_deps:
        comm.Disconnect()
    my_left_comm.Disconnect()
    my_right_comm.Disconnect()
    comm.Disconnect()

if __name__ == '__main__':
    if 'child' in sys.argv:
        main_child()