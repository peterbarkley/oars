
import sys
from mpi4py import MPI
import numpy as np
from time import time

def distributed_three_block_solve(n, data, resolvents, warmstartprimal, warmstartdual=None, w_own=0, w_other=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None, verbose=False):

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
    z = np.array(4/n)
    if w_other is None:
        w_other = np.array(4/n)
    else:
        w_other = np.array(w_other)
    
    icomm.bcast((n, z, w_own, w_other, gamma, alpha, itrs, vartol), root=MPI.ROOT)
    
    for i in range(n//2):
        icomm.send(data[i], dest=i) #z, w, gamma, alpha, v0
        icomm.send(data[i+n//2], dest=i) #resolvent
        icomm.send(resolvents[i], dest=i) #resolvent
        icomm.send(resolvents[i+n//2], dest=i) #resolvent
        icomm.send(v0[i], dest=i)
        icomm.send(v0[i+n//2], dest=i)
    if verbose:print('Data sent to workers')
        
    x_bar = icomm.recv(source=0)
    results = icomm.recv(source=0)
    # for i in range(n//2):
    #     result = icomm.recv(source=i)
    #     results.append(result)

    # if verbose:print('x_bar', x_bar)
    icomm.Disconnect()
    return x_bar, results

def main_child():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker(icomm)
    icomm.Disconnect()

def xbar_diff(my_x, my_y, sum_x, sum_y, sum_z, n, comm):
    xbar = (sum_x + sum_y + sum_z)/n
    diff = np.linalg.norm(xbar - my_x, ord='fro')**2 + np.linalg.norm(xbar - my_y, ord='fro')**2
    sum_diff = np.array(0.0)
    comm.Allreduce([diff, MPI.DOUBLE], [sum_diff, MPI.DOUBLE], op=MPI.SUM)
    return sum_diff**0.5

def zero_diff(v, comm):
    sum_v = np.zeros(v[0].shape)
    v_total = sum(v)
    comm.Allreduce([v_total, MPI.DOUBLE], [sum_v, MPI.DOUBLE], op=MPI.SUM)
    return np.linalg.norm(sum_v, ord='fro')

def worker(icomm):
    myrank = icomm.Get_rank()
    print('Worker', myrank, 'started')

    # Build intracommunicator
    comm = MPI.COMM_WORLD

    # Receive data from parent
    n, z, w_own, w_other, gamma, alpha, itrs, vartol = icomm.bcast((), root=0)
    first_data = icomm.recv(source=0)
    second_data = icomm.recv(source=0)
    first_resolvent = icomm.recv(source=0)
    second_resolvent = icomm.recv(source=0)
    res = [first_resolvent(first_data), second_resolvent(second_data)]
    first_v0 = icomm.recv(source=0)
    second_v0 = icomm.recv(source=0)
    v = [first_v0, second_v0]

    shape = first_v0.shape
    sum_x = np.zeros(shape)
    sum_y = np.zeros(shape)
    sum_z = np.zeros(shape)
    old_delta = np.array(-1.0)
    delta = np.array(0.0)
    itr_period = itrs // 10
    check_period = 1
    t = time()
    if myrank < n//4:
        group_1 = comm.group.Incl([i for i in range(n//4)])
        comm_group_1 = comm.Create_group(group_1)
        my_x = res[0].prox(v[0], alpha)
        comm_group_1.Allreduce([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
    else:
        group_2 = comm.group.Incl([i for i in range(n//4, n//2)])
        comm_group_2 = comm.Create_group(group_2)

    comm.Bcast(sum_x, root=0)
    
    my_y = res[1].prox(v[1] + z*sum_x, alpha)
    comm.Allreduce([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)

    if myrank < n//4:
        update_0 = gamma*(2*my_x - w_other*sum_y - w_own*sum_x)
        v[0] = v[0] - update_0

    for itr in range(itrs):

        # First block
        old_sum_x = sum_x.copy()
        if myrank < n//4:
            old_my_x = my_x.copy()
            my_x = res[0].prox(v[0], alpha)
            comm_group_1.Allreduce([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        else:
            my_z = res[0].prox(v[0] + z*sum_y, alpha)
            comm_group_2.Allreduce([my_z, MPI.DOUBLE], [sum_z, MPI.DOUBLE], op=MPI.SUM)

        comm.Bcast(sum_x, root=0)
        comm.Bcast(sum_z, root=n//4)
        # if myrank == 0:
        #     print('Iteration', itr, 'sum x', np.linalg.norm(sum_x), 'sum z', np.linalg.norm(sum_z),flush=True)
        # Second block
        update_1 = gamma*(2*my_y - w_other*old_sum_x - w_own*sum_y - w_other*sum_z)
        v[1] = v[1] - update_1
        old_sum_y = sum_y.copy()
        my_y = res[1].prox(v[1]+z*sum_x, alpha)
        comm.Allreduce([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)

        # Update v
        if myrank < n//4:
            old_v0 = v[0].copy()
            update_0 = gamma*(2*my_x - w_other*sum_y - w_own*sum_x)
        else:
            update_0 = gamma*(2*my_z - w_other*old_sum_y - w_own*sum_z)
        v[0] = v[0] - update_0

        if itr % check_period == 0:
            v_sq = np.linalg.norm(update_0)**2 + np.linalg.norm(update_1)**2
            comm.Allreduce([v_sq, MPI.DOUBLE], [delta, MPI.DOUBLE], op=MPI.SUM)
            delta_rt = np.sqrt(delta)
            change = max(np.abs(delta_rt - old_delta), vartol, 1e-8)
            # print('Worker', myrank, 'Iteration', itr, 'Delta', delta, 'Change', change, 'Vartol', vartol, 'Check period', check_period, 'old_delta', old_delta, 'v_sq', v_sq)
            check_period = max(1, min(itr_period, int(delta_rt/change)))
            if myrank < n//4:
                sum_diff = xbar_diff(my_x, my_y, sum_x, sum_y, sum_z, n, comm)
                u0 = old_v0 - old_my_x # v0 + (L-I) x
                u1 = v[1] - my_y + z*sum_x # v1 + (L-I) x
                sum_zero_diff = zero_diff([u0, u1], comm)
            else:
                sum_diff = xbar_diff(my_z, my_y, sum_z, sum_y, sum_x, n, comm)
                u0 = v[0] - my_z + z*sum_y # v0 + (L-I) x
                u1 = v[1] - my_y + z*sum_x # v1 + (L-I) x
                sum_zero_diff = zero_diff([u0, u1], comm)

            if delta_rt < vartol:
                if myrank == 0:
                    print('Converged at iteration', itr, 'Delta v', delta_rt, 'Sum diff', sum_diff, 'Zero diff', sum_zero_diff, flush=True)
                break
            old_delta = delta_rt.copy()
        
        if myrank == 0 and itr % itr_period == 0:
            timedelta = (time()-t)
            print('Iteration', itr, 'Time', timedelta, 'Delta v', old_delta, 'Sum Diff', sum_diff, 'Zero diff', sum_zero_diff, flush=True)

    if myrank == 0:
        print('Worker', myrank, 'finished, time', time()-t, flush=True)

    if myrank < n//4:
        x = my_x
    else:
        x = my_z
    log = [{'first_x': x, 'second_x': my_y, 'first_v0': v[0], 'second_v0': v[1]}]
    if hasattr(res[0], 'log'):
        log += res[0].log
        log += res[1].log

    logs = comm.gather(log, root=0)
    if myrank == 0:
        xbar = (sum_x + sum_y + sum_z)/n
        icomm.send(xbar, dest=0)
        icomm.send(logs, dest=0)

    # comm.Disconnect()

if __name__ == '__main__':
    if 'child' in sys.argv:
        main_child()


