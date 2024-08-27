
import sys
from mpi4py import MPI
import numpy as np

def distributed_block_solve_restart(n, data, resolvents, warmstartprimal, warmstartdual=None, w_own=0, w_other=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None, verbose=False):
    # assert len(sys.argv) <= 2

    # if 'server' in sys.argv:
    #     main_server(MPI.COMM_WORLD)
    # elif 'client' in sys.argv:
    #     main_client(MPI.COMM_WORLD)
    # elif 'child'  in sys.argv:
    #     main_child()

    assert(MPI.COMM_WORLD.Get_size() == 1)
    assert(n % 2 == 0)
    icomm = MPI.COMM_WORLD.Spawn(command=sys.executable,
                                 args=[__file__, 'child'],
                                 maxprocs=n//2)

    # Send data to workers
    print('Sending data to workers')
    shape = warmstartprimal.shape
    v0 = np.zeros(shape)
    z = np.array(4/n)
    if w_other is None:
        w_other = np.array(4/n)
    else:
        w_other = np.array(w_other)
    w_own = np.array(w_own)
    gamma = np.array(gamma)
    alpha = np.array(alpha)
    itrs = np.array(itrs)
    vartol = np.array(vartol)
    
    icomm.bcast((n, z, w_own, w_other, gamma, alpha, itrs, vartol), root=MPI.ROOT)
    # data = comm.scatter(data, root=MPI.ROOT)
    for i in range(n//2):
        icomm.send(data[i], dest=i) #z, w, gamma, alpha, v0
        icomm.send(data[i+n//2], dest=i) #resolvent
        icomm.send(resolvents[i], dest=i) #resolvent
        icomm.send(resolvents[i+n//2], dest=i) #resolvent
        icomm.send(warmstartprimal, dest=i)
    print('Data sent to workers')
        
    x_bar = np.zeros(shape)
    x_bar = icomm.recv(source=0)
    results = []
    for i in range(n//2):
        # x = icomm.recv(source=i)
        # y = icomm.recv(source=i)
        result = icomm.recv(source=i)
        results.append(result)
        # if verbose:
        #     print('Received x from worker', i, x)
        #     print('Received y from worker', i, y)
    #     x_bar += x + y
    # x_bar /= n
    if verbose:print('x_bar', x_bar)    

    icomm.Disconnect()
    return x_bar, results

def main_child_restart():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker_restart(icomm)
    icomm.Disconnect()


def worker_restart(icomm):
    myrank = icomm.Get_rank()
    print('Worker', myrank, 'started')

    # Build intracommunicator
    comm = MPI.COMM_WORLD

    # Receive data from parent
    n = 0
    z = np.array(0.0)
    w_own = np.array(0.0)
    w_other = np.array(0.0)
    gamma = np.array(0.0)
    alpha = np.array(0.0)
    itrs = np.array(0)
    vartol = np.array(0.0)
    alg_data = (n, z, w_own, w_other, gamma, alpha, itrs, vartol)
    n, z, w_own, w_other, gamma, alpha, itrs, vartol = icomm.bcast(alg_data, root=0)
    first_data = icomm.recv(source=0)
    second_data = icomm.recv(source=0)
    first_resolvent = icomm.recv(source=0)
    second_resolvent = icomm.recv(source=0)
    res = [first_resolvent(first_data), second_resolvent(second_data)]
    v0 = icomm.recv(source=0)
    v = [v0, -v0]

    # print('Worker', myrank, 'received data:', z, w_own, w_other, gamma, alpha, first_data, first_resolvent, second_data, second_resolvent, v0)
    # log = []
    shape = v0.shape
    sum_x = np.zeros(shape)
    sum_y = np.zeros(shape)
    old_delta = np.array(-1.0)
    delta = np.array(0.0)
    itr_period = itrs // 10
    check_period = 1
    for itr in range(itrs):

        # First block
        my_x = res[0].prox(v[0], alpha)
        comm.Allreduce([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)

        # Second block
        my_y = res[1].prox(v[1]+z*sum_x, alpha)
        comm.Allreduce([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)

        # Update v
        update_0 = gamma*(2*my_x - w_other*sum_y - w_own*sum_x)
        update_1 = gamma*(2*my_y - w_other*sum_x - w_own*sum_y)
        v[0] = v[0] - update_0
        v[1] = v[1] - update_1

        # Check convergence
        if itr % check_period == 0:
            v_sq = np.linalg.norm(update_0) + np.linalg.norm(update_1)
            comm.Allreduce([v_sq, MPI.DOUBLE], [delta, MPI.DOUBLE], op=MPI.SUM)
            change = max(np.abs(delta - old_delta), vartol, 1e-8)
            check_period = max(1, min(itr_period, int(delta/change)))
            if delta < vartol:
                if myrank == 0:
                    print('Converged at iteration', itr, 'Delta', delta)
                break
            old_delta = delta.copy()
        
        if myrank == 0 and itr % itr_period == 0:
            print('Iteration', itr, 'Delta', delta)

        # log.append((itr, myrank, my_x, my_y, sum_x, sum_y, update_0, update_1))

    xbar = (sum_x + sum_y)/n
    # xbar = np.zeros(shape)
    # comm.Allreduce([sumxy, MPI.DOUBLE], [xbar, MPI.DOUBLE], op=MPI.SUM)
    # xbar /= n
    v = [xbar.copy(), -xbar.copy()]
    
    old_delta = np.array(-1.0)
    delta = np.array(0.0)
    itr_period = itrs // 10
    check_period = 1
    if myrank == 0:
        print('Worker', myrank, 'finished first phase')
    for itr in range(itrs):

        # First block
        my_x = res[0].prox(v[0], alpha)
        comm.Allreduce([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)

        # Second block
        my_y = res[1].prox(v[1]+z*sum_x, alpha)
        comm.Allreduce([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)

        # Update v
        update_0 = gamma*(2*my_x - w_other*sum_y - w_own*sum_x)
        update_1 = gamma*(2*my_y - w_other*sum_x - w_own*sum_y)
        v[0] = v[0] - update_0
        v[1] = v[1] - update_1

        # Check convergence
        if itr % check_period == 0:
            v_sq = np.linalg.norm(update_0) + np.linalg.norm(update_1)
            comm.Allreduce([v_sq, MPI.DOUBLE], [delta, MPI.DOUBLE], op=MPI.SUM)
            change = max(np.abs(delta - old_delta), vartol, 1e-8)
            check_period = max(1, min(itr_period, int(delta/change)))
            if delta < vartol:
                if myrank == 0:
                    print('Converged at iteration', itr, 'Delta', delta)
                break
            old_delta = delta.copy()
        
        if myrank == 0 and itr % itr_period == 0:
            print('Iteration', itr, 'Delta', delta)


    if myrank == 0:
        print('Worker', myrank, 'finished second phase')
        xbar = (sum_x + sum_y)/n
        icomm.send(xbar, dest=0)
    log = []
    if hasattr(res[0], 'log'):
        log = res[0].log
        log += res[1].log
    icomm.send(log, dest=0)
    #     source = dest = 0
    # else:
    #     source = dest = MPI.PROC_NULL
    # n =  N.array(0, 'i')
    # icomm.Recv([n, MPI.INT], source=source)
    # pi = comp_pi(n, comm=MPI.COMM_WORLD, root=0)
    # pi = N.array(pi, 'd')
    # icomm.Send([pi, MPI.DOUBLE], dest=dest)

if __name__ == '__main__':
    if 'child' in sys.argv:
        main_child_restart()