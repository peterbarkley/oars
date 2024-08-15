
import sys
from mpi4py import MPI
import numpy as np

def distributed_block_solve(n, data, resolvents, warmstartprimal, warmstartdual=None, w_own=0, w_other=None, itrs=1001, gamma=0.9, alpha=1.0, vartol=None, verbose=False):
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
    icomm.Bcast([z, MPI.INT], root=MPI.ROOT)
    icomm.Bcast([w_own, MPI.INT], root=MPI.ROOT)
    icomm.Bcast([w_other, MPI.INT], root=MPI.ROOT)
    icomm.Bcast([gamma, MPI.DOUBLE], root=MPI.ROOT)
    icomm.Bcast([alpha, MPI.DOUBLE], root=MPI.ROOT)
    icomm.Bcast([itrs, MPI.INT], root=MPI.ROOT)
    # data = comm.scatter(data, root=MPI.ROOT)
    for i in range(n//2):
        icomm.send(data[i], dest=i) #z, w, gamma, alpha, v0
        icomm.send(data[i+n//2], dest=i) #resolvent
        icomm.send(resolvents[i], dest=i) #resolvent
        icomm.send(resolvents[i+n//2], dest=i) #resolvent
        icomm.send(warmstartprimal, dest=i)
    print('Data sent to workers')

    itr_period = itrs // 10
    for itr in range(itrs):

        update_0 = icomm.recv(source=0)
        update_1 = icomm.recv(source=0)
        delta = np.linalg.norm(update_0) + np.linalg.norm(update_1)
        if itr % itr_period == 0: print('Iteration', itr, 'Delta', delta)
        
    x_bar = np.zeros(shape)
    results = []
    for i in range(n//2):
        x = icomm.recv(source=i)
        y = icomm.recv(source=i)
        result = icomm.recv(source=i)
        results.append(result)
        print('Received x from worker', i, x)
        print('Received y from worker', i, y)
        x_bar += x + y
    x_bar /= n
    print('x_bar', x_bar)
    icomm.Disconnect()
    return x_bar, results

def main_child():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker(icomm)
    icomm.Disconnect()


def worker(icomm):
    myrank = icomm.Get_rank()
    print('Worker', myrank, 'started')

    # Build intracommunicator
    comm = MPI.COMM_WORLD

    # Receive data from parent
    z = np.array(0.0)
    w_own = np.array(0.0)
    w_other = np.array(0.0)
    gamma = np.array(0.0)
    alpha = np.array(0.0)
    itrs = np.array(0)
    icomm.Bcast([z, MPI.INT], root=0)
    icomm.Bcast([w_own, MPI.INT], root=0)
    icomm.Bcast([w_other, MPI.DOUBLE], root=0)
    icomm.Bcast([gamma, MPI.DOUBLE], root=0)
    icomm.Bcast([alpha, MPI.DOUBLE], root=0)
    icomm.Bcast([itrs, MPI.INT])
    # data = None
    # data = icomm.scatter(data, root=0)
    first_data = icomm.recv(source=0)
    second_data = icomm.recv(source=0)
    data = [first_data, second_data]
    first_resolvent = icomm.recv(source=0)
    second_resolvent = icomm.recv(source=0)
    res = [first_resolvent(first_data), second_resolvent(second_data)]
    v0 = icomm.recv(source=0)
    v = [v0, -v0]

    # print('Worker', myrank, 'received data:', z, w_own, w_other, gamma, alpha, first_data, first_resolvent, second_data, second_resolvent, v0)
    # log = []
    shape = v0.shape
    # my_x = np.zeros(shape, 'd')
    # my_y = np.zeros(shape, 'd')
    sum_x = np.zeros(shape)
    sum_y = np.zeros(shape)
    # allreduce_request_x = comm.Allreduce_init([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
    # allreduce_request_y = comm.Allreduce_init([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
    for itr in range(itrs):

        # First block
        my_x = res[0].prox(v[0], alpha)
        # sum_x = np.zeros(my_x.shape, 'd')
        # my_x = np.array(my_x, 'd')
        comm.Allreduce([my_x, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        # allreduce_request_x.Start()
        # comm.Barrier()
        my_y = res[1].prox(v[1]+z*sum_x, alpha)
        # sum_y = np.zeros(my_y.shape, 'd')
        # my_y = np.array(my_y, 'd')
        comm.Allreduce([my_y, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
        # allreduce_request_y.Start()
        # comm.Barrier()


        # Update v
        update_0 = gamma*(2*my_x - w_other*sum_y - w_own*sum_x)
        update_1 = gamma*(2*my_y - w_other*sum_x - w_own*sum_y)
        v[0] = v[0] - update_0
        v[1] = v[1] - update_1

        if myrank == 0:
            icomm.send(update_0, dest=0)
            icomm.send(update_1, dest=0)

        # log.append((itr, myrank, my_x, my_y, sum_x, sum_y, update_0, update_1))
    icomm.send(my_x, dest=0)
    icomm.send(my_y, dest=0)
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
        main_child()