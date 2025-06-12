import pandas as pd
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from oars.matrices.core import getMfromWCholesky
from oars.pep import getReducedContractionOptGamma

def getIterationTime(t, l, Z, W, itrs=None):
    '''
    Get iteration time using critical path method
    
    Args:
        t (list): list of resolvent times for each agent
        l (ndarray): n x n array of communication times
        Z (ndarray): n x n array of resolvent multipliers
        W (ndarray): n x n array of consensus multipliers
        itrs (int, optional): number of iterations, 3n-1 by default

    Returns:
        cycle_length (float): cycle time
        s (ndarray): itrs x n array of start times for each agent in each iteration
        X (ndarray): n x n array of communication relationships
                     X[i,j] is 1 if Z[i,j] is nonzero, i>j
                     X[j,i] is 1 if W[i,j] is nonzero, i<j
    
    Examples:
        >>> from oars.matrices import getMT
        >>> t = [1, 2, 3]
        >>> l = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        >>> Z, W = getMT(3)
        >>> itr_t, s, X = getIterationTime(t, l, Z, W)
        >>> itr_t
        6.999999999996042
        >>> s
        array([[ 0.,  2.,  5.],
            [ 5.,  9., 12.],
            [12., 16., 19.],
            [19., 23., 26.],
            [26., 30., 33.],
            [33., 37., 40.],
            [40., 44., 47.],
            [47., 51., 54.]])
        >>> X
        array([[0., 1., 0.],
            [1., 0., 1.],
            [1., 1., 0.]])
    '''

    # Build communication matrix  
    n = len(t)
    X = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            if not np.isclose(Z[i,j],0.0,atol=1e-2):
                X[i,j] = 1
            if not np.isclose(W[i,j],0.0,atol=1e-2):
                X[j,i] = 1

    # Declare variables
    if itrs is None:
        itrs = 3*n-1
    M = itrs*(max(t) + np.max(l))*n
    s = cvx.Variable((itrs,n), nonneg=True) # start time for each agent in each iteration

    # Build constraints
    cons = [s[i,k] - s[i,j] >= (t[j] + l[j,k]) for i in range(itrs) for j in range(n-1) for k in range(j+1, n) if X[k,j] == 1]

    cons += [s[i+1, k] - s[i, j] >= (t[j] + l[j,k]) for i in range(itrs-1) for k in range(1,n) for j in range(k) if X[j,k] == 1]
    cons += [s[i+1, j] - s[i, k] >= (t[k] + l[k,j] + M)*X[j,k] - M for i in range(itrs-1) for k in range(1,n) for j in range(k)]

    # Build objective
    obj = cvx.Minimize(n**2*cvx.max(s[itrs-1,:]) + cvx.sum(s))

    # Solve problem
    prob = cvx.Problem(obj, cons)
    prob.solve()
    cycle_length = (s[itrs-1,n-1]-s[itrs-2,n-1]).value
    return cycle_length, s.value, X

def getGantt(t, l, Z, W, itrs=None):
    """
    Get Gantt chart for the parallel algorithm

    Args:
        t (list): list of resolvent times for each agent
        l (ndarray): n x n array of communication times
        Z (ndarray): n x n array of resolvent multipliers
        W (ndarray): n x n array of consensus multipliers
        title (str, optional): title for the chart
        itrs (int, optional): number of iterations, 2n by default

    Returns:
        fig (plotly figure): Gantt chart of the parallel algorithm

    Examples:
        >>> from oars.utils import getGantt
        >>> from oars.matrices import getMT
        >>> import numpy as np
        >>> t = [1, 2, 3]
        >>> l = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        >>> Z, W = getMT(3)
        >>> fig = getGantt(t, l, Z, W)
        >>> fig.show()
    """
    n = len(t)
    if itrs is None:
        itrs = 2*n
    _, s, _ = getIterationTime(t, l, Z, W, itrs=itrs)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(itrs): # Iterations
        data = {'Task': [f'Node {j+1}' for j in range(n)],
                'Start': [s[i,j] for j in range(n)],
                'End': [s[i,j] + t[j] for j in range(n)]}
        df = pd.DataFrame(data)
        ax.hlines(df.index[::-1], df['Start'], df['End'], color=f'C{i}', label=f'Iteration {i+1}', linewidth=20) # lines for tasks
    ax.set_yticks(df.index) # y ticks
    ax.set_yticklabels(df['Task'][::-1]) # y labels

    ax.set_xlabel('Time (s)')
    leg = ax.legend()
    for i in range(itrs):
        leg.get_lines()[i].set_linewidth(5)

    return fig

def getMetrics(Z, W, t=None, l=None, ls=None, mus=None, contraction_target=0.5):
    '''
    Return the iteration time, contraction factor, cycles to contraction target, 
    and total_time to contraction target for the Cholesky reduced algorithm design 
    using the optimal step size

    Args:
        Z (ndarray): n x n resolvent matrix
        W (ndarray): n x n consensus matrix
        t (list, optional): n compute times (default 1)
        l (ndarray, optional): n x n communication times (default 1)
        ls (list, optional): n Lipschitz constants for operators (default 2, last np.inf)
        mus (list, optional): n monotonicity constsants for operators (default 1, last 0)
        contraction_target (float, optional): target contraction factor (default 0.5)

    Returns:
        cycle (float): iteration time
        tau (float): contraction factor
        contraction_cycles (int): cycles to contraction_target
        total_time (float): total time to target

    Example:
        >>> from oars.matrices import getMT
        >>> Z, W = getMT(3)
        >>> getMetrics(Z, W)
        (3.999, 0.854, 5.0, 19.999)
    '''
    n = Z.shape[0]
    if t is None: t = np.ones(n)
    if l is None: l = np.ones((n,n))
    cycle, _, _ = getIterationTime(t, l, Z, W)
    M = getMfromWCholesky(W)
    if ls is None:
        ls = np.ones(n)*2
        ls[n-1] = np.inf
    if mus is None:
        mus = np.ones(n)
        mus[n-1] = 0
    tau, gam = getReducedContractionOptGamma(Z, M, ls, mus)
    contraction_cycles = np.ceil(np.log(contraction_target)/np.log(tau))
    total_time = cycle*contraction_cycles
    return cycle, tau, contraction_cycles, total_time
