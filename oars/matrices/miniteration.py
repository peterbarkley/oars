import pyomo.environ as pyo
import numpy as np
from oars.matrices.core import getBlockFixed, getMinSpectralDifference

def getMinIteration(n, objective=getMinSpectralDifference, **kwargs):
    """
    Get the minimum iteration time algorithm for a given objective function
    and optional keyword arguments

    Args:
        n (int): number of resolvents
        objective (function): function to minimize
        kwargs: additional keyword arguments for the algorithm

            - t (list): list of resolvent compute times
            - l (ndarray): n x n array of communication times
            - fixed_X (dict): dictionary of fixed communication relationships for Z
            - fixed_Y (dict): dictionary of fixed communication relationships for W
            - r (int): number of iterations to optimize over
            - minZ (int): the number of edges required for each resolvent in Z
            - minW (int): the number of edges required for each resolvent in W
            - Zedges (int): the minimum number of edges in Z as a whole
            - Wedges (int): the minimum number of edges in W as a whole
            - minfixed (bool, 'Z', or 'W'): whether to include the number of edges in the objective with weight weight
            - weight (float): the coefficient for the number of edges in the objective

    Returns:
        Z (ndarray): Z matrix n x n numpy array
        W (ndarray): W matrix n x n numpy array
        alpha (float): the proximal scaling parameter if eps != 0
    """
    Z_fixed, W_fixed = getMinFlow(n, **kwargs)

    return objective(n, fixed_Z=Z_fixed, fixed_W=W_fixed, **kwargs)

def setWarmstart(n, m, edges):
    Zf, _ = getBlockFixed(n, n//2)
    if n%2 == 1:
        odd_weight = (2-2/(n-1))/(n//2)
    for j,k in edges:
        if (k,j) in Zf:
            m.x[j,k] = 0
            m.wx[j,k] = 0
            m.y[j,k] = 0
        else:
            m.x[j,k] = 1
            if n%2 == 0:
                m.wx[j,k] = 4/n #Fix this
            elif k==n-1:
                m.wx[j,k] = 2/(n-1)
            else:
                m.wx[j,k] = odd_weight
            m.y[j,k] = 1
    flow_weight = (n-1)/(n-n//2)
    sec_flow_wt = flow_weight/(n//2-1)
    for k in range(n//2, n):
        m.fy[0,k] = flow_weight
        for j in range(1, n//2):
            m.fy[k,j] = sec_flow_wt

def getMinFlow(n, t=None, l=None, fixed_X={}, fixed_Y={}, r=None, minfixed=True, weight=None, minZ=2, minW=1, Wedges=None, Zedges=None, solver='gurobi', **kwargs):
    """
    Get the minimum iteration time algorithm using a flow formulation

    Args:
        n (int): number of resolvents
        t (list): list of resolvent compute times
        l (ndarray): n x n array of communication times
        fixed_X (dict): dictionary of fixed communication relationships for Z
        fixed_Y (dict): dictionary of fixed communication relationships for W
        r (int): number of iterations to optimize over
        minZ (int): the number of edges required for each resolvent in Z
        minW (int): the number of edges required for each resolvent in W
        Zedges (int): the minimum number of edges in Z as a whole
        Wedges (int): the minimum number of edges in W as a whole
        minfixed (bool, 'Z', or 'W'): whether to include the number of edges in the objective with weight weight
        weight (float): the coefficient for the number of edges in the objective

    Returns:
        Z_fixed (dict): dictionary of fixed communication relationships for Z
        W_fixed (dict): dictionary of fixed communication relationships for W
    """
    
    nodes = set(range(n))

    if 'timelimit' in kwargs:
        timelimit = kwargs['timelimit']
    else:
        timelimit=60
    local=False
    if 'verbose' in kwargs:
        verbose=kwargs['verbose']
        if verbose=='local':
            local = True
            verbose = False
    else:
        verbose=False

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))

    # Define the optimization problem
    if t is None: t = np.ones(n)
    if l is None: l = np.ones((n,n))
    tl = (t + l).T
    if r is None: r = 2*n
    tmax = np.max(t)
    tmin = np.min(t)
    if weight is None: weight=tmin/(2*n**2)
    mm = r*n*tmax
    mask = np.ones(l.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    q = r*(tmax+2*l[mask].min() + tmin)

    # Make x and y addressable in a dictionary by the edge
    e = {edge: i for i, edge in enumerate(edges)}

    # Add reverse edges
    for i, (j, k) in enumerate(edges):
        e[(k, j)] = i

    iters = pyo.Set(initialize=range(r))
    nodes = pyo.Set(initialize=range(n))

    # put n-1 into source node and -1 into sink nodes
    b = -1*np.ones(n)
    b[0] = n-1

    # Define model
    m = pyo.ConcreteModel()

    # Define variables
    m.x = pyo.Var(edges, domain=pyo.Binary)
    m.wx = pyo.Var(edges, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(edges, domain=pyo.Binary)
    m.fy = pyo.Var(nodes, nodes, domain=pyo.NonNegativeReals)
    m.s = pyo.Var(iters, nodes, domain=pyo.NonNegativeReals)
    m.v = pyo.Var(domain=pyo.NonNegativeReals)

    # Warmstart to 2-block design
    setWarmstart(n, m, edges)

    # Constraints
    def row_sum(mvar, k):
        return sum(mvar[j,k] for j in range(k)) + sum(mvar[k,j] for j in range(k+1, n))

    # Node/row constraints
    m.min_cycle_time = pyo.ConstraintList() 
    m.wx2 = pyo.ConstraintList()
    m.xentries = pyo.ConstraintList()
    m.yentries = pyo.ConstraintList()
    m.source = pyo.ConstraintList() 
    for k in range(n):
        # v is distance from min cycle time
        m.min_cycle_time.add(m.v >= m.s[r-1, k] - q)

        # wx sums to 2 in each column
        m.wx2.add(row_sum(m.wx, k)  == 2)

        # Z has at least 2 edges incident on each node
        m.xentries.add(row_sum(m.x, k) >= minZ)

        # W has at least 1 edge incident on each node
        m.yentries.add(row_sum(m.y, k) >= minW)

        # flow constraints
        m.source.add(sum(m.fy[k,j] for j in range(n) if j != k) == b[k] + sum(m.fy[j,k] for j in range(n) if j != k))
        
    # Set fixed values
    m.fixedX = pyo.ConstraintList()
    for idx,val in fixed_X.items():
        m.fixedX.add(m.x[idx] == val)

    m.fixedY = pyo.ConstraintList()
    for idx,val in fixed_Y.items():
        m.fixedY.add(m.y[idx] == val)

    # Edge constraints
    m.wxc = pyo.ConstraintList()
    m.yx = pyo.ConstraintList()
    m.xconnect = pyo.ConstraintList()
    m.sin = pyo.ConstraintList()
    m.sout = pyo.ConstraintList()
    m.flow = pyo.ConstraintList()
    for i, (j, k) in enumerate(edges):
        # wx is 0 if x is 0
        m.wxc.add(m.wx[j,k] <= 2*m.x[j,k])

        # y is less than or equal to x 
        # (so Z-W is positive semidefinite)
        m.yx.add(m.y[j,k] <= m.x[j,k])

        # ensure that the graph is connected in wx
        m.xconnect.add(m.wx[j,k] >= m.x[j,k]/(n-1))

        # within iteration, node k starts after node j finishes if edge j,k is selected for Z
        # b/t iterations, node k/j starts after node j/k finishes if edge j,k is selected for W
        for ii in range(r):
            m.sin.add(m.s[ii, k] >= m.s[ii, j] + (tl[j, k]+mm)*m.x[j,k]-mm)
            if ii < r-1:
                m.sout.add(m.s[ii+1, k] >= m.s[ii, j] + (tl[j, k]+mm)*m.y[j,k]-mm)
                m.sout.add(m.s[ii+1, j] >= m.s[ii, k] + (tl[k, j]+mm)*m.y[j,k]-mm)

        # bounds on the flow matrix (can only flow from node j to k if edge j,k is selected)
        m.flow.add(m.fy[j,k] <= (n-1)*m.y[j,k])
        m.flow.add(m.fy[k,j] <= (n-1)*m.y[j,k])
    if Zedges != None:
        m.zedges = pyo.Constraint(expr=sum(m.x[j,k] for j,k in edges) >= Zedges)
    if Wedges != None:
        m.zedges = pyo.Constraint(expr=sum(m.y[j,k] for j,k in edges) >= Wedges)
    objexpr = m.v
    if minfixed == True or minfixed == 'W':
        objexpr = objexpr - weight*sum(m.y[j,k] for j,k in edges)
    if minfixed == True or minfixed == 'Z':
        objexpr = objexpr - weight*sum(m.x[j,k] for j,k in edges)


    # Pyomo objective
    m.obj = pyo.Objective(expr=objexpr)

    # Solve with Pyomo
    solver = pyo.SolverFactory(solver)

    # Add time limit
    solver.options['TimeLimit'] = timelimit

    solver.solve(m, tee=verbose, warmstart=True)

    # Process the results
    Z_fixed = {}
    W_fixed = {}
    for j,k in edges:
        if m.x[j,k]() == 0:
            Z_fixed[(j,k)] = 0
        if m.y[j,k]()  == 0:
            W_fixed[(j,k)] = 0

    if local:
        print('Z edges', pyo.value(sum(m.x[j,k] for j,k in edges)))
        print('W edges', pyo.value(sum(m.y[j,k] for j,k in edges)))
        print('v', pyo.value(m.v))
    return Z_fixed, W_fixed

def getMinCore(n, t=None, l=None, fixed_X={}, fixed_Y={}, r=None, minfixed=True, weight=None, minZ=2, minW=1, Wedges=None, Zedges=None, solver='gurobi', **kwargs):
    """
    Get the minimum iteration time algorithm using a core formulation

    Args:
        n (int): number of resolvents
        t (list): list of resolvent compute times
        l (ndarray): n x n array of communication times
        fixed_X (dict): dictionary of fixed communication relationships for Z
        fixed_Y (dict): dictionary of fixed communication relationships for W
        r (int): number of iterations to optimize over
        minZ (int): the number of edges required for each resolvent in Z
        minW (int): the number of edges required for each resolvent in W
        Zedges (int): the minimum number of edges in Z as a whole
        Wedges (int): the minimum number of edges in W as a whole
        minfixed (bool, 'Z', or 'W'): whether to include the number of edges in the objective with weight weight
        weight (float): the coefficient for the number of edges in the objective

    Returns:
        
    """
    
    nodes = set(range(n))

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))

    # Define the optimization problem
    if t is None: t = np.ones(n)
    if l is None: l = np.ones((n,n))
    tl = (t + l).T
    if r is None: r = 2*n
    tmax = np.max(t)
    tmin = np.min(t)
    if weight is None: weight=tmin/(2*n**2)
    mm = r*n*tmax
    mask = np.ones(l.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    q = r*(tmax+2*l[mask].min() + tmin)

    # Make x and y addressable in a dictionary by the edge
    e = {edge: i for i, edge in enumerate(edges)}

    # Add reverse edges
    for i, (j, k) in enumerate(edges):
        e[(k, j)] = i

    iters = pyo.Set(initialize=range(r))
    nodes = pyo.Set(initialize=range(n))

    # put n-1 into source node and -1 into sink nodes
    b = -1*np.ones(n)
    b[0] = n-1

    # Define model
    m = pyo.ConcreteModel()

    # Define variables
    m.x = pyo.Var(edges, domain=pyo.Binary)
    m.wx = pyo.Var(edges, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(edges, domain=pyo.Binary)
    m.fy = pyo.Var(nodes, nodes, domain=pyo.NonNegativeReals)
    m.s = pyo.Var(iters, nodes, domain=pyo.NonNegativeReals)
    m.v = pyo.Var(domain=pyo.NonNegativeReals)

    # Warmstart to 2-block design
    setWarmstart(n, m, edges)

    # Constraints
    def row_sum(mvar, k):
        return sum(mvar[j,k] for j in range(k)) + sum(mvar[k,j] for j in range(k+1, n))

    # Node/row constraints
    m.min_cycle_time = pyo.ConstraintList() 
    m.wx2 = pyo.ConstraintList()
    m.xentries = pyo.ConstraintList()
    m.yentries = pyo.ConstraintList()
    m.source = pyo.ConstraintList() 
    for k in range(n):
        # v is distance from min cycle time
        m.min_cycle_time.add(m.v >= m.s[r-1, k] - q)

        # wx sums to 2 in each column
        m.wx2.add(row_sum(m.wx, k)  == 2)

        # Z has at least 2 edges incident on each node
        m.xentries.add(row_sum(m.x, k) >= minZ)

        # W has at least 1 edge incident on each node
        m.yentries.add(row_sum(m.y, k) >= minW)

        # flow constraints
        m.source.add(sum(m.fy[k,j] for j in range(n) if j != k) == b[k] + sum(m.fy[j,k] for j in range(n) if j != k))
        
    # Set fixed values
    m.fixedX = pyo.ConstraintList()
    for idx,val in fixed_X.items():
        m.fixedX.add(m.x[idx] == val)

    m.fixedY = pyo.ConstraintList()
    for idx,val in fixed_Y.items():
        m.fixedY.add(m.y[idx] == val)

    # Edge constraints
    m.wxc = pyo.ConstraintList()
    m.yx = pyo.ConstraintList()
    m.xconnect = pyo.ConstraintList()
    m.sin = pyo.ConstraintList()
    m.sout = pyo.ConstraintList()
    m.flow = pyo.ConstraintList()
    for i, (j, k) in enumerate(edges):
        # wx is 0 if x is 0
        m.wxc.add(m.wx[j,k] <= 2*m.x[j,k])

        # y is less than or equal to x 
        # (so Z-W is positive semidefinite)
        m.yx.add(m.y[j,k] <= m.x[j,k])

        # ensure that the graph is connected in wx
        m.xconnect.add(m.wx[j,k] >= m.x[j,k]/(n-1))

        # within iteration, node k starts after node j finishes if edge j,k is selected for Z
        # b/t iterations, node k/j starts after node j/k finishes if edge j,k is selected for W
        for ii in range(r):
            m.sin.add(m.s[ii, k] >= m.s[ii, j] + (tl[j, k]+mm)*m.x[j,k]-mm)
            if ii < r-1:
                m.sout.add(m.s[ii+1, k] >= m.s[ii, j] + (tl[j, k]+mm)*m.y[j,k]-mm)
                m.sout.add(m.s[ii+1, j] >= m.s[ii, k] + (tl[k, j]+mm)*m.y[j,k]-mm)

        # bounds on the flow matrix (can only flow from node j to k if edge j,k is selected)
        m.flow.add(m.fy[j,k] <= (n-1)*m.y[j,k])
        m.flow.add(m.fy[k,j] <= (n-1)*m.y[j,k])
    if Zedges != None:
        m.zedges = pyo.Constraint(expr=sum(m.x[j,k] for j,k in edges) >= Zedges)
    if Wedges != None:
        m.zedges = pyo.Constraint(expr=sum(m.y[j,k] for j,k in edges) >= Wedges)
    objexpr = m.v
    if minfixed == True or minfixed == 'W':
        objexpr = objexpr - weight*sum(m.y[j,k] for j,k in edges)
    if minfixed == True or minfixed == 'Z':
        objexpr = objexpr - weight*sum(m.x[j,k] for j,k in edges)


    return m, m.x, m.wx, m.y, m.fy, m.s, m.v, objexpr, edges