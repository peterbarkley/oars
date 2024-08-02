from PEPit import PEP, null_point, Constraint
from PEPit.primitive_steps import proximal_step
from PEPit.operators import MonotoneOperator, LipschitzStronglyMonotoneOperator
from numpy import array
# TODO import what you need from the pipeline
# from PEPit.functions import ``THE FUNCTION CLASSES YOU NEED``
# from PEPit.operators import ``THE OPERATOR CLASSES YOU NEED``
# from primitive_steps import ``THE PRIMITIVE STEPS YOU NEED``


def wc_frugal_resolvent_splitting(L, W, lipschitz_values, mu_values, alpha=1, gamma=0.5, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the the monotone inclusion problem

    .. math:: \\mathrm{Find}\\, x:\\, 0 \\in \\sum_{i=1}^{n} A_i(x),

    where :math:`A_i` is :math:`l_i`-Lipschitz and :math:`\\mu_i`-strongly monotone for :math:`i < n`, and :math:`A_n` is maximally monotone. We denote by :math:`J_{\\alpha A_i}` the resolvent of :math:`\\alpha A_i`. We denote the lifted vector operator :math:`A` as :math:`A = [A_1, \\dots, A_n]`, and use lifted :math:`x` and :math:`v` as :math:`x = [x_1, \\dots, x_n]` and :math:`v = [v_1, \\dots, v_n]`. We denote by :math:`L` and :math:`W` the design matrices of the lifted operator :math:`A`, and by :math:`l` and :math:`\\mu` the vectors of Lipschitz and strong monotonicity constants of the lifted operator :math:`A`. :math:`L` is assumed to strictly lower diagonal and sum to :math:`n`, and :math:`W` is assumed to be PSD such that :math:`W 1 = 0`.

    This code computes a worst-case guarantee for any frugal resolvent splitting with design matrices :math:`L, W`. This can include the Malitsky-Tam [1], Ryu Three Operator Splitting [2], Douglas-Rachford [3], or block splitting algorithms [4].
    That is, given two lifted initial points (each of which sums to 0):math:`v^{(0)}_t` and :math:`v^{(1)}_t`,
    this code computes the smallest possible :math:`\\tau(L, W, l, \\mu, \\alpha, \\gamma)`
    (a.k.a. "contraction factor") such that the guarantee

    .. math:: \\|v^{(0)}_{t+1} - v^{(1)}_{t+1}\\|^2 \\leqslant \\tau(L, W, l, \\mu, \\alpha, \\gamma) \\|v^{(0)}_{t} - v^{(1)}_{t}\\|^2,

    is valid, where :math:`v^{(0)}_{t+1}` and :math:`v^{(1)}_{t+1}` are obtained after one iteration of the frugal resolvent splitting from respectively :math:`v^{(0)}_{t}` and :math:`v^{(1)}_{t}`.

    In short, for given values of :math:`L`, :math:`W`, :math:`l`, :math:`\\mu`, :math:`\\alpha` and :math:`\\gamma`, the contraction factor :math:`\\tau(L, \\mu, \\alpha, \\theta)` is computed as the worst-case value of
    :math:`\\|v^{(0)}_{t+1} - v^{(1)}_{t+1}\\|^2` when :math:`\\|v^{(0)}_{t} - v^{(1)}_{t}\\|^2 \\leqslant 1`.

    **Algorithm**: One iteration of the parameterized frugal resolvent splitting is described as follows,
    for :math:`t \\in \\{ 0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & J_{\\alpha A} (L x_{t+1} + v_t),\\\\
                v_{t+1} & = & v_t - \\gamma W x_{t+1}.
            \\end{eqnarray}

    **References**:
    `[1] Y. Malitsky, M. Tam (2023). Resolvent splitting for sums of monotone operators
    with minimal lifting. Mathematical Programming 201(1-2):231–262. <https://arxiv.org/pdf/2108.02897.pdf>`_

    `[2] E. Ryu (2020). Uniqueness of drs as the 2 operator resolvent-splitting and
    impossibility of 3 operator resolvent-splitting. Mathematical Programming 182(1-
    2):233–273. <https://arxiv.org/pdf/1802.07534>`_

    `[3] W. Moursi, L. Vandenberghe (2019). Douglas–Rachford Splitting for the Sum of a Lipschitz Continuous and a Strongly Monotone Operator. Journal of Optimization Theory and Applications 183, 179–198. <https://arxiv.org/pdf/1805.09396.pdf>`_

    `[4] R. Bassett, P. Barkley (2024). 
    Optimal Design of Resolvent Splitting Algorithms. arxiv:2407.16159
    Conference or journal (Acronym of conference or journal).
    <https://arxiv.org/pdf/2407.16159.pdf>`_

    Args:
        L (ndarray): description of arg1.
        W (ndarray): description of arg2.
        lipschitz_values (array-like): description of arg3.
        mu_values (array-like): description of arg4.
        alpha (float): description of arg5.
        gamma (float): description of arg6.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value

    Example:
        >>> import numpy as np

        >>> pepit_tau = wc_frugal_resolvent_splitting(L=np.array([[0,0],[2,0]]), arg2=value2, arg3=value3, wrapper="cvxpy", solver=None, verbose=1)
        ``OUTPUT MESSAGE``

    """

    # Instantiate PEP
    problem = PEP()

    # Declare monotone operators
    operators = [problem.declare_function(LipschitzStronglyMonotoneOperator, L=l, mu=mu) for l, mu in zip(lipschitz_values, mu_values)]
    #operators.append(problem.declare_function(MonotoneOperator))

    # Then define the starting points v0 and v1
    n = W.shape[0]
    v0 = [problem.set_initial_point() for _ in range(n)]
    v1 = [problem.set_initial_point() for _ in range(n)]
    
    # Set the initial constraint that is the distance between v0 and v1
    problem.set_initial_condition(sum((v0[i] - v1[i]) ** 2 for i in range(n)) <= 1)

    # Constraint on the lifted starting points so each sums to 0
    v0constraint = Constraint(expression=sum((v0[i] for i in range(n)), start=null_point)**2, equality_or_inequality="equality")
    v1constraint = Constraint(expression=sum((v1[i] for i in range(n)), start=null_point)**2, equality_or_inequality="equality")
    problem.set_initial_condition(v0constraint)
    problem.set_initial_condition(v1constraint)

    # Define the step for each element of the lifted vector    
    def resolvent(i, x, v, L, alpha):
        Lx = sum((L[i, j]*x[j] for j in range(i)), start=null_point)
        x, _, _ = proximal_step(v[i] + Lx, operators[i], alpha)
        return x

    x0 = []
    x1 = []
    for i in range(n):
        x0.append(resolvent(i, x0, v0, L, alpha))
        x1.append(resolvent(i, x1, v1, L, alpha))

    z0 = []
    z1 = []
    for i in range(n):
        z0.append(v0[i] - gamma*W[i,:]@x0)
        z1.append(v1[i] - gamma*W[i,:]@x1)

    
    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric(sum((z0[i] - z1[i]) ** 2 for i in range(n)))
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of parameterized frugal resolvent splitting` ***')
        print('\tPEPit guarantee:\t ||v_(t+1)^0 - v_(t+1)^1||^2 <= {:.6} ||v_(t)^0 - v_(t)^1||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau




if __name__ == "__main__":
    L = array([[0,0],[2,0]])
    W = array([[1,-1],[-1,1]])
    lipschitz_values = [2, 1000]
    mu_values = [1, 0]
    pepit_tau = wc_frugal_resolvent_splitting(L, W, lipschitz_values, mu_values)

    # Comparison for 4 operators across MT, Fully Connected, and optimized 2-Block
    # with and without optimized step sizes
    n = 4
    lipschitz_values = [2, 2, 2, 2]
    mu_values = [1, 1, 1, 1]
    L_MT = array([[0,0,0,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [1,0,1,0]])
    W_MT = array([[1,-1,0,0],
                  [-1,2,-1,0],
                  [0,-1,2,-1],
                  [0,0,-1,1]])
    L_full = array([[0,0,0,0],
                    [2/3,0,0,0],
                    [2/3,2/3,0,0],
                    [2/3,2/3,2/3,0]])
    W_full = array([[2, -2/3, -2/3, -2/3],
                    [-2/3, 2, -2/3, -2/3],
                    [-2/3, -2/3, 2, -2/3],
                    [-2/3, -2/3, -2/3, 2]])
    L_block = array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,0,0],
                     [1,1,0,0]])
    W_block = array([[ 1.348, -0.357, -0.511, -0.479],
                     [-0.357,  1.348, -0.511, -0.479],
                     [-0.511, -0.511,  1.529, -0.507],
                     [-0.479, -0.479, -0.507,  1.466]])
    tau_MT = wc_frugal_resolvent_splitting(L_MT, W_MT, lipschitz_values, mu_values, gamma=0.5, verbose=-1)
    tau_MTo = wc_frugal_resolvent_splitting(L_MT, W_MT, lipschitz_values, mu_values, gamma=0.074, verbose=-1)
    print('MT\t', tau_MT, tau_MTo)
    tau_f = wc_frugal_resolvent_splitting(L_full, W_full, lipschitz_values, mu_values, gamma=0.5, verbose=-1)
    tau_fo = wc_frugal_resolvent_splitting(L_full, W_full, lipschitz_values, mu_values, gamma=1, verbose=-1)
    print('Full\t', tau_f, tau_fo)
    tau_b = wc_frugal_resolvent_splitting(L_block, W_block, lipschitz_values, mu_values, gamma=0.5, verbose=-1)
    tau_bo = wc_frugal_resolvent_splitting(L_block, W_block, lipschitz_values, mu_values, gamma=1, verbose=-1)
    print('Block\t', tau_b, tau_bo)

    
