from scipy.sparse import csr_array

def getNZIndicesLower(Z, tol=1e-5):
    """
    Get the indices in the lower diagonal which are not within tolerance of zero

    Returns:
        list: list of non-zero indices
    """
    n = Z.shape[0]
    Zidx = []
    for i in range(n):
        row_idx = []
        for j in range(i):
            if abs(Z[i,j]) > tol:
                row_idx.append(j)
        Zidx.append(row_idx)
    return Zidx
    
def getNZIndices(W, tol=1e-5):
    """
    Get the indices which are not within tolerance of zero

    Returns:
        list: list of non-zero indices
    """
    n = W.shape[0]
    Widx = []
    for i in range(n):
        row_idx = []
        for j in range(n):
            if abs(W[i,j]) > tol:
                row_idx.append(j)
        Widx.append(row_idx)
    return Widx


def serialSparseAlgorithm(n, data, resolvents, W, Z, itrs=1000, gamma=0.9, alpha=1.0, vartol=None, checkperiod=10, verbose=False):
    """
    Run the serial algorithm with sparse matrices

    Args:
        n (int): the number of resolvents
        data (list): list containing the problem data for each resolvent
        resolvents (list): list of :math:`n` resolvent classes
        W (csr_matrix): size (n, n) csr_matrix for the :math:`W` matrix
        Z (csr_matrix): size (n, n) csr_matrix for the :math:`Z` matrix
        itrs (int, optional): the number of iterations
        gamma (float, optional): parameter in :math:`v^{k+1} = v^k - \\gamma W x^k`
        alpha (float, optional): the resolvent step size in :math:`x^{k+1} = J_{\\alpha F^i}(y^k)`
        vartol (float, optional): is the variable tolerance
        checkperiod (int, optional): the period to check for convergence
        verbose (bool, optional): True for verbose output

    Returns:
        x (ndarray): the solution
        results (list): list of dictionaries with the results for each resolvent
    """
    # Initialize the resolvents and variables
    all_x = []
    for i in range(n):
        resolvents[i] = resolvents[i](data[i])
        if i == 0:
            m = resolvents[0].shape
        x = csr_array(m, dtype=float)
        all_x.append(x)
    all_v = [csr_array(m, dtype=float) for _ in range(n)]
    wx = [csr_array(m, dtype=float) for _ in range(n)]
    # Get non-zero indices by row
    Zidx = getNZIndicesLower(Z)
    Widx = getNZIndices(W)

    convergence = ConvergenceChecker(all_v, vartol=vartol, checkperiod=checkperiod)
    # Run the algorithm
    if verbose:
        xresults = [all_x.copy()]
        vresults = [all_v.copy()]
        print('Starting Serial Algorithm')
        start_time = time()
    verbose_itr = itrs//10

    for itr in range(itrs):
        if itr % verbose_itr == 0:
            print(f'Iteration {itr+1}')

        for i in range(n):
            y = all_v[i] - sum(Z[i,j]*all_x[j] for j in Zidx[i])
            all_x[i] = resolvents[i].prox(y, alpha)

        for i in range(n):     
            wx[i] = sum(W[i,j]*all_x[j] for j in Widx[i])       
            all_v[i] = all_v[i] - gamma*wx[i]

        if verbose and itr % verbose_itr == 0:

            for i in range(n):
                print('x', i, all_x[i])
                print('v', i, all_v[i])
            xresults.append(all_x.copy())
            vresults.append(all_v.copy())
        if convergence.check(all_v, verbose=verbose):
            print('Converged in value, iteration', itr+1)
            break
        
    if verbose:
        print('Serial Algorithm Loop Time:', time()-start_time)
    x = sum(all_x)/n

    results = []
    return x, results

class ConvergenceChecker():

    def __init__(self, all_v, vartol=None, checkperiod=10):
        self.vartol = vartol
        self.checkperiod = checkperiod
        self.checkperiodcounter = 0
        self.old_v = all_v.copy()

    def check(self, all_v, verbose=False):
        """
        Check for convergence

        Args:
            all_v (list): list of the v variables
            verbose (bool): True for verbose output

        Returns:
            bool: True if converged, False otherwise
        """
        if self.vartol is None:
            return False
        self.checkperiodcounter += 1
        if self.checkperiodcounter % self.checkperiod == 0:
            self.checkperiodcounter = 0
            
            if verbose:
                print('Checking for convergence')
            delta = sum(np.linalg.norm(all_v[i] - self.old_v[i], 'fro') for i in range(len(all_v)))
            if verbose:
                print('Delta:', delta)
            if delta < self.vartol:
                return True