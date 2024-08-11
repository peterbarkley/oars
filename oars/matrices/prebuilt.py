from numpy import zeros, ones, eye

def getMT(n):
    '''
    Get Malitsky-Tam values for Z and W

    Args:
        n (int): number of resolvents
        
    Returns:
        Z (ndarray): Z matrix n x n numpy array
        W (ndarray): W matrix n x n numpy array
    '''
    W = zeros((n,n))
    W[0,0] = 1
    W[0,1] = -1
    for r in range(1,n-1):
        W[r,r-1] = -1
        W[r,r] = 2
        W[r,r+1] = -1
    W[n-1,n-1] = 1
    W[n-1,n-2] = -1

    Z = W.copy()
    Z[n-1,0] -= 1
    Z[0,n-1] -= 1
    Z[0,0] += 1
    Z[n-1,n-1] += 1
    
    return Z, W

def getFull(n):
    '''
    Return Z, W for a fully connected graph with n nodes
    and weight 2 evenly distributed among all edges

    Args:
        n (int): number of resolvents

    Returns:
        Z (ndarray): n x n numpy array for the graph Laplacian of a fully connected weighted graph with n nodes
                     where the weights are evenly distributed among all edges and the weighted degree of each node is 2 
        W (ndarray): n x n numpy array which is the same as Z
    '''
    
    v = 2/(n-1)
    W = -v*ones((n,n))
    # Set diagonal of W to 2
    for i in range(n):
        W[i,i] = 2
    
    Z = W.copy()

    return Z, W


def getRyu(n):
    '''
    Get Tam's extension of the Ryu algorithm 
    values for Z and W

    Args:
        n (int): number of resolvents
    
    Returns:
        Z (ndarray): Z matrix n x n numpy array
        W (ndarray): W matrix n x n numpy array
    '''

    W = eye(n)
    W[n-1,:] = -ones(n) # Set last row/col to -1
    W[:,n-1] = -ones(n)
    W[n-1,n-1] = n-1
    W = 2/(n-1)*W

    # Set lower triangle of Z to 1
    
    v = 2/(n-1)
    Z = -v*ones((n,n))
    # Set diagonal of W to 2
    for i in range(n):
        Z[i,i] = 2
    return Z, W

