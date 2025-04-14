

class quadGrad():
    """
    gradient of the function f(x) = 0.5 x^T Q x - P x
    """
    def __init__(self, Q, P):
        self.Q = Q
        self.P = P
        self.shape = P.shape

    def grad(self, y):
        return self.Q@y - self.P