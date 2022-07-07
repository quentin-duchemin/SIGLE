import numpy as np
from sklearn.linear_model import LinearRegression

class LogitRegressionContinuous(LinearRegression):

    def fit(self, x, p):
        p = np.asarray(p)
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)


def binary_encoding(yt):
    """Computes the integer encoded by the vector of bits yt.
    
    Parameters
    ----------
    yt : list of 0 or 1
        vector of bits.
    
    Returns
    -------
    res : integer encoding yt.
    """
    res = 0
    for el in yt[::-1]:
        res *= 2
        res += el
    return res

def binary_encoding_inv(binary,n):
    """Computes the inverse of the function 'binary_encoding'.
    
    Parameters
    ----------
    binary : integer
    n : integer
        dimension expected for the output.
        
    Returns
    -------
    y : vector of bits encoding 'binary'.
    """
    y=np.zeros(n)
    div = 2
    for i in range(n):
        if binary%2==1:
            y[i]=1
            binary -= 1
        binary /= 2
    return y

def sigmoid1(u):
    """Derivative of the sigmoid function."""
    return (np.exp(-u)/(1+np.exp(-u))**2)

def sigmoid(u):
    """Sigmoid function."""
    return (1/(1+np.exp(-u)))

def equalset(A,B):
    """Checks if two sets of integers are equal.
    
    Parameters
    ----------
    A, B : lists or sets of integers.
    
    Returns
    -------
    A boolean which is True if and only if the sets A and B are equal.
    """
    res = True
    for a in A:
        if a not in B:
            res = False
    return res

def logtemp(x):
    return (0.4/np.log(x))

def compute_proba(z,bern):
    """Computes the probability of the vector of bits 'z' when the expected value of the response vector is given by 'bern'

    Parameters
    ----------
    z : list of bits.
    bern : vector of floats.
    """
    n = len(bern)
    return np.exp( np.sum( z*np.log(bern) + (np.ones(n)-z)*np.log(np.ones(n)-bern)))
    