import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from .inverse_map import inverse_map, train_network

from PSILOGIT.Methods import Methods
from .Sampling import Sampling
from .tools import *

class PSILOGIT(Methods, Sampling):
    """Class allowing to conduct post-selection inference procedures in the logistic model with l1 penalty.
    """
    def __init__(self, regularization=1, truetheta=None, X=None, n=None, p=None, yobs=None, M=None, SM=None, sampling_algorithm=None, seed=1):
        super(Methods, self).__init__()
        np.random.seed(seed)
        if X is None:
            assert (n != None), 'If the design is not specified, you need to provide its number of lines: n.'
            assert (p != None), 'If the design is not specified, you need to provide its number of columns: p.'
            X = np.random.normal(0,1,(n,p))
        self.X = X
        n,p = X.shape
        if M is None:
            self.truetheta = truetheta
            self.sig = sigmoid(self.X @ self.truetheta)
            yobs = np.random.rand(n) <= self.sig
            self.yobs = yobs
            if np.shape(regularization) == ():
                self.lamb = regularization
                model = LogisticRegression(C = 1/self.lamb, penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(self.X, self.yobs)
            else:
                list_regu = [1./lregu for lregu in regularization]
                model = LogisticRegressionCV(Cs = list_regu, cv=2, penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(self.X, self.yobs)
                self.lamb = 1/model.C_[0]
            self.theta_obs = model.coef_[0]
            self.M = np.where( np.abs(self.theta_obs) > 1e-5)[0]
            self.SM = np.sign(self.theta_obs[M])
        else:
            self.M = M
            self.SM = SM

        Sampling.__init__(self, sampling_algorithm)


        