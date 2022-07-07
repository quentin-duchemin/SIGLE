import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from inverse_map import inverse_map, train_network

from PSILOGIT.Methods import Methods
from .Sampling import Sampling
from .tools import *

class PSILOGIT(Methods, Sampling):
    def __init__(self, truetheta, lamb, X=None, n=None, yobs=None, M=None, SM=None, sampling_algorithm=None):
        super(Methods, self).__init__()
        self.truetheta = truetheta
        p = len(truetheta)
        if X is None:
            np.random.seed(1)
            assert (n != None), 'If the design is not specified, you need to provide its number of lines: n.'
            X = np.random.normal(0,1,(n,p))
        self.X = X
        self.sig = sigmoid(self.X @ self.truetheta)
        if M is None:
            if yobs is None:
                yobs = np.random.rand(n) <= self.sig
            self.yobs = yobs
            if np.shape(lamb) == ():
                self.lamb = lamb
                model = LogisticRegression(C = 1/self.lamb, penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(self.X, self.yobs)
            else:
                model = LogisticRegressionCV(Cs = lamb, cv=2, penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(self.X, self.yobs)
                self.lamb = 1/model.C_[0]
            self.theta_obs = model.coef_[0]
            self.M = np.where( np.abs(self.theta_obs) > 1e-5)[0]
            self.SM = np.sign(self.theta_obs[M])
        else:
            self.M = M
            self.SM = SM

        Sampling.__init__(self, sampling_algorithm)


        