import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from ..tools import *
from tqdm.notebook import tqdm


class Taylor:
    def __init__(self):
        pass

    def low_up_taylor(self, theta_obs, SM, gamma):
        """Computes the truncation bounds for the approximate gaussian distribution of the statistic used in the post-selection inference method from Taylor & Tibshirani '18.

        Parameters
        ----------
        theta_obs : 
            solution of the l1-penalized likelihood model.
        SM : vector of -1 or +1
            observed vector of signs, i.e. SM = sign(theta_obs[M]).
        M : vector of integers
            selected support.
        X : 2 dimensional matrix
            design matrix.
        lamb : float
            regularization parameter for the l1-penalty.
        ind : None or integer
            integer between 1 and |M| corresponding to the coordinate of the parameter vector that we want to focus on for the test. If None, the linear statistic of the vector of parameter is eta.T @ theta where eta is the observed vector of signs. 

        Returns
        -------
        vlow : lower bound of the truncation interval.
        vup : upper bound of the truncation interval.

        Ref
        ---
        Taylor J, Tibshirani R. 
        Post-Selection Inference for ℓ1-Penalized Likelihood Models. 
        2017
        Can J Stat. 2018 Mar;46(1):41-61. 
        doi: 10.1002/cjs.11313.
        """
        bhat = theta_obs[self.M]
        matXtrue = self.X[:,self.M]
        pihat = sigmoid(self.X @ theta_obs)
        MM = np.linalg.inv(matXtrue.T @ np.diag(pihat*(1-pihat)) @ matXtrue)
        b1 = -self.lamb * np.diag(SM) @ MM @ SM
        A1 = -np.diag(SM)
        bbar = bhat + self.lamb* MM @ SM
        c = MM @ gamma * (gamma.T @ MM @ gamma)**(-1)
        r = (np.eye(len(self.M))-np.dot(c.reshape(-1,1),gamma.reshape(1,-1))) @ bbar
        vup = np.Inf
        vlow = -np.Inf
        v0 = np.Inf
        Ac = A1@c
        Ar = A1@r
        for j in range(len(b1)):
            if (Ac)[j]==0:
                v0 = min(v0,b1[j]-Ar[j])
            elif Ac[j]<0:
                vlow = max(vlow,(b1[j]-Ar[j])/Ac[j])
            else:
                vup = min(vup,(b1[j]-Ar[j])/Ac[j])
        return vlow,vup

    

    def pval_taylor(self, states, gamma=None, show_distributions=False, thetanull=None, nsamples=100000):
        """Computes the P-values using the post-selection inference method from Taylor & Tibshirani '18.

        Parameters
        ----------
        probas : list of float
            conditional probabilities under a prescribed alternative associated to each vectors in selection event (corresponding to each entry of the input 'states').
        states : list of vectors in {0,1}^n
            all the vector of the hypercube belonging to the selection event.
        X : 2 dimensional matrix
            design matrix.
        lamb : float
            regularization parameter for the l1-penalty.
        M : array of integers
            selected support.
        show_distributions : bool
            If True, plot the 

        Returns
        -------
        lspvalstay: samples of p-values using the post-selection method from Taylor & Tibshirani '18.

        Ref
        ---
        Taylor J, Tibshirani R. 
        Post-Selection Inference for ℓ1-Penalized Likelihood Models. 
        2017
        Can J Stat. 2018 Mar;46(1):41-61. 
        doi: 10.1002/cjs.11313.
        """
        n,p = (self.X).shape
        lspvals_taylor = np.zeros((1,len(states)))
        lssamplesalt = []
        matXtrue = self.X[:,self.M]
        signull = sigmoid(self.X @ thetanull)
        Wnull = np.linalg.inv(matXtrue.T @ np.diag(signull * (1-signull)) @  matXtrue)
        for ind in range(1):
            for idx in tqdm(range(len(states))):
                y = np.array(states[idx])
                model = LogisticRegression(C = 1/self.lamb, penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(self.X, y)
                theta_obs = model.coef_[0]
                bhat = theta_obs[self.M]
                SM = np.sign(bhat)
                pihat = sigmoid(self.X @ theta_obs)
                MM = np.linalg.inv(matXtrue.T @ np.diag(pihat*(1-pihat)) @ matXtrue)
                bbar = bhat + self.lamb* MM @ SM
                lssamplesalt.append(bbar[ind])

                if gamma is None:
                    gamma = SM / np.linalg.norm(SM)
                
                if thetanull is None:
                    thetanull = np.zeros(p)

                vlow, vup = self.low_up_taylor(theta_obs,SM,gamma)

                samples = np.random.normal(thetanull[self.M[ind]],np.sqrt(gamma.T @ Wnull @ gamma),nsamples)
                selected_idxs = np.where((samples>=vlow) & (samples<=vup))[0]
                samplesnull = samples[selected_idxs]
                if len(samplesnull)>=10:
                    pval = 2*min(np.mean(samplesnull<=bbar[ind]),np.mean(samplesnull>=bbar[ind]))
                else:
                    pval = 0
                lspvals_taylor[ind,idx]=pval    
        lspvalstay = np.mean(lspvals_taylor, axis=0)
        if show_distributions:
            a = plt.hist(samplesnull,density=True,alpha=0.2,label='Heuristic conditional null distribution for a specific vector of signs')
            a = plt.hist(lssamplesalt,density=True,alpha=0.2, label='Observed conditional distribution')
            plt.legend()
        return lspvalstay
