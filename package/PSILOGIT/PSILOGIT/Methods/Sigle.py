import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import scipy
import scipy as sc
import scipy.stats
from tqdm.notebook import tqdm
from ..tools import *
from .FiguresSigle import FiguresSigle

class Sigle(FiguresSigle):
    
    def __init__(self):
        pass   

    def params_saturated(self, bern, states):
        """Computes the probability of the vector of bits 'z' when the expected value of the response vector is given by 'bern'

        Parameters
        ----------
        bern : vector of floats
            expected value of the response vector.
        matXtrue : 2 dimension matrix
            design matrix corresponding to the restriction of the orginal matrix of features to the columns with integers in the selected support.
        lsy : list of vectors of bits
            lsy should contain states sampled from the uniform distribution on the selection event.

        Returns
        -------
        tildeGN_12 : 2 dimensional matrix
            matrix  (\bar G_N(\theta^*))^{-1/2} 
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        """
        n = len(bern)
        matXtrue = self.X[:,self.M]
        normalization = 0
        barpi = np.zeros(n)
        if self.sampling_algorithm == 'SA':
            for y in states:
                proba = compute_proba(y,bern)
                barpi += y * proba
                normalization += proba
            barpi /= normalization
        else:
            for y in tqdm(states):
                barpi += y
            barpi /= len(states)
        rho = matXtrue.T @ barpi.T
        tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
        usvd,s,vt = np.linalg.svd(tildeGN)
        tildeGN12 = usvd @ np.diag(np.sqrt(s)) @ vt
        tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt
        return tildeGN_12, barpi

    def compute_theta_bar(self, barpi, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}):
        matXtrue = self.X[:,self.M]
        model = LogitRegressionContinuous()
        model.fit(matXtrue, barpi)
        tildetheta = model.coef_

        gaps = []
        if np.max(np.abs(sigmoid(matXtrue @ tildetheta) -  barpi))>1e-3:
            L = (1/4) * np.linalg.norm(matXtrue.T @  matXtrue, ord=2) * np.sqrt( np.sum( ((np.sum(np.abs(matXtrue),axis=1))**2)))
            def grad(the):
                return (matXtrue.T @ np.diag(sigmoid1(matXtrue@the)) @ matXtrue @ matXtrue.T @ (sigmoid(matXtrue @ the)-barpi))
            count = 0
            gap = np.max(np.abs(matXtrue.T @ (sigmoid(matXtrue@tildetheta) - barpi)))
            lr = grad_descent['lr']
            while (count<grad_descent['max_ite'] and gap>1e-3):
                tildetheta = tildetheta - (lr / L) * grad(tildetheta)
                gap = np.max(np.abs(matXtrue.T @ (sigmoid(matXtrue@tildetheta) - barpi)))
                count += 1
                gaps.append(gap)
        return (tildetheta, gaps)
    
    def upper_bound_condition_CCLT(self, states, barpi, tildeGN_12):
        n = self.X.shape[0]
        matXtrue = X[:,M]
        coarse_upper_bound, upper_bound = 0, 0
        barcov = np.zeros((n,n))
        for y in states:
            barcov += (y-barpi) @ (y-barpi).T / len(states)
        for i in range(n):
            coarse_upper_bound += np.abs(1-2*barpi[i]) * np.sqrt( np.linalg.norm(matXtrue[:i,:].T @ barcov[:i,:i] @ matXtrue[:i,:]) )
            upper_bound += np.sqrt(barpi[i]*(1-barpi[i])) * np.abs(1-2*barpi[i]) * np.sqrt( np.matrix.trace(tildeGN_12 @ matXtrue[:i,:].T @ barcov[:i,:i] @ matXtrue[:i,:] @ tildeGN_12 ) )
        coarse_upper_bound /= np.max(np.linalg.norm(matXtrue, axis=0)) * np.sqrt(n) * np.sqrt(len(self.M))
        upper_bound /= n * np.sqrt(len(self.M))
        return coarse_upper_bound, upper_bound

    def pval_SIGLE(self, states, barpi, net=None, use_net_MLE=False, l2_regularization=10, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}, calibrated_from_samples=False, statesnull=None, signull=None):
        """Computes the P-values using the post-selection inference method SIGLE (both in the saturated and the selected model).

        Parameters
        ----------
        states : list of vectors in {0,1}^n
            all the vector of the hypercube belonging to the selection event.
        probas : list of float
            conditional probabilities under a prescribed alternative associated to each vectors in selection event (corresponding to each entry of the input 'states').
        X : 2 dimensional matrix
            design matrix.
        M : array of integers
            selected support.
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        net: neural network
            neural network trained to compute \Psi = \Xi^{-1}. Stated otherwise, given some \rho \in \mathds R^s (where s=|M|), the network should output the unique (if it exists) vector \theta \in \mathds R^s such that \rho = X[:,M].T \sigma(X[:,M] @ theta). 

        Returns
        -------
        lspvals_selec: samples of p-values using SIGLE in the selected model.
        lspvals_sat: samples of p-values using SIGLE in the saturated model.


        Note
        ----
        If \rho \in \mathds R^s (where s=|M|) can be written as \rho = X[:,M].T @ y where y \in \{0,1\}^n, then net(\rho) is the unconditional unpenalized MLE using the design X[:,M] and the response variable y.
        """
        n,p = (self.X).shape
        matXtrue = self.X[:,self.M]
        if net is not None:
            rho = matXtrue.T @ barpi.T
            tildetheta = net(torch.from_numpy(rho.T).float())
            tildetheta = tildetheta.detach().numpy()
        else:
            tildetheta, gaps = self.compute_theta_bar(barpi, grad_descent=grad_descent)

        tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
        usvd,s,vt = np.linalg.svd(tildeGN)
        tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt
        GNtilde = matXtrue.T @ np.diag(sigmoid1(matXtrue @ tildetheta)) @ matXtrue
        VN = tildeGN_12 @ GNtilde

        lspvals_selec = []
        lspvals_sat = []

        lsstat_sat, lsstatnull_sat = [], []
        lsstat_selec, lsstatnull_selec = [], []
        for i in tqdm(range(len(states))):
            y = np.array(states[i])
            # selected
            if use_net_MLE:
                rho = matXtrue.T @ y.T
                theta = net(torch.from_numpy(rho.T).float())
                theta = theta.detach().numpy()
            else:
                model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
                model.fit(matXtrue, y)
                theta = model.coef_[0]
            stat = np.linalg.norm( VN @ (theta - tildetheta))**2
            if not(calibrated_from_samples):
                df = len(self.M)
                lspvals_selec.append(1-scipy.stats.chi2.cdf(stat, df))
            else:
                lsstat_selec.append(stat)
            # saturated
            stat = np.linalg.norm( tildeGN_12 @ matXtrue.T @ (y-barpi))**2
            if not(calibrated_from_samples):
                df = len(self.M)
                lspvals_sat.append(1-scipy.stats.chi2.cdf(stat, df))
            else:
                lsstat_sat.append(stat)
        if calibrated_from_samples:
            for i in tqdm(range(len(statesnull))):
                y = np.array(statesnull[i])
                # selected
                if use_net_MLE:
                    rho = matXtrue.T @ y.T
                    theta = net(torch.from_numpy(rho.T).float())
                    theta = theta.detach().numpy()
                else:
                    model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
                    model.fit(matXtrue, y)
                    theta = model.coef_[0]
                stat = np.linalg.norm( VN @ (theta - tildetheta))**2
                lsstatnull_selec.append(stat)
                # saturated
                stat = np.linalg.norm( tildeGN_12 @ matXtrue.T @ (y-barpi))**2
                lsstatnull_sat.append(stat)
            lsstatnull_sat = np.array(lsstatnull_sat)
            lsstatnull_selec = np.array(lsstatnull_selec)
            if self.sampling_algorithm == 'RS':
                for j in range(len(states)):
                    lspvals_sat.append(np.mean(lsstat_sat[j]<=lsstatnull_sat))
                    lspvals_selec.append(np.mean(lsstat_selec[j]<=lsstatnull_selec))  
            else:                
                for j in range(len(states)):
                    pval_selec, pval_sat = 0,0
                    normalization = 0
                    for l,y in enumerate(statesnull):
                        proba = compute_proba(y,signull)
                        normalization += proba
                        pval_selec += proba * (lsstat_selec[j]<=lsstatnull_selec[l])
                        pval_sat += proba * (lsstat_sat[j]<=lsstatnull_selec[l])
                    lspvals_sat.append(pval_selec / normalization)
                    lspvals_selec.append(pval_selec / normalization)
        if grad_descent['return_gaps']:
            return lspvals_selec, lspvals_sat, gaps
        else:
            return lspvals_selec, lspvals_sat