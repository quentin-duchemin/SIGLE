import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import scipy
import scipy as sc
import scipy.stats
from tqdm.notebook import tqdm
from ..tools import *
from .FiguresSigle import FiguresSigle

class Sigle(FiguresSigle):
    """Class implementing the Post-Selection Inference procedure proposed with the SIGLE method in both the selected and the saturated model.
    """
    
    def __init__(self):
        pass   

    def params_saturated(self, bern, states):
        """Computes the probability of the vector of bits 'z' when the expected value of the response vector is given by 'bern'

        Parameters
        ----------
        bern : vector of floats
            expected value of the response vector.
        states : list of vectors of bits
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').

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
        """Computes :math:`\overline \\theta(\\theta^*) \in \mathbb R^s` which is the unique vector satisfying :math:`\mathbf X_M^{\\top}\sigma(\mathbf X_M \overline \\theta (\\theta^*))=\mathbf X_M^{\\top} \overline \pi^{\\theta^*}` 
        where :math:`\overline \pi^{\\theta^*}` is the input parameter 'barpi'.
        Parameters
        ----------
        barpi: vector
            :math:`\overline \pi^{\\theta^*}`
        """
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
        """Computes quantities arising in the assumption of our conditional Central Limit Theorem.
        
        Parameters
        ----------
        tildeGN_12 : 2 dimensional matrix
            matrix  (\bar G_N(\theta^*))^{-1/2} 
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        
        Returns
        -------
        coarse_upper_bound : float
            Quantity arising in the assumption of our Conditional Central Limit Theorem and that should go to 0 as $N\to \infty$ to meet our assumption.
        upper_bound : float
            Finer bound that arise in our Theorem and that should be tending to 0 as $N \to \infty$ to meet our assumption. 
        """
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

    def pval_SIGLE(self, states, barpi, net=None, use_net_MLE=False, l2_regularization=100000, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}, calibrated_from_samples=False, statesnull=None, signull=None):
        """Computes the P-values using the post-selection inference method SIGLE (both in the saturated and the selected model).

        Parameters
        ----------
        states : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        net: neural network
            neural network trained to compute \Psi = \Xi^{-1}. Stated otherwise, given some \rho \in \mathds R^s (where s=|M|), the network should output the unique (if it exists) vector \theta \in \mathds R^s such that \rho = X[:,M].T \sigma(X[:,M] @ theta). 
        use_net_MLE : bool
            Set to True if one wants to use the neural network in 'net' to compute the MLE.
        l2_regularization : float
            l2 regularization used to compute the MLE from the solver of Logistic Regression in sk-learn if 'use_net_MLE' is False. It should be as large as possible to remove regularization.
        grad_descent : dictionary
            Should be of the form {'lr':0.01,'return_gaps':True,'max_ite':100} and it is used for the method 'compute_theta_bar'. 
            - 'lr' is the step size for the gradient descent algorithm.
            - 'return_gaps' is set to True if one wants to return at the end of the algorithm the list of $\| X_M^{\top} (\sigma(X_M \bar \theta^{(t)})- \bar \pi) \|_{\infty}$ for the iterates $\theta^{(t)}$ of the gradient descent algorithm.
            - 'max_ite' is the maximal number of iterations in the gradient descent algorithm.
        calibrated_from_samples : bool
            Set to 'True' if we want to calibrate the testing procedure from samples in 'statesnull'.
        statesnull : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS') when we consider the null hypothesis.
        signull : vector in [0,1]^n
            Expectation of the response vector under the null. It is used when we 'calibrated_from_samples' is True and when 'sampling_algorithm' is 'SA' (Simulated Annealing).

        Returns
        -------
        lspvals_selec : list of floats 
            samples of p-values using SIGLE in the selected model.
        lspvals_sat : list of floats
            samples of p-values using SIGLE in the saturated model.

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
        

    def pval_alt_SIGLE_sat(self, states, center_sat, cov_sat, l2_regularization=100000, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}, statesnull=None, signull=None):
        """Computes the P-values using the post-selection inference method SIGLE (both in the saturated and the selected model).

        Parameters
        ----------
        states : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        l2_regularization : float
            l2 regularization used to compute the MLE from the solver of Logistic Regression in sk-learn if 'use_net_MLE' is False. It should be as large as possible to remove regularization.
        grad_descent : dictionary
            Should be of the form {'lr':0.01,'return_gaps':True,'max_ite':100} and it is used for the method 'compute_theta_bar'. 
            - 'lr' is the step size for the gradient descent algorithm.
            - 'return_gaps' is set to True if one wants to return at the end of the algorithm the list of $\| X_M^{\top} (\sigma(X_M \bar \theta^{(t)})- \bar \pi) \|_{\infty}$ for the iterates $\theta^{(t)}$ of the gradient descent algorithm.
            - 'max_ite' is the maximal number of iterations in the gradient descent algorithm.
        calibrated_from_samples : bool
            Set to 'True' if we want to calibrate the testing procedure from samples in 'statesnull'.
        statesnull : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS') when we consider the null hypothesis.
        signull : vector in [0,1]^n
            Expectation of the response vector under the null. It is used when we 'calibrated_from_samples' is True and when 'sampling_algorithm' is 'SA' (Simulated Annealing).

        Returns
        -------
        lspvals_selec : list of floats 
            samples of p-values using SIGLE in the selected model.
        lspvals_sat : list of floats
            samples of p-values using SIGLE in the saturated model.

        Note
        ----
        If \rho \in \mathds R^s (where s=|M|) can be written as \rho = X[:,M].T @ y where y \in \{0,1\}^n, then net(\rho) is the unconditional unpenalized MLE using the design X[:,M] and the response variable y.
        """
        n,p = (self.X).shape
        matXtrue = self.X[:,self.M]
        
        lspvals_sat = []

        lsstat_sat, lsstatnull_sat = [], []
        for i in tqdm(range(len(states))):
            y = np.array(states[i])
            # saturated
            stat = np.linalg.norm( cov_sat @ (y-center_sat))**2
            lsstat_sat.append(stat)
        lsstat_sat = np.array(lsstat_sat)
        for i in tqdm(range(len(statesnull))):
            y = np.array(statesnull[i])
            # saturated
            stat = np.linalg.norm( cov_sat @ (y-center_sat))**2
            lsstatnull_sat.append(stat)
        lsstatnull_sat = np.array(lsstatnull_sat)
        if self.sampling_algorithm == 'RS':
            for j in range(len(states)):
                lspvals_sat.append(np.mean(lsstat_sat[j]<=lsstatnull_sat))
        else:     
            probas = np.zeros(len(statesnull))
            for l,y in enumerate(statesnull):
                probas[l] = compute_proba(y,signull)
            probas /= np.sum(probas)
            for j in range(len(states)):
                pval_sat = np.sum(probas * (lsstat_sat[j]<=lsstatnull_sat))
                lspvals_sat.append(pval_sat)
#             for j in range(len(states)):
#                 pval_sat = 0
#                 normalization = 0
#                 for l,y in enumerate(statesnull):
#                     proba = compute_proba(y,signull)
#                     normalization += proba
#                     pval_sat += proba * (lsstat_sat[j]<=lsstatnull_sat[l])
#                 lspvals_sat.append(pval_sat / normalization)
        return lspvals_sat
 
    
    def pval_alt_SIGLE_sel(self, states, center_sel, cov_sel, l2_regularization=100000, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}, statesnull=None, signull=None):
        """Computes the P-values using the post-selection inference method SIGLE (both in the saturated and the selected model).

        Parameters
        ----------
        states : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        l2_regularization : float
            l2 regularization used to compute the MLE from the solver of Logistic Regression in sk-learn if 'use_net_MLE' is False. It should be as large as possible to remove regularization.
        grad_descent : dictionary
            Should be of the form {'lr':0.01,'return_gaps':True,'max_ite':100} and it is used for the method 'compute_theta_bar'. 
            - 'lr' is the step size for the gradient descent algorithm.
            - 'return_gaps' is set to True if one wants to return at the end of the algorithm the list of $\| X_M^{\top} (\sigma(X_M \bar \theta^{(t)})- \bar \pi) \|_{\infty}$ for the iterates $\theta^{(t)}$ of the gradient descent algorithm.
            - 'max_ite' is the maximal number of iterations in the gradient descent algorithm.
        calibrated_from_samples : bool
            Set to 'True' if we want to calibrate the testing procedure from samples in 'statesnull'.
        statesnull : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS') when we consider the null hypothesis.
        signull : vector in [0,1]^n
            Expectation of the response vector under the null. It is used when we 'calibrated_from_samples' is True and when 'sampling_algorithm' is 'SA' (Simulated Annealing).

        Returns
        -------
        lspvals_selec : list of floats 
            samples of p-values using SIGLE in the selected model.
        lspvals_sat : list of floats
            samples of p-values using SIGLE in the saturated model.

        Note
        ----
        If \rho \in \mathds R^s (where s=|M|) can be written as \rho = X[:,M].T @ y where y \in \{0,1\}^n, then net(\rho) is the unconditional unpenalized MLE using the design X[:,M] and the response variable y.
        """
        n,p = (self.X).shape
        matXtrue = self.X[:,self.M]
        
        lspvals_selec = []

        lsstat_selec, lsstatnull_selec = [], []
        for i in tqdm(range(len(states))):
            y = np.array(states[i])
            # selected
            model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
            model.fit(matXtrue, y)
            theta = model.coef_[0]
            stat = np.linalg.norm( cov_sel @ (theta - center_sel))**2
            lsstat_selec.append(stat)
        for i in tqdm(range(len(statesnull))):
            y = np.array(statesnull[i])
            # selected
            model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
            model.fit(matXtrue, y)
            theta = model.coef_[0]
            stat = np.linalg.norm( cov_sel @ (theta - center_sel))**2
            lsstatnull_selec.append(stat)
        lsstatnull_selec = np.array(lsstatnull_selec)
        if self.sampling_algorithm == 'RS':
            for j in range(len(states)):
                lspvals_selec.append(np.mean(lsstat_selec[j]<=lsstatnull_selec))  
        else:
            probas = np.zeros(len(statesnull))
            for l,y in enumerate(statesnull):
                probas[l] = compute_proba(y,signull)
            probas /= np.sum(probas)
            for j in range(len(states)):
                pval_selec = np.sum(probas * (lsstat_selec[j]<=lsstatnull_selec))
                lspvals_selec.append(pval_selec)
#             for j in range(len(states)):
#                 pval_selec, pval_sat = 0,0
#                 normalization = 0
#                 for l,y in enumerate(statesnull):
#                     proba = compute_proba(y,signull)
#                     normalization += proba
#                     pval_selec += proba * (lsstat_selec[j]<=lsstatnull_selec[l])
#                 lspvals_selec.append(pval_selec / normalization)
        return lspvals_selec
        
    
    def pval_alt_SIGLE(self, states, center_sat, cov_sat, center_sel, cov_sel, l2_regularization=100000, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}, statesnull=None, signull=None):
        """Computes the P-values using the post-selection inference method SIGLE (both in the saturated and the selected model).

        Parameters
        ----------
        states : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        l2_regularization : float
            l2 regularization used to compute the MLE from the solver of Logistic Regression in sk-learn if 'use_net_MLE' is False. It should be as large as possible to remove regularization.
        grad_descent : dictionary
            Should be of the form {'lr':0.01,'return_gaps':True,'max_ite':100} and it is used for the method 'compute_theta_bar'. 
            - 'lr' is the step size for the gradient descent algorithm.
            - 'return_gaps' is set to True if one wants to return at the end of the algorithm the list of $\| X_M^{\top} (\sigma(X_M \bar \theta^{(t)})- \bar \pi) \|_{\infty}$ for the iterates $\theta^{(t)}$ of the gradient descent algorithm.
            - 'max_ite' is the maximal number of iterations in the gradient descent algorithm.
        calibrated_from_samples : bool
            Set to 'True' if we want to calibrate the testing procedure from samples in 'statesnull'.
        statesnull : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS') when we consider the null hypothesis.
        signull : vector in [0,1]^n
            Expectation of the response vector under the null. It is used when we 'calibrated_from_samples' is True and when 'sampling_algorithm' is 'SA' (Simulated Annealing).

        Returns
        -------
        lspvals_selec : list of floats 
            samples of p-values using SIGLE in the selected model.
        lspvals_sat : list of floats
            samples of p-values using SIGLE in the saturated model.

        Note
        ----
        If \rho \in \mathds R^s (where s=|M|) can be written as \rho = X[:,M].T @ y where y \in \{0,1\}^n, then net(\rho) is the unconditional unpenalized MLE using the design X[:,M] and the response variable y.
        """
        n,p = (self.X).shape
        matXtrue = self.X[:,self.M]
        
        lspvals_selec = []
        lspvals_sat = []

        lsstat_sat, lsstatnull_sat = [], []
        lsstat_selec, lsstatnull_selec = [], []
        for i in tqdm(range(len(states))):
            y = np.array(states[i])
            # selected
            model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
            model.fit(matXtrue, y)
            theta = model.coef_[0]
            stat = np.linalg.norm( cov_sel @ (theta - center_sel))**2
            lsstat_selec.append(stat)
            # saturated
            stat = np.linalg.norm( cov_sat @ (y-center_sat))**2
            lsstat_sat.append(stat)
        for i in tqdm(range(len(statesnull))):
            y = np.array(statesnull[i])
            # selected
            model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
            model.fit(matXtrue, y)
            theta = model.coef_[0]
            stat = np.linalg.norm( cov_sel @ (theta - center_sel))**2
            lsstatnull_selec.append(stat)
            # saturated
            stat = np.linalg.norm( cov_sat @ (y-center_sat))**2
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
                    pval_sat += proba * (lsstat_sat[j]<=lsstatnull_sat[l])
                lspvals_sat.append(pval_selec / normalization)
                lspvals_selec.append(pval_selec / normalization)
        return lspvals_selec, lspvals_sat
 