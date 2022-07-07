import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from tqdm.notebook import tqdm
import torch.optim as optim
import torch
import os
        
from .tools import *

class Sampling:
    
    def __init__(self, sampling_algorithm):
        if sampling_algorithm is None:
            if (self.X).shape[1]<=15:
                sampling_algorithm = 'RS' # Rejection Sampling
            else:
                sampling_algorithm = 'SA' # Simulated Annealing
        self.sampling_algorithm = sampling_algorithm
    
    
    def change_sampling_algorithm(self, sampling_algorithm):
        self.sampling_algorithm = sampling_algorithm

    def SEI_by_sampling(self, sig, nb_ite=100000):
        """Computes states belonging to the selection event.

        Parameters
        ----------
        sig : list of float
            unconditional expectation of the response vector.
        X : 2 dimensional matrix
            design matrix.
        lamb : float
            regularization parameter for the l1-penalty.
        M : array of integers
            selected support.

        Returns
        -------
        states : vectors of the hypercube belonging to the selection event.

        Note
        ----
        Each vector appearing in the returned list is present only once. Note that the set of states returned be the algorithm might not contains all the vectors belonging to the selection event.
        """
        self.sampling_algorithm = 'RS'
        saved_states = []
        count = 0
        n,p = (self.X).shape
        for i in tqdm(range(nb_ite)):
            y = np.random.rand(n)<=sig
            # We compute the solution of the Sparse Logistic Regression with the current 'y'
            if np.sum(y) not in [0,n]:
                model = LogisticRegression(penalty='l1', C = 1/self.lamb, solver='liblinear', fit_intercept=False)
                model.fit(self.X, y)
                theta_hat = model.coef_[0]
                M2 = np.where( np.abs(theta_hat) > 1e-5)[0]
                if equalset(self.M,M2) and (len(self.M)==len(M2)):
                    saved_states.append(y)
                    count += 1
        return saved_states

    def SEI_SLR(self, temperature=None, delta=0.009, total_length_SEISLR_path=1500, backup_start_time=1000, random_start=True, conditioning_signs=True):
        """SEI-SLR (Selection Event Identification for the Sparse Logistic Regression) algorithm.

        Parameters
        ----------
        settings : dictionary of with the settings of the experiments to conduct
            each entry of the dictionary should be of the form {'n':20,'p':10,'X':np.random.normal(n,p,(0,1)),'truetheta':np.ones(p),'cross_val':False,'lamb':2,'temperature':callable_function}.
        conditioning_signs : bool
                If True, we consider that the selection event corresponds to the states allowing to recover both the selected support (i.e. the none zero entries of thete_obs) and the vector of signs (i.e. the correct signs of the none zero entries of theta_obs).
        total_length_SEISLR_path : bool
            total number of steps performed by the SEI-SLR algorithm.
        compute_pvalues : bool
            If True, the function will compute and save the p-values associated to the last visited states.

        Saved files
        -----------
        lamb.npy : chosen regularization parameter
        yobs.npy : observed response vector
        theta_obs.npy : solution of the l1-penalized likelihood model corresponding to yobs
        sig.npy : expectation of the response vector
        M.npy : selected support
        X.npy : design matrix

        last_y.npy : list of the last visiated vectors by the SEI-SLR
        FNR.npy : false negative rate for the last visited states.
        """
        
        self.sampling_algorithm = 'SA'
        def b(x):
            return (1-np.sqrt(np.min([-x/delta, 1])))

        n,p = np.shape(self.X)
        if temperature is None:
            temperature = logtemp
        try:
            idjob = os.getpid()
        except:
            idjob = 0
        np.random.seed()
        idjob += np.random.randint(1,1000)
        np.random.seed(idjob)

        # Definition of the selection event
        complementM = [i for i in range(p) if i not in self.M]
        y = np.copy(self.yobs)
        d = len(self.M)
        matX = self.X[:,self.M]
        complement_XT = self.X[:,[k for k in range(p) if k not in self.M]].T

        # list containing the last visited states
        last_y = []
        # list containing the false negative rate for the last visited states
        ls_FNR = []
        if random_start:
            y = np.random.rand(n)>=0.5
            # We compute the solution of the Sparse Logistic Regression with the current 'y'
            model = LogisticRegression(penalty='l1', C = 1/self.lamb, solver='liblinear', fit_intercept=False)
            model.fit(self.X, y)
            theta_hat = model.coef_[0]
            sig_hat = sigmoid(self.X @ theta_hat)

            # We set the sign the vector that best suits the KKT constraints
            S = np.clip(self.X.T @ (y-sig_hat) / self.lamb, -1, 1)

            # We compute energy
            if conditioning_signs:
                p1y = np.sum(1 - S[self.M]*self.SM)/len(self.M)
            else:
                p1y = np.sum(1 - np.abs(S[self.M]))/len(self.M)
            p2y = b(np.max(np.abs(S[complementM]))-1)
            E = (np.max([0, p1y, p2y]))
            old_energy = E
        else:
            old_energy = 0

        # Lists to save information along the algorithm
        nsave = 10
        saved_y = -np.ones((nsave,n))
        index_save = 0
        saved_theta_hat = np.zeros((nsave,(self.X).shape[1]))


        # We keep a list of the last 10 admissible points "y" visited. When the time spent between 
        # the last admissible point seen is greater than 'time_before_restart', we restart the chain from a 'y' randomly
        # chosen among those ten admissible points.
        time_last_admissible = 1
        time_before_restart = 3000
        y_admissible = np.zeros((10,n))
        y_admissible[0,:] = self.yobs
        next_y_admis = 1

        for j in range(total_length_SEISLR_path):
            if time_last_admissible > time_before_restart:
                # We restart the procedure from a admissible 'y'
                ind = np.random.randint(0,min(y_admissible.shape[0],next_y_admis))
                y = np.copy(y_admissible[ind,:])
                time_last_admissible = 0
            else:
                # We visit a adjacent point. We ensure to avoid the two vectors (with all zeros or all ones), otherwise
                # solving the Logistic Regression with sklearn fails
                corner_avoided = False
                while(not(corner_avoided)):
                    pos = np.random.randint(0,high=n)
                    y[pos] = (y[pos]+1)%2
                    if np.sum(y) in [len(y),0]:
                        y[pos] = (y[pos]+1)%2
                    else:
                        corner_avoided = True
            foundy = False
            count = 0
            while ((foundy != True) and count < nsave):
                if (y==saved_y[(count+index_save-1)%nsave,:]).all():
                    foundy = True
                    theta_hat = saved_theta_hat[(count+index_save-1)%nsave,:]
                else:
                    count += 1
            if not(foundy):
                # We compute the solution of the Sparse Logistic Regression with the current 'y'
                model = LogisticRegression(penalty='l1', C = 1/self.lamb, solver='liblinear', fit_intercept=False)
                model.fit(self.X, y)
                theta_hat = model.coef_[0]
                saved_theta_hat[index_save,:] = theta_hat
                index_save = (1+index_save)%nsave
            sig_hat = sigmoid(self.X @ theta_hat)

            # We set the sign the vector that best suits the KKT constraints
            S = np.clip(self.X.T @ (y-sig_hat) / self.lamb, -1, 1)

            # We compute the energy
            if conditioning_signs:
                p1y = np.sum(1 - S[self.M]*self.SM)/len(self.M)
            else:
                p1y = np.sum(1 - np.abs(S[self.M]))/len(self.M)
            p2y = b(np.max(np.abs(S[complementM]))-1)
            E = (np.max([0, p1y, p2y]))
            new_energy = E
            # Energy gap
            deltaE = new_energy - old_energy

            T = temperature(j+2)
            if time_last_admissible != 0:
                if np.exp(- deltaE / T) <= np.random.rand(): 
                    # If the current 'y' does not come from a restart of the chain from an admissible point
                    y[pos] = (y[pos]+1)%2 # We come back to the previous state
                else:
                   # If the current 'y' comes from a restart of the chain, we always accept it
                    old_energy = new_energy

            if np.max(np.abs(theta_hat[complementM]))>1e-7:
                time_last_admissible += 1
            else:
                time_last_admissible = 1
                if np.sum([(y==y_admis).all() for y_admis in y_admissible])==0:
                    y_admissible[next_y_admis % y_admissible.shape[0],:] = np.copy(y)
                    next_y_admis += 1

            # Saving data        
            if j>backup_start_time:
                last_y.append(np.array(y))
                Mhat = np.where(np.abs(theta_hat)>1e-5)[0]
                false_negatives = 0
                for k in self.M:
                    if k not in Mhat:
                        false_negatives += 1
                ls_FNR.append(false_negatives/len(self.M))
        return last_y, ls_FNR


