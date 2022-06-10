import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from inverse_map import inverse_map, train_network
import torch
import scipy
import scipy as sc
import scipy.stats

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

def SEI_SLR(settings=None,random_start=True,conditioning_signs=True, total_length_SEISLR_path=1500, compute_pvalues = False):
    """SEI-SLR (Selection Event Identification for the Sparse Logistic Regression) algorithm.
    
    Parameters
    ----------
    settings : dictionary of with the settings of the experiments to conduct
        each entry of the dictionary should be of the form {'n':20,'p':10,'X':np.random.normal(n,p,(0,1)),'theta':np.ones(p),'cross_val':False,'lamb':2,'temperature':callable_function, 'nb_expe':nb_expe, 'file':'test/'}.
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
    import numpy as np
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import scipy
    from scipy.integrate import odeint
    import scipy.stats
    from inverse_map import inverse_map, train_network
    import torch
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
    import os
    def sigmoid1(u):
        return (np.exp(-u)/(1+np.exp(-u))**2)
    def sigmoid(u):
        return (1/(1+np.exp(-u)))
    def b(x):
        delta = 0.009
        return (1-np.sqrt(np.min([-x/delta, 1])))


    for setting in settings:
        np.random.seed(1)
        n = setting['n']
        p = setting['p']
        temperature = setting['temperature']
        nb_expe = setting['nb_expe']
        if setting['X'] is None:
            X = np.random.normal(0,1,(n,p))
            X /= np.tile(np.linalg.norm(X,axis=0),(n,1))
        else:
            X = setting['X']
        path = setting['file']       
        theta = setting['theta']
        sig = sigmoid(X @ theta)

        # Observation (that is used to define the selection event by solving the Sparse Logistic Regression)
        if setting['cross_val']:
            M = []
            while len(M)==0:
                yobs = np.random.rand(n) <= sig
                model = LogisticRegressionCV(Cs = [0.15*(i+1) for i in range(33)], penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(X, yobs)
                lamb = 1/model.C_[0]

                theta_obs = model.coef_[0]
                M = np.where( np.abs(theta_obs) > 1e-5)[0]
                SMY = np.sign(theta_obs[M])
        else:
            M = []
            while len(M)==0:
                yobs = np.random.rand(n) <= sig
                lamb = setting['lamb']
                model = LogisticRegression(C=1/lamb, penalty='l1', solver='liblinear', fit_intercept=False)
                model.fit(X, yobs)
                theta_obs = model.coef_[0]
                M = np.where( np.abs(theta_obs) > 1e-5)[0]
                SMY = np.sign(theta_obs[M])
        
        
        try:
            idjob = os.getpid()
        except:
            idjob = 0
        np.random.seed()
        idjob += np.random.randint(1,1000)
        
        
        np.save(path+str(idjob)+'lamb.npy',np.array([lamb]))
        np.save(path+str(idjob)+'yobs.npy',yobs)
        np.save(path+str(idjob)+'theta_obs.npy',theta_obs)
        np.save(path+str(idjob)+'sig.npy',sig)
        np.save(path+str(idjob)+'M.npy',M)
        np.save(path+str(idjob)+'X.npy',X)

        # Definition of the selection event
        complementM = [i for i in range(p) if i not in M]
        y = np.copy(yobs)
        d = len(M)
        matX = X[:,M]
        complement_XT = X[:,[k for k in range(p) if k not in M]].T
    
        # list containing the last visited states
        last_y = []
        # list containing the p-values for SIGLE in the selected model
        lspvalsSEL = []
        lsstatsSEL = []
        # list containing the p-values for SIGLE in the selected model
        lspvalsSAT = []
        lsstatsSAT = []
        # list containing the false negative rate for the last visited states
        ls_FNR = []
        for i in range(nb_expe):
            np.random.seed(idjob+i)
            if random_start:
                y = np.random.rand(n)>=0.5
                # We compute the solution of the Sparse Logistic Regression with the current 'y'
                model = LogisticRegression(penalty='l1', C = 1/lamb, solver='liblinear', fit_intercept=False)
                model.fit(X, y)
                theta_hat = model.coef_[0]
                sig_hat = sigmoid(X @ theta_hat)

                # We set the sign the vector that best suits the KKT constraints
                S = np.clip(X.T @ (y-sig_hat) / lamb, -1, 1)

                # We compute energy
                if conditioning_signs:
                    p1y = np.sum(1 - S[M]*SMY)/len(M)
                else:
                    p1y = np.sum(1 - np.abs(S[M]))/len(M)
                p2y = b(np.max(np.abs(S[complementM]))-1)
                E = (np.max([0, p1y, p2y]))
                old_energy = E
            else:
                old_energy = 0

            # Lists to save information along the algorithm
            nsave = 10
            saved_y = -np.ones((nsave,n))
            index_save = 0
            saved_theta_hat = np.zeros((nsave,X.shape[1]))

            # Boolean: set to True at the end of the simulated annealing to allow transition only towards admissible points
            ending_with_hard_transitions = False
            HARD_transition = False

            # We keep a list of the last 10 admissible points "y" visited. When the time spent between 
            # the last admissible point seen is greater than 'time_before_restart', we restart the chain from a 'y' randomly
            # chosen among those ten admissible points.
            time_last_admissible = 1
            time_before_restart = 3000
            y_admissible = np.zeros((10,n))
            y_admissible[0,:] = yobs
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
                    model = LogisticRegression(penalty='l1', C = 1/lamb, solver='liblinear', fit_intercept=False)
                    model.fit(X, y)
                    theta_hat = model.coef_[0]
                    saved_theta_hat[index_save,:] = theta_hat
                    index_save = (1+index_save)%nsave
                sig_hat = sigmoid(X @ theta_hat)

                # We set the sign the vector that best suits the KKT constraints
                S = np.clip(X.T @ (y-sig_hat) / lamb, -1, 1)

                # We compute the energy
                if conditioning_signs:
                    p1y = np.sum(1 - S[M]*SMY)/len(M)
                else:
                    p1y = np.sum(1 - np.abs(S[M]))/len(M)
                p2y = b(np.max(np.abs(S[complementM]))-1)
                E = (np.max([0, p1y, p2y]))
                new_energy = E
                # Energy gap
                deltaE = new_energy - old_energy

                # Do we accept the transition towards the current 'y' ?
                if HARD_transition:
                    # In case we only accept if the point is admissible
                    if np.max(np.abs(theta_hat[complementM]))>1e-7:
                        y[pos] = (y[pos]+1)%2
                    else:
                        old_energy = new_energy
                else:
                    # In case we do the simulated annealing
                    T = temperature(j+2)
                    if time_last_admissible != 0:
                        if np.exp(- deltaE / T) <= np.random.rand(): 
                            # If the current 'y' does not come from a restart of the chain from an admissible point
                            y[pos] = (y[pos]+1)%2 # We come back to the previous state
                        else:
                           # If the current 'y' comes from a restart of the chain, we always accept it
                            old_energy = new_energy

                    if ending_with_hard_transitions and j>int(0.9*total_length_SEISLR_path):
                        HARD_transition = True    

                    if np.max(np.abs(theta_hat[complementM]))>1e-7:
                        time_last_admissible += 1
                    else:
                        time_last_admissible = 1
                        if np.sum([(y==y_admis).all() for y_admis in y_admissible])==0:
                            y_admissible[next_y_admis % y_admissible.shape[0],:] = np.copy(y)
                            next_y_admis += 1

                # Saving data        
                if j>int(0.85*total_length_SEISLR_path):
                    last_y.append(list(y))
                    
                    Mhat = np.where(np.abs(theta_hat)>1e-5)[0]
                    false_negatives = 0
                    for k in M:
                        if k not in Mhat:
                            false_negatives += 1
                    ls_FNR.append(false_negatives/len(M))
            np.save(path+'last_y'+str(idjob)+'.npy',np.array(last_y))
            np.save(path+'FNR'+str(idjob)+'.npy',np.array(ls_FNR))


        def compute_proba(z,bern):
            """Computes the probability of the vector of bits 'z' when the expected value of the response vector is given by 'bern'

            Parameters
            ----------
            z : list of bits.
            bern : vector of floats.
            """
            n = len(bern)
            return np.exp( np.sum( z*np.log(bern) + (np.ones(n)-z)*np.log(np.ones(n)-bern)))

        def params_saturated(bern, matXtrue, lsy):
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
            normalization = 0
            barpi = np.zeros(n)
            for y in lsy:
                proba = compute_proba(y,bern)
                barpi += y * proba
                normalization += proba
            barpi /= normalization
            rho = matXtrue.T @ barpi.T
            tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
            usvd,s,vt = np.linalg.svd(tildeGN)
            tildeGN12 = usvd @ np.diag(np.sqrt(s)) @ vt
            tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt
            return tildeGN_12, barpi

        if compute_pvalues:
            # Computing the P-values on the last visited states for SIGLE in the SATURATED model
            yobs = np.load(path+str(idjob)+'yobs.npy')
            sig = np.load(path+str(idjob)+'sig.npy')
            M = np.load(path+str(idjob)+'M.npy')
            X = np.load(path+str(idjob)+'X.npy')
            lsy = np.load(path+'last_y'+str(idjob)+'.npy')
            lamb = np.load(path+str(idjob)+'lamb.npy')[0]
            theta_obs = np.load(path+str(idjob)+'theta_obs.npy')
            n,p = np.shape(X)
            truetheta = np.zeros(p)
            bern = sigmoid(X @ truetheta)
            matXtrue = X[:,M]
            tildeGN_12, barpi = params_saturated(bern, matXtrue, lsy)
            for y in lsy:
                stat = np.linalg.norm( tildeGN_12 @ matXtrue.T @ (y-barpi))**2
                lsstatsSAT.append(stat)
                df = len(M)
                lspvalsSAT.append(1-scipy.stats.chi2.cdf(stat, df))
            np.save(path+str(idjob)+'barpi.npy',barpi)
            np.save(path+str(idjob)+'pval_saturated.npy',np.array(lspvalsSAT))
            np.save(path+str(idjob)+'stat.npy',np.array(lsstatsSAT))


            # Computing the P-values on the last visited states for SIGLE in the SELECTED model
            yobs = np.load(path+str(idjob)+'yobs.npy')
            sig = np.load(path+str(idjob)+'sig.npy')
            M = np.load(path+str(idjob)+'M.npy')
            X = np.load(path+str(idjob)+'X.npy')
            lsy = np.load(path+'last_y'+str(idjob)+'.npy')
            lamb = np.load(path+str(idjob)+'lamb.npy')[0]
            theta_obs = np.load(path+str(idjob)+'theta_obs.npy')
            n,p = np.shape(X)
            truetheta = np.zeros(p)
            bern = sigmoid(X @ truetheta)
            matXtrue = X[:,M]
            truetheta = np.zeros(p)
            bern = sigmoid(X @ truetheta)
            tildeGN_12, barpi = params_saturated(bern, matXtrue, lsy)
            net, losses = train_network(matXtrue,3)
            rho = matXtrue.T @ barpi.T
            tildetheta = net(torch.from_numpy(rho.T).float())
            tildetheta = tildetheta.detach().numpy()

            tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
            usvd,s,vt = np.linalg.svd(tildeGN)
            tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt

            GNtilde = matXtrue.T @ np.diag(sigmoid1(matXtrue @ tildetheta)) @ matXtrue
            VN = tildeGN_12 @ GNtilde

            for y in lsy:
                rho = matXtrue.T @ y.T
                theta = net(torch.from_numpy(rho.T).float())
                theta = theta.detach().numpy()
                stat = np.linalg.norm( VN @ (theta - tildetheta))**2
                lsstatsSEL.append(stat)
                df = len(M)
                lspvalsSEL.append(1-scipy.stats.chi2.cdf(stat, df))    

            np.save(path+str(idjob)+'tildetheta.npy',tildetheta)
            np.save(path+str(idjob)+'pval_selected.npy',np.array(lspvalsSEL))
            np.save(path+str(idjob)+'stat_selected.npy',np.array(lsstatsSEL))

        
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
        
        
def compute_selection_event(theta_obs,X,yobs,lamb,conditioning_signs=True, compare_with_energy=False):
    """Finds all the vectors belonging to the selection event.
    
    Parameters
    ----------
    theta_obs : 
        solution of the l1-penalized likelihood model
    X : 2 dimensional matrix
        design matrix.
    yobs : vector of bits
        observed response vector.
    lamb : float
        regularization parameter for the l1-penalty.
    conditioning_signs : bool
        If True, we consider that the selection event corresponds to the states allowing to recover both the selected support (i.e. the none zero entries of thete_obs) and the vector of signs (i.e. the correct signs of the none zero entries of theta_obs).
    compare_with_energy : bool
        If True, we show a warning when some state wuold have been not correctly classified (as in or out of the selection event) based on the energy.

    Returns
    -------
    nbM_admissibles : integers encoding all the vectors of the hypercube belonging to the selection event. This encoding is performed using the function 'binary_encoding'. 
    ls_states_admissibles : vectors of the hypercube belonging to the selection event.
    """
    def b(x):
        delta = 0.009
        return (1-np.sqrt(np.min([-x/delta, 1])))
    n,p = np.shape(X)
    assert(n<20)
    M = np.where( np.abs(theta_obs) > 1e-5)[0]
    SMY = np.sign(theta_obs[M])
    SM = np.sign(theta_obs)[M] 
    y = np.copy(yobs)
    d = len(M)
    matX = X[:,M]
    complement_XT = X[:,[k for k in range(p) if k not in M]].T
    complementM = [ i for i in range(p) if i not in M]
    y = np.ones(n)
    counter = 0
    lsE = []
    ls_states_admissibles = []
    for k in range(2**n):
        const = k
        for i in range(len(yobs)):
            y[i] = int(const%2)
            const -= const%2
            const /= 2
        if np.sum(y) in [len(y),0]:
            pass
        else:
            model = LogisticRegression(penalty='l1', C = 1/lamb, solver='liblinear', fit_intercept=False)
            model.fit(X, y)
            theta_hat = model.coef_[0]
            sig_hat = sigmoid( X @ theta_hat)
            S = np.clip(X.T @ (y-sig_hat) / lamb, -1, 1)

            # We compute the energy
            if conditioning_signs:
                p1y = np.sum(1 - S[M]*SMY)/len(M)
            else:
                p1y = np.sum(1 - np.abs(S[M]))/len(M)
            p2y = b(np.max(np.abs(S[complementM]))-1)
            E = (np.sum([0, p1y, p2y]))
            lsE.append(E)
            
            if (np.max(np.abs(theta_hat[complementM]))<1e-5) and (np.min(np.abs(theta_hat[M])) > 1e-5):
                counter += 1
                ls_states_admissibles.append(list(y))
                if compare_with_energy and (p2y>1e-3 and p1y>1e-3):
                    print('Warning: a state would have been uncorrectly classified using the energy',p2y)
            else:
                if compare_with_energy and (p2y<1e-3 and p1y<1e-3):
                    print('Warning: a state would have been uncorrectly classified using the energy',p2y )

    nbM_admissibles = np.sort(list(set(map(binary_encoding,ls_states_admissibles))))
    return nbM_admissibles, ls_states_admissibles

def params_saturated(bern, matXtrue, lsy, repetitions_removed=False):
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
    normalization = 0
    barpi = np.zeros(n)
    if repetitions_removed:
        for y in lsy:
            proba = compute_proba(y,bern)
            barpi += y * proba
            normalization += proba
        barpi /= normalization
    else:
        for y in lsy:
            barpi += y
        barpi /= len(lsy)
    rho = matXtrue.T @ barpi.T
    tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
    usvd,s,vt = np.linalg.svd(tildeGN)
    tildeGN12 = usvd @ np.diag(np.sqrt(s)) @ vt
    tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt
    return tildeGN_12, barpi

def compute_proba(z,bern):
    """Computes the probability of the vector of bits 'z' when the expected value of the response vector is given by 'bern'
    
    Parameters
    ----------
    z : list of bits.
    bern : vector of floats.
    """
    n = len(bern)
    return np.exp( np.sum( z*np.log(bern) + (np.ones(n)-z)*np.log(np.ones(n)-bern)))


def histo_time_in_selection_event(indexes, path, ls_states_admissibles):
    """Histogram showing the time spent in the selection event using the SEI-SLR algorithm.
    
    Parameters
    ----------
    indexes : list of integers
        each entry if the integer corresponding to a specific experiment launched using the SEI-SLR algorithm.
    path : string
        path to find the files saved by the SEI-SLR (exemple: 'myfiles/').
    ls_states_admissibles : list of list
        vectors of the hypercube belonging to the selection event.
    """
    selectionevent = [binary_encoding(ya) for ya in ls_states_admissibles]
    state2time = np.zeros(len(selectionevent)+1)
    for k in range(len(indexes)):
        last_y = np.load(path+'last_y'+str(indexes[k])+'.npy')[-10000:-100,:]
        last_fy = [binary_encoding(y) for y in last_y]
        for i,fy in enumerate(last_fy):
            found = False
            for j,fevy in enumerate(selectionevent):
                if fy==fevy:
                    state2time[j] += 1
                    found = True
            if not(found):
                state2time[-1] += 1

    state2time /= np.sum(state2time)
    labels = [str(int(fevy)) for fevy in selectionevent]
    labels += ['   not in $E_{M_0}$']
    cs = ['green' for i in range(len(selectionevent))]
    cs += ['gray']
    plt.bar([i for i in range(len(state2time))],state2time,color=cs, tick_label=labels)
    plt.ylabel('Proposition of time spent',fontsize=12)
    plt.xticks(fontsize=13)
    plt.savefig('repartition_EM0.png',dpi=250)
      
def last_visited_states(indexes, path, lsM_admissibles):
    """Shows that time spent in the selection event using the SEI-SLR algorithm.
    
    Parameters
    ----------
    indexes : list of integers
        each entry if the integer corresponding to a specific experiment launched using the SEI-SLR algorithm.
    path : string
        path to find the files saved by the SEI-SLR (exemple: 'myfiles/').
    lsM_admissibles : list of integers
        each entry encodes some vector of bits (using the function 'binary_encoding').
    """
    X = np.load(path+indexes[0]+'X.npy')
    n,p = np.shape(X)
    M = np.load(path+indexes[0]+'M.npy')
    last_y = []
    for i in range(len(indexes)):
        last_y = last_y+ list(np.load(path+'last_y'+str(indexes[i])+'.npy'))
    last_fy = [binary_encoding(y) for y in last_y]

    try:
        selectionevent = [binary_encoding(ya) for ya in lsM_admissibles]
    except:
        selectionevent = []
    plt.figure()
    color = True
    if color:
        lsx_in = []
        lsx_out = []
        lsy_in = []
        lsy_out = []
        for i,fy in enumerate(last_fy):
            if fy in selectionevent:
                lsx_in.append(i)
                lsy_in.append(fy)
            else:
                lsx_out.append(i)
                lsy_out.append(fy)
        plt.scatter(lsx_in, lsy_in,marker='+',c='green', label='$y^{(t)}$ in $E_{M_0}$')
        plt.scatter(lsx_out ,lsy_out,marker='x',c='gray',label='$y^{(t)}$ outside of $E_{M_0}$')
    else:
        plt.scatter([i for i in range(len(last_fy))], last_fy)
        len(list(set(last_fy))),2**n
    if n<=20:
        nbM_admissibles = np.sort(list(set(map(binary_encoding,lsM_admissibles))))
        for j,fyls_ele in enumerate(nbM_admissibles):
            if j==0:
                plt.plot([0,len(last_fy)], [fyls_ele,fyls_ele], c='red', linestyle=(0, (5, 10)),linewidth=2.0, label='$y \in $E_{M_0}$')
            else:
                plt.plot([0,len(last_fy)], [fyls_ele,fyls_ele], c='red', linestyle=(0, (5, 10)),linewidth=2.0)
    plt.ylabel('Visited state $y_t$', fontsize=13)
    plt.xlabel('Last time steps', fontsize=13)
    
def time_in_selection_event(indexes, path, ls_states_admissibles):
    """Shows that time spent in the selection event using the SEI-SLR algorithm for several different excursions.
    
    Parameters
    ----------
    indexes : list of integers
        each entry if the integer corresponding to a specific experiment launched using the SEI-SLR algorithm.
    path : string
        path to find the files saved by the SEI-SLR (exemple: 'myfiles/').
    ls_states_admissibles : list of list
        vectors of the hypercube belonging to the selection event.
    """
    proportions = np.zeros(len(indexes))
    for k in range(len(indexes)):
        last_y = np.load(path+'last_y'+str(indexes[k])+'.npy')[-5000:-100,:]

        last_fy = [binary_encoding(y) for y in last_y]
        selectionevent = [binary_encoding(ya) for ya in ls_states_admissibles]

        inselectionevent = np.zeros(len(last_fy))
        for i,fy in enumerate(last_fy):
            if fy in selectionevent:
                inselectionevent[i] = 1
        proportions[k] = (np.sum(inselectionevent)/len(inselectionevent))
    plt.scatter(np.array([i for i in range(len(proportions))]), np.sort(proportions))
    plt.xlabel('Index of the simulation', fontsize=12)
    plt.ylabel('Proportion of time spent in $E_{M_0}$', fontsize=12)
    

    
########### TAYLOR

def low_up_taylor(theta_obs,SM,M,X,lamb,ind=None):
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
    bhat = theta_obs[M]
    matXtrue = X[:,M]
    pihat = sigmoid(X @ theta_obs)
    MM = np.linalg.inv(matXtrue.T @ np.diag(pihat*(1-pihat)) @ matXtrue)
    b1 = -lamb * np.diag(SM) @ MM @ SM
    A1 = -np.diag(SM)
    bbar = bhat + lamb* MM @ SM
    
    gamma = np.zeros(len(M))
    if ind is None:
        for i in range(len(M)):
            gamma[i]=SM[i]
        gamma /= np.linalg.norm(gamma)
    else:
        gamma[ind] = 1
    c = MM @ gamma * (gamma.T @ MM @ gamma)**(-1)
    r = (np.eye(len(M))-np.dot(c.reshape(-1,1),gamma.reshape(1,-1))) @ bbar
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

def samples_taylor(path,index,ind=None, thetatrue=None):
    def sigmoid(u):
        return (1/(1+np.exp(-u)))
    if thetatrue is None:
        thetatrue = np.zeros(X.shape[1])
    if index >=0:
        yobs = np.load(path+index+'yobs.npy')
        sig = np.load(path+index+'sig.npy')
        M = np.load(path+index+'M.npy')
        X = np.load(path+index+'X.npy')
        lsy = np.load(path+'last_y'+index+'.npy')
        lamb = np.load(path+index+'lamb.npy')[0]
        theta_obs = np.load(path+index+'theta_obs.npy')
        n,p = X.shape
    elif index == -1:
        np.random.seed(0)
        n,p = 100,20
        X = np.random.normal(0,1,(n,p))
        X /= np.tile(np.linalg.norm(X,axis=0),(n,1))
        d = 2
        proj = X[:,:d] @ np.linalg.inv(X[:,:d].T @ X[:,:d]) @ X[:,:d].T
        X[:,d:] = (np.eye(n)-proj) @ X[:,d:] 
        X /= np.tile(np.linalg.norm(X,axis=0),(n,1))
        matXtrue = X[:,:d]
        sig = sigmoid(matXtrue @ thetatrue)
        M = []
        yobs = np.random.rand(n) <= sig
        lamb=0.7
        model = LogisticRegression(C=1/lamb, penalty='l1', solver='liblinear', fit_intercept=False)
        model.fit(X, yobs)

        theta_obs = model.coef_[0]
        M = np.where( np.abs(theta_obs) > 1e-5)[0]
        print(M)
    else:
        
        np.random.seed(1)
        n,p= 12,30
        X = np.random.normal(0,1,(n,p))
        X /= np.tile(np.linalg.norm(X,axis=0),(n,1))
        sig = sigmoid(X @ thetatrue)
        yobs = np.random.rand(n) <= sig
        lamb=0.7
        model = LogisticRegression(C=1/lamb, penalty='l1', solver='liblinear', fit_intercept=False)
        model.fit(X, yobs)

        theta_obs = model.coef_[0]
        M = np.where( np.abs(theta_obs) > 1e-5)[0]
        print("support",M)


#     truetheta = np.zeros((X.shape[1]))
#     truetheta[:4] = 0*np.ones(4)
#     sig = sigmoid(X @ truetheta)

#     yobs = np.random.rand(n) <= sig
#     model = LogisticRegression(C = 1/lamb, penalty='l1', solver='liblinear', fit_intercept=False)
#     if np.sum(yobs) not in [0,n]:
#         model.fit(X, yobs)
#         theta_obs = model.coef_[0]
#         M = np.where( np.abs(theta_obs) > 1e-5)[0]
#         SM = np.sign(theta_obs[M])
#         matXtrue = X[:,M]
#         print(M)


    bhat = theta_obs[M]
    SM = np.sign(bhat)
    matXtrue = X[:,M]
    pihat = sigmoid(X @ theta_obs)
    MM = np.linalg.inv(matXtrue.T @ np.diag(pihat*(1-pihat)) @ matXtrue)
    b1 = -lamb * np.diag(SM) @ MM @ SM
    A1 = -np.diag(SM)
    bbar = bhat + lamb* MM @ SM
    
    gamma = np.zeros(len(M))
    if ind is None:
        for i in range(len(M)):
            gamma[i]=SM[i]
        gamma /= np.linalg.norm(gamma)
    else:
        gamma[ind] = 1
    c = MM @ gamma * (gamma.T @ MM @ gamma)**(-1)
    r = (np.eye(len(M))-np.dot(c.reshape(-1,1),gamma.reshape(1,-1))) @ bbar
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

#     samples = np.random.normal(gamma.T @ theta_obs[M],np.sqrt(gamma.T @ np.linalg.inv(matXtrue.T @ np.diag(pihat*(1-pihat))@  matXtrue)@gamma),10000)
    
    pitrue = sigmoid(X @ thetatrue)
    samples = np.random.normal(gamma.T @ thetatrue[M],np.sqrt(gamma.T @ np.linalg.inv(matXtrue.T @ np.diag(pitrue*(1-pitrue))@  matXtrue)@gamma),10000)

    idxs = np.where((samples>=vlow) & (samples<=vup))[0]
    samples_truncated = samples[idxs]
    return (samples_truncated)


###############################################
###############################################
###################################### P-VALUES
###############################################
###############################################


def true_conditional_distribution(theta_obs,X,yobs,lamb,truetheta,conditioning_signs=True,states=None):
    """Computes the probability distribution of the response variable conditional to selection event.
    
    Parameters
    ----------
    theta_obs : 
        solution of the l1-penalized likelihood model
    X : 2 dimensional matrix
        design matrix.
    yobs : vector of bits
        observed response vector.
    lamb : float
        regularization parameter for the l1-penalty.
    truetheta : vector of float
        parameter vector such that the expected response variable is sigma( X @ truetheta ).
    conditioning_signs : bool
        If True, we consider that the selection event corresponds to the states allowing to recover both the selected support (i.e. the none zero entries of thete_obs) and the vector of signs (i.e. the correct signs of the none zero entries of theta_obs).
    states : None or list of vectors
        if not None, the vectors in 'states' are considered to be the vectors belonging to the selection event.

    Returns
    -------
    states : vectors of the hypercube belonging to the selection event.
    
    Note
    ----
    If 'state' is None, we find all the states belonging to the selection event by visiting all the vectors in \{0,1\}^n. Note that when 'states' is not None, this list of vectors might have been computed using the function 'SEI_by_sampling': in this case, the list 'states' might not contain all the vectors belonging to the selection event.
    """
    if states is None:
        nbM_admi, states = compute_selection_event(theta_obs,X,yobs,lamb,conditioning_signs=conditioning_signs)       
    bern = sigmoid(X @ truetheta)
    normalization = 0
    ls_probas = []
    for y in states:
        proba = compute_proba(y,bern)
        normalization += proba
        ls_probas.append(proba)
    ls_probas = np.array(ls_probas)
    ls_probas /= normalization
    return (ls_probas, states)


def SEI_by_sampling(sig, X, lamb, M, remove_repetitions=False, nb_ite=100000):
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
    saved_states = []
    count = 0
    n,p = X.shape
    for i in range(nb_ite):
        y = np.random.rand(n)<=sig
        # We compute the solution of the Sparse Logistic Regression with the current 'y'
        if np.sum(y) not in [0,n]:
            model = LogisticRegression(penalty='l1', C = 1/lamb, solver='liblinear', fit_intercept=False)
            model.fit(X, y)
            theta_hat = model.coef_[0]
            M2 = np.where( np.abs(theta_hat) > 1e-5)[0]
            if equalset(M,M2) and (len(M)==len(M2)):
                saved_states.append(y)
                count += 1
        if (count-1)%20==0:
            print(count,' states in the selection event found so far')
            count += 1
    if remove_repetitions:
        states = [list(item) for item in set(tuple(row) for row in saved_states)]
    return saved_states

def pval_weak_learner(statesnull,statesalt,barpi):
    """Computes the P-values obtained from the weak-learner which is a two-sided test based on the statistic \sum_{i=1}^n |\bar \pi^{\pi^0}_i-y_i| where \bar \pi^{\pi^0} is the expectation of the vector of observations under the null conditional to the selection event.
    
    Parameters
    ----------
    probas : list of float
        conditional probabilities under a prescribed alternative associated to each vectors in selection event (corresponding to each entry of the input 'states').
    probasnull : list of float
        conditional probabilities under the null associated to each vectors in selection event (corresponding to each entry of the input 'states').
    states : list of vectors in {0,1}^n
        all the vector of the hypercube belonging to the selection event.
    barpi : list of float
        expectation of the vector of observations under the null conditional to the selection event.

    Returns
    -------
    lspvalsnaive: samples of p-values obtained from the weak-learner.
    """
#     idxs_null = np.random.choice([i for i in range(len(states))], size=300, p=probasnull)
#     idxs = np.random.choice([i for i in range(len(states))], size=300, p=probas)

    samplesnull = []
    for i in range(len(statesnull)):
        samplesnull.append(np.sum(np.abs(barpi-statesnull[i])))

    lspvalsnaive = []
    samplesalt = []
    for idxj in range(len(statesalt)):
        stat = np.sum(np.abs(barpi-statesalt[idxj]))
        samplesalt.append(stat)
        pval = 2*min(np.mean(samplesnull>=stat),np.mean(samplesnull<=stat))
        lspvalsnaive.append(pval)
    return lspvalsnaive


def pval_taylor(states,X,lamb,M,show_distributions=False,thetanull=None):
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
    n,p = X.shape
    # idxs = np.random.choice([i for i in range(len(states))], size=300, p=probas)
    lspvals_taylor = np.zeros((1,len(states)))
    lssamplesalt = []
    matXtrue = X[:,M]
    for ind in range(1):
        for idx in range(len(states)):
            y = np.array(states[idx])
            model = LogisticRegression(C = 1/lamb, penalty='l1', solver='liblinear', fit_intercept=False)
            model.fit(X, y)
            theta_obs = model.coef_[0]
            bhat = theta_obs[M]
            SM = np.sign(bhat)
            pihat = sigmoid(X @ theta_obs)
            MM = np.linalg.inv(matXtrue.T @ np.diag(pihat*(1-pihat)) @ matXtrue)
            bbar = bhat + lamb* MM @ SM
            lssamplesalt.append(bbar[ind])

            vlow, vup = low_up_taylor(theta_obs,SM,M,X,lamb,ind)
            gamma = np.zeros(len(M))
            gamma[ind] = 1
            if thetanull is None:
                thetanull = np.zeros(p)
            signull = sigmoid(X @ thetanull)
            samples = np.random.normal(thetanull[M[ind]],np.sqrt(gamma.T @ np.linalg.inv(matXtrue.T @ np.diag(signull * (1-signull)) @  matXtrue)@gamma),100000)
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

def pval_SIGLE(states, X, M, barpi, net=None, use_net_MLE=False, l2_regularization=10):
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
    n,p = X.shape
    matXtrue = X[:,M]
    if net is not None:
        rho = matXtrue.T @ barpi.T
        tildetheta = net(torch.from_numpy(rho.T).float())
        tildetheta = tildetheta.detach().numpy()
    else:
        model = LogitRegressionContinuous()
        model.fit( matXtrue, barpi)
        tildetheta = model.coef_
    tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
    usvd,s,vt = np.linalg.svd(tildeGN)
    tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt
    GNtilde = matXtrue.T @ np.diag(sigmoid1(matXtrue @ tildetheta)) @ matXtrue
    VN = tildeGN_12 @ GNtilde

    lspvals_selec = []
    lspvals_sat = []

    for i in range(len(states)):
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
        df = len(M)
        lspvals_selec.append(1-scipy.stats.chi2.cdf(stat, df))

        # saturated
        stat = np.linalg.norm( tildeGN_12 @ matXtrue.T @ (y-barpi))**2
        df = len(M)
        lspvals_sat.append(1-scipy.stats.chi2.cdf(stat, df))
    return lspvals_selec, lspvals_sat


def plot_cdf_pvalues(lists_pvalues, names, name_figsave=None):
    """Shows the cumulative distribution function of the p-values.
    
    Parameters
    ----------
    lists_pvalues : list of list of float
        each sublist contains p-values obtained from a specific method.
    names : list of string
        contains the names of the different methods used to compute the sublist of p-values from 'lists_pvalues'.
    """
    assert len(lists_pvalues)==len(names),'You should provide exactly one name for each list of p-values'
    for j in range(len(names)):
        lspvals = lists_pvalues[j]
        lspvals = np.sort(lspvals)
        lsseuil = np.linspace(0,1,100)
        PVALs = np.zeros(len(lsseuil))
        for i,t in enumerate(lsseuil):
            PVALs[i] = np.sum(lspvals<=t)/len(lspvals)

        plt.ylim(ymax = 1, ymin = 0)
        plt.xlim(xmax = 1, xmin = 0)
        plt.plot(lsseuil,PVALs,label=names[j])
    plt.plot([0,1],[0,1],'--')
    plt.legend(fontsize=13)
    plt.title('CDF of the p-values',fontsize=13)
    if name_figsave is not None:
        plt.savefig(name_figsave,dpi=300)