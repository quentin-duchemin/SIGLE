import numpy as np
from tqdm.notebook import tqdm
from ..tools import *
    
class Weak:
    """This weak learner is a two-sided test based on the statistic :math:`\sum_{i=1}^n |\overline \pi_i^{\\theta_0} - y_i|` where :math:`\overline \pi^{\\theta_0}` is the expectation of the vector of observations under the null conditional to the selection event.
    """
    def __init__(self):
        pass

    def pval_weak_learner(self, statesnull, statesalt, barpi, signull=None):
        """Computes the P-values obtained from the weak-learner which is a two-sided test based on the statistic :math:`\sum_{i=1}^n |\overline \pi^{\pi^0}_i-y_i|` where :math:`\overline \pi^{\pi^0}` is the expectation of the vector of observations under the null conditional to the selection event.

        Parameters
        ----------
        states : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').
        barpi : list of float
            Expectation of the vector of observations under the null conditional to the selection event.
        signull : vector in [0,1]^n
            Expectation of the response vector under the null. It is used when 'sampling_algorithm' is 'SA' (Simulated Annealing).
            
        Returns
        -------
        lspvalsnaive: samples of p-values obtained from the weak-learner.
        """
        lspvalsnaive = []
        if self.sampling_algorithm == 'SA':
            for idxj in tqdm(range(len(statesalt))):
                normalization = 0
                pvalsup, pvalinf = 0,0
                for i in range(len(statesnull)):
                    proba = compute_proba(statesnull[i],signull)
                    normalization += proba
                    statnull = np.sum(np.abs(barpi-statesnull[i]))
                    statalt = np.sum(np.abs(barpi-statesalt[idxj]))
                    if statnull==statalt:
                        if np.random.rand()<0.5:
                            pvalsup += proba
                        else:
                            pvalinf += proba
                    else:
                        pvalsup += proba * (statnull>statalt)
                        pvalinf += proba * (statnull<statalt) 
                        
                lspvalsnaive.append(2*min(pvalsup,pvalinf)/normalization)
        else:
            samplesnull = []
            for i in tqdm(range(len(statesnull))):
                samplesnull.append(np.sum(np.abs(barpi-statesnull[i])))
            samplesalt = []
            for idxj in tqdm(range(len(statesalt))):
                stat = np.sum(np.abs(barpi-statesalt[idxj]))
                samplesalt.append(stat)
                pval = 2*min(np.mean(samplesnull>=stat),np.mean(samplesnull<=stat))
                lspvalsnaive.append(pval)
        return lspvalsnaive