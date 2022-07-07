import numpy as np
from tqdm.notebook import tqdm
from ..tools import *
    
class Weak:
    
    def __init__(self):
        pass

    def pval_weak_learner(self, statesnull, statesalt, barpi, signull=None):
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