from .Weak import Weak
from .Sigle import Sigle
from .Taylor import Taylor
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ..tools import *

class Methods(Sigle,Weak,Taylor):
    
    def __init__(self):
        super(Sigle, self).__init__()
        super(Weak, self).__init__()
        super(Taylor, self).__init__()
        
        
    def compute_power(self, lists_pvalues, names, alpha=0.05, states=None, sigalt=None):
        assert len(lists_pvalues)==len(names),'You should provide exactly one name for each list of p-values'
        PVALs = np.zeros(len(names))
        if self.sampling_algorithm == 'RS':
            for j in range(len(names)):
                lspvals = lists_pvalues[j]
                lspvals = np.sort(lspvals)
                PVALs[j] = np.mean(lspvals<=alpha)
        else:
            for j in range(len(names)):
                ls = []
                normalization = 0
                for i in range(len(states)):
                    proba = compute_proba(states[i], sigalt)
                    normalization += proba
                    PVALs[j] += proba * (lists_pvalues[j][i]<=alpha)
                PVALs /= normalization
        return PVALs

    def plot_cdf_pvalues(self, lists_pvalues, names, states=None, sigalt=None, legend_outfig=False, figname=None):
        """Shows the cumulative distribution function of the p-values.

        Parameters
        ----------
        lists_pvalues : list of list of float
            each sublist contains p-values obtained from a specific method.
        names : list of string
            contains the names of the different methods used to compute the sublist of p-values from 'lists_pvalues'.
        """
        assert len(lists_pvalues)==len(names),'You should provide exactly one name for each list of p-values'
        lsseuil = np.linspace(0,1,100)
        if self.sampling_algorithm == 'RS':
            for j in range(len(names)):
                lspvals = lists_pvalues[j]
                lspvals = np.sort(lspvals)
                PVALs = np.zeros(len(lsseuil))
                for i,t in enumerate(lsseuil):
                    PVALs[i] = np.sum(lspvals<=t)/len(lspvals)
                plt.ylim(ymax = 1, ymin = 0)
                plt.xlim(xmax = 1, xmin = 0)
                plt.plot(lsseuil,PVALs,label=names[j])
        else:
            for j in range(len(names)):
                ls = []
                normalization = 0
                PVALs = np.zeros(len(lsseuil))
                for i in range(len(states)):
                    proba = compute_proba(states[i], sigalt)
                    normalization += proba
                    for l,t in enumerate(lsseuil):
                        PVALs[l] += proba * (lists_pvalues[j][i]<=t)
                PVALs /= normalization
                plt.ylim(ymax = 1, ymin = 0)
                plt.xlim(xmax = 1, xmin = 0)
                plt.plot(lsseuil,PVALs,label=names[j])
                    
        plt.plot([0,1],[0,1],'--')
        if legend_outfig:
            plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.01+0.13*len(lists_pvalues)))
        else:
            plt.legend(fontsize=13)
        plt.title('CDF of the p-values',fontsize=13)
        if figname is not None:
            plt.savefig(figname,dpi=300)
            