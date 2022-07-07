from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from ..inverse_map import inverse_map, train_network
from ..tools import *
import matplotlib
import scipy
import matplotlib.pyplot as plt

class FiguresSigle:
    """Class allowing to plot figures analyzing the Sigle procedure and the SEI-SLR algorithm.
    """
    
    def __init__(self):
        pass

    def ellipse_testing(self, states, barpi, signull=None, alpha=0.05, figname=None, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':100}, l2_regularization=100000):
        """For a selected support of size 2, this method show the proportion of stats following in the ellipse characterizing the rejection rejection of the hypothesis test with SIGLE in the selected model.
        
        Parameters
        ----------
        states : list of vectors in {0,1}^n
            'states' should contain binary vectors sampled either from the uniform distribution on the selection event (in which case the attribute 'sampling_algorithm' is equal to 'SA') or sampled from the conditional distribution (in which cas the attribute 'sampling_algorithm' is equal to 'RS').
        barpi : list of float
            expectation of the vector of observations under the null conditional to the selection event.
        signull : vector in [0,1]^n
            Expectation of the response vector under the null. 
            
            
            
            
            
            
            
            It is used when we 'calibrated_from_samples' is True and when 'sampling_algorithm' is 'SA' (Simulated Annealing).
        
        
        
        
        alpha : float
            Level of the test.
        figname : bool
            If not None, the figure will be saved with the path 'figname'.
        grad_descent : dictionary
            Should be of the form {'lr':0.01,'return_gaps':True,'max_ite':100} and it is used for the method 'compute_theta_bar'. 
            - 'lr' is the step size for the gradient descent algorithm.
            - 'return_gaps' is set to True if one wants to return at the end of the algorithm the list of $\| X_M^{\top} (\sigma(X_M \bar \theta^{(t)})- \bar \pi) \|_{\infty}$ for the iterates $\theta^{(t)}$ of the gradient descent algorithm.
        l2_regularization : float
            l2 regularization used to compute the MLE from the solver of Logistic Regression in sk-learn. It should be as large as possible to remove regularization.
        """
        if len(self.M)!=2:
            print('The selected support should be of size 2 for 2D visualization.')
        else:
            from math import pi, cos, sin
            import matplotlib
            matplotlib.rcParams.update({'font.size': 12})
            n,p = np.shape(self.X)
            matXtrue = self.X[:,self.M]

            tildetheta, gaps = self.compute_theta_bar(barpi, grad_descent=grad_descent)
            u=tildetheta[0]       #x-position of the center
            v=tildetheta[1]       #y-position of the center 
            tildeGN = matXtrue.T @ np.diag(barpi*(np.ones(n)-barpi)) @ matXtrue
            usvd,s,vt = np.linalg.svd(tildeGN)
            tildeGN_12 = usvd @ np.diag(1/np.sqrt(s)) @ vt
            GNtilde = matXtrue.T @ np.diag(sigmoid1(matXtrue @ tildetheta)) @ matXtrue
            VN = tildeGN_12 @ GNtilde

            usvd,s,vt = np.linalg.svd(VN)
            df = matXtrue.shape[1]
            quantile_chi2 = scipy.stats.chi2.ppf(1-alpha, df)
            width = np.sqrt(quantile_chi2 / s[1]**2)
            height = np.sqrt(quantile_chi2 / s[0]**2)
            angle = np.arccos(np.vdot(np.array([1,0]),vt[1,:]))

            a=width       #radius on the x-axis
            b=height      #radius on the y-axis
            t_rot=angle #rotation angle

            t = np.linspace(0, 2*pi, 100)
            Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
                 #u,v removed to keep the same center location
            R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
                 #2-D rotation matrix
            Ell_rot = np.zeros((2,Ell.shape[1]))
            for i in range(Ell.shape[1]):
                Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'darkorange',linewidth=3 )    #rotated ellipse
            plt.grid(color='lightgray',linestyle='--')

            lsxin = []
            lsyin = []
            lsxout = []
            lsyout = []
            sizein = []
            sizeout = []

            for i,y in enumerate(states):
                model = LogisticRegression(C=l2_regularization, solver='liblinear', fit_intercept=False)
                model.fit(matXtrue, y)
                hattheta = model.coef_[0]
                a = hattheta[0]
                b = hattheta[1]
                if self.sampling_algorithm == 'SA':
                    proba = compute_proba(y, signull)
                if np.linalg.norm( VN @ (np.array([a,b]) - tildetheta))**2<=quantile_chi2:
                    lsxin.append(a)
                    lsyin.append(b)
                    if self.sampling_algorithm == 'SA':
                        sizein.append(proba)
                else:
                    lsxout.append(a)
                    lsyout.append(b)
                    if self.sampling_algorithm == 'SA':
                        sizeout.append(proba)
            if self.sampling_algorithm == 'RS':
                plt.scatter(lsxin,lsyin,c='green',marker='+',label=str(int(100*len(lsxin)/(len(lsxin)+len(lsxout))))+' % in the ellipse')
                plt.scatter(lsxout,lsyout,c='red',marker='x',label=str(int(100*len(lsxout)/(len(lsxin)+len(lsxout))))+' % outside of the ellipse')
            else:
                plt.scatter(lsxin,lsyin,c='green',marker='+',s=sizein, label=str(int(100*len(lsxin)/(len(lsxin)+len(lsxout))))+' % in the ellipse')
                plt.scatter(lsxout,lsyout,c='red',marker='x',s=sizeout, label=str(int(100*len(lsxout)/(len(lsxin)+len(lsxout))))+' % outside of the ellipse')

            plt.legend(fontsize=13)

            print('Proportion in the ellipse : ', len(lsxin)/(len(lsxin)+len(lsxout)))
            print('Proportion outside of the ellipse : ', len(lsxout)/(len(lsxin)+len(lsxout)))
            if figname is not None:
                plt.savefig(name_figsave,dpi=300)
            plt.show()
            

    def compute_selection_event(self, conditioning_signs=False, compare_with_energy=False, delta=0.009):
        """Finds all the vectors belonging to the selection event.

        Parameters
        ----------
        conditioning_signs : bool
            If True, we consider that the selection event corresponds to the states allowing to recover both the selected support (i.e. the none zero entries of thete_obs) and the vector of signs (i.e. the correct signs of the none zero entries of theta_obs).
        compare_with_energy : bool
            If True, we show a warning when some state wuold have been not correctly classified (as in or out of the selection event) based on the energy.
        delta : float
            Parameter allowing to define the energy (corresponding of the simulated annealing algorithm).

        Returns
        -------
        nbM_admissibles : integers encoding all the vectors of the hypercube belonging to the selection event. This encoding is performed using the function 'binary_encoding'. 
        ls_states_admissibles : vectors of the hypercube belonging to the selection event.
        """
        def b(x):
            return (1-np.sqrt(np.min([-x/delta, 1])))
        n,p = np.shape(self.X)
        assert(n<20)
        M = np.where( np.abs(self.theta_obs) > 1e-5)[0]
        SMY = np.sign(self.theta_obs[self.M])
        SM = np.sign(self.theta_obs)[self.M] 
        y = np.copy(self.yobs)
        d = len(self.M)
        matX = self.X[:,self.M]
        complement_XT = self.X[:,[k for k in range(p) if k not in self.M]].T
        complementM = [ i for i in range(p) if i not in self.M]
        y = np.ones(n)
        counter = 0
        lsE = []
        ls_states_admissibles = []
        for k in range(2**n):
            const = k
            for i in range(len(self.yobs)):
                y[i] = int(const%2)
                const -= const%2
                const /= 2
            if np.sum(y) in [len(y),0]:
                pass
            else:
                model = LogisticRegression(penalty='l1', C = 1/self.lamb, solver='liblinear', fit_intercept=False)
                model.fit(self.X, y)
                theta_hat = model.coef_[0]
                sig_hat = sigmoid( self.X @ theta_hat)
                S = np.clip(self.X.T @ (y-sig_hat) / self.lamb, -1, 1)

                # We compute the energy
                if conditioning_signs:
                    p1y = np.sum(1 - S[self.M]*self.SM)/len(self.M)
                else:
                    p1y = np.sum(1 - np.abs(S[self.M]))/len(self.M)
                p2y = b(np.max(np.abs(S[complementM]))-1)
                E = (np.sum([0, p1y, p2y]))
                lsE.append(E)

                if (np.max(np.abs(theta_hat[complementM]))<1e-5) and (np.min(np.abs(theta_hat[self.M])) > 1e-5):
                    counter += 1
                    ls_states_admissibles.append(list(y))
                    if compare_with_energy and (p2y>1e-3 and p1y>1e-3):
                        print('Warning: a state would have been uncorrectly classified using the energy',p2y)
                else:
                    if compare_with_energy and (p2y<1e-3 and p1y<1e-3):
                        print('Warning: a state would have been uncorrectly classified using the energy',p2y )

        nbM_admissibles = np.sort(list(set(map(binary_encoding,ls_states_admissibles))))
        return nbM_admissibles, ls_states_admissibles


    def histo_time_in_selection_event(self, states, ls_states_admissibles, figname=None):
        """Histogram showing the time spent in the selection event using the SEI-SLR algorithm.

        Parameters
        ----------
        states : list of vectors belonging to {0,1}^n
            Sequence of states obtained using the SEI-SLR algorithm.
        ls_states_admissibles : list of list
            Vectors of the hypercube belonging to the selection event.
        """
        selectionevent = [binary_encoding(ya) for ya in ls_states_admissibles]
        state2time = np.zeros(len(selectionevent)+1)
        last_fy = [binary_encoding(y) for y in states]
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
        plt.ylabel('Proposition of time spent in $E_{M}$',fontsize=12)
        plt.xticks(fontsize=13)
        if figname is not None:
            plt.savefig(figname,dpi=250)
    
    def last_visited_states(self, states, lsM_admissibles, figname=None):
        """Shows that time spent in the selection event using the SEI-SLR algorithm.

        Parameters
        ----------
        states : list of vectors belonging to {0,1}^n
            Sequence of states obtained using the SEI-SLR algorithm.
        lsM_admissibles : list of integers
            Each entry encodes some vector of bits (using the function 'binary_encoding').
        """
        last_fy = [binary_encoding(y) for y in states]
        n = (self.X).shape[0]
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
        if figname is not None:
            plt.savefig(figname,dpi=250)
    
    def time_in_selection_event(self, states, ls_states_admissibles, fig_name=None):
        """Shows that time spent in the selection event using the SEI-SLR algorithm for several different excursions.

        Parameters
        ----------
        states : list of vectors belonging to {0,1}^n
            Sequence of states obtained using the SEI-SLR algorithm.
        ls_states_admissibles : list of list
            All vectors of the hypercube belonging to the selection event.
        """
        proportions = np.zeros(len(indexes))
        last_fy = [binary_encoding(y) for y in states]
        selectionevent = [binary_encoding(ya) for ya in ls_states_admissibles]
        inselectionevent = np.zeros(len(last_fy))
        for i,fy in enumerate(last_fy):
            if fy in selectionevent:
                inselectionevent[i] = 1
        proportions[k] = (np.sum(inselectionevent)/len(inselectionevent))
        plt.scatter(np.array([i for i in range(len(proportions))]), np.sort(proportions))
        plt.xlabel('Index of the simulation', fontsize=12)
        plt.ylabel('Proportion of time spent in $E_{M_0}$', fontsize=12)
        if figname is not None:
            plt.savefig(figname,dpi=250)
    
