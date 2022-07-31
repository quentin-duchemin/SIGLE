===========
Get started
===========

In this starter example, we will toy datasets. 

Beforehand, make sure to install ``PSILOGIT``.


Generate a toy dataset
----------------------

``PSILOGIT`` allows to consider toy examples with randomly generated parameters or to specify by hand the parameters of the model.

1. **Toy example with randomly generated parameters**

The following code will create an instance of the PSILOGIT class using random generators.
We specify the true parameter vector :math:`\vartheta \in \mathbb R^p` and the regularization parameter :math:`\lambda` for the :math:`\ell_1`-penalty.

.. code-block:: python

    # imports
    import PSILOGIT
    import numpy as np
    from PSILOGIT.tools import *

    # model built with a randomly generated dataset
    n = 20
    p = 10
    theta = np.ones(p)
    lamb = 1
    model = PSILOGIT.PSILOGIT(truetheta=theta, regularization=lamb,  n=n, p=p)



2. **Specifying the parameters of the model**

The following code will create an instance of the PSILOGIT class using specified parameter.

2.1. Specifying the design matrix :math:`X`, the selected support :math:`M` and the vector of signs :math:`S_M`

.. code-block:: python

    # imports
    import PSILOGIT
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from PSILOGIT.tools import *

    # model built with specified parameters
    lamb = 1
    n,p = 20,10
    X = np.random.normal(0,5, (n,p))
    LR = LogisticRegression(C = 1/lamb, penalty='l1', solver='liblinear', fit_intercept=False)
    yobs = np.random.rand(n) <= 0.5 * np.ones(n)
    LR.fit(X, yobs)
    theta_obs = LR.coef_[0]
    M = np.where( np.abs(theta_obs) > 1e-5)[0]
    SM = np.sign(self.theta_obs[M])
    model = PSILOGIT.PSILOGIT(regularizaition=lamb, X=X, M=M, SM=SM)


2.2. Specifying only the design matrix :math:`X`.

One needs to specify the true parameter vector :math:`\vartheta`.

.. code-block:: python

    # imports
    import PSILOGIT
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from PSILOGIT.tools import *

    # model built with a specified design and parameter vector
    lamb = 1
    n,p = 20,10
    truetheta = np.ones(p)
    X = np.random.normal(0,5, (n,p))
    model = PSILOGIT.PSILOGIT(regularizaition=lamb, X=X, truetheta=truetheta)


Note that to allow reproducibility of the experiments, the random seed is fixed to :math:`1` and can be modified using the optional parameter 'seed' of the function ``PSILOGIT.PSILOGIT``.


Sampling states from the selection event
----------------------------------------

.. In our paper, we present conditional central limit theorems. These theoretical results hold under some conditions on the parameter of the problem considered. We found that these conditions hold under small dependency (typically when the selected support is large). 

In order to use the SIGLE procedures, one needs to sample states from the distribution of observations conditional to the selection event. In our paper, we proposed two different approaches to achieve this goal:

- a simple rejection sampling method,

- the SEI-SLR algorithm.

 Depending on the context, it is more convenient to use one method or the other. In this short tutorial, the number of predictors :math:`p=10` is small enough so that we can use the rejection sampling method.


Hypothesis testing
------------------

We consider the following composite hypothesis testing problem

.. math:: \mathbb H_0 : \; \theta^*=\theta_0^* \quad   VS   \quad \mathbb H_1 : \; \theta^*\neq \theta_0^*.

The following code allows to obtain the p-value associated to

- SIGLE in the selected and the saturated models

- Bonferroni method (TT-Bonferroni) and simple method (TT-1): from Taylor & Tibshirani 

- a weak learner

.. code-block:: python
    
    # Example of vector of observations for which we want to provide a p-value
    observed_state = model.yobs
    
    # Sampling under the null
    thetanull = np.zeros(model.X.shape[1])

    signull = sigmoid(model.X @ thetanull)
    statesnull = model.SEI_by_sampling(signull, nb_ite=100000)
    tildeGN12, barpi = model.params_saturated(signull, statesnull)
    SIGLEselec, SIGLEsat, gaps = model.pval_SIGLE([observed_state], barpi, l2_regularization=100000, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':10000}, 
                                                        calibrated_from_samples=False, statesnull=statesnull)
    gamma = np.zeros(len(model.M))
    gamma[0] = 1
    TT1 = model.pval_taylor([observed_state], thetanull=thetanull, gamma=gamma)
    TBon = model.pval_taylor([observed_state], thetanull=thetanull, mode='Bonferroni')
    Weak = model.pval_weak_learner(statesnull, [observed_state], barpi, signull=signull)
    lists_pvalues = [Weak, TT1, TBon, SIGLEselec, SIGLEsat]
    names = ['Weak learner', "TT-1", 'TT-Bonferroni', 'SIGLE Selected', 'SIGLE Saturated']
    plt.scatter([j for j in range(len(names))], [pval[0] for pval in lists_pvalues])
    plt.xticks([j for j in range(len(names))], names, rotation='vertical')
    plt.ylabel('p-value')


Further links
-------------

This was just a starter example. Get familiar with ``PSILOGIT`` by browsing its :ref:`API documentation` or
explore the :ref:`Examples Gallery`, which includes examples on real-life datasets as well as 
timing comparison with other solvers.