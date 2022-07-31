# PSILOGIT: a package to post-selection inference in the logistic model 

## A tool for post-selection inference in the logistic model

``PSILOGIT`` is a Python package that solves post-selection inference problems for the logistic model when model selection in performed using a :math:`\ell_1`-penalised likelihood approach. PSILOGIT is particularly well suited to tackle composite hypothesis testing problems. 

Currently, the package allows to solve post-selection inference problems using:

- SIGLE: a valid procedure for Selective Inference with the Generalized Linear Lasso. This is a new method based on a conditional MLE viewpoint to tackle selective inference in the logistic model. This method can be deployed for both the selected and the saturated model and is fully described in the following paper.


.. code-block:: bibtex

    @InProceedings{sigle2022,
      title     = {SIGLE: a valid procedure for Selective Inference with the Generalized Linear Lasso},
      author    = {Duchemin, Quentin and De Castro, Yohann},
      year      = {2022},
      url       = {https://hal.archives-ouvertes.fr/hal-03622196},
    }

- the method a Taylor & Tibshirani (see [their paper ](https://arxiv.org/abs/1602.07358)).


## Why ``PSILOGIT``?


``PSILOGIT`` is specially designed to address composite hypothesis testing problem after model selection using the non-zero coefficients obtained solving the :math:`\ell_1`-penalised logistic regression. 

Our experiments have shown that the method from Taylor & Tibshirani is most of the time correctly calibrated but the authors do not provide theoretical guarantees for their approach. Their method is motivated by non rigorous asymptotic considerations. In [our paper ](https://hal.archives-ouvertes.fr/hal-03622196), we are the fist to propose a method for selective inference in the logistic model with theoratical guarantees under some well defined conditions.

We show in [our paper ](https://hal.archives-ouvertes.fr/hal-03622196) that SIGLE seems always more powerful than the approach from [Taylor and Tibshirani ](https://arxiv.org/abs/1602.07358).


## Get started

Make sure to browse [``PSILOGIT`` documentation ](https://quentin-duchemin.github.io/SIGLE/) to learn more about its features.


