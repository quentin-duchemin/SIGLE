PSILOGIT
=========


A tool for post-selection inference in the logistic model
---------------------------------------------------------

``PSILOGIT`` is a Python package that solves post-selection inference problems for the logistic model when model selection in performed using a $\ell_1$-penalised likelihood approach. PSILOGIT is particularly well suited to tackle composite hypothesis testing problems. 

Currently, the package handles the following problems:

.. list-table:: The supported lasso-like problems
   :header-rows: 1

   * - Problem
     - Support of weights
     - Native cross-validation
   * - Lasso
     - ✓
     - ✓
   * - ElasticNet 
     - ✓
     - ✓
   * - Group Lasso 
     - ✓
     - ✓
   * - Multitask Lasso
     - ✕
     - ✓
   * - Sparse Logistic regression
     - ✕
     - ✕


Why ``celer``?
--------------

``celer`` is specially designed to handle Lasso-like problems which enable it to solve them quickly.
``celer`` comes particularly with

- automated parallel cross-validation
- support of sparse and dense data
- optional feature centering and normalization
- unpenalized intercept fitting

``celer`` also provides easy-to-use estimators as it is designed under the ``scikit-learn`` API.


Install ``PSILOGIT``
--------------------

``PSILOGIT`` can be easily installed through the Python package manager ``pip``.
To get the laster version of the package, run::

    $ pip install -U PSILOGIT

Head directly to the :ref:`Get started` page to get a hands-on example of how to use ``celer``.



Cite
----

``PSILOGIT`` is an open source package licensed under 
the `BSD 3-Clause License <https://github.com/quentin-duchemin/celer/blob/main/LICENSE>`_.
Hence, you are free to use it. And if you do so, do not forget to cite:


.. code-block:: bibtex

    @InProceedings{pmlr-v80-massias18a,
      title     = {SIGLE: a valid procedure for Selective Inference with the Generalized Linear Lasso},
      author    = {Duchemin, Quentin and De Castro, Yohann},
      year      = {2022},
    }

Explore the documentation
-------------------------

.. toctree::
    :maxdepth: 1

    get_started.rst
    api.rst
    auto_examples/index.rst