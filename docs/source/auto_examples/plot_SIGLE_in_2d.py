"""
============================================
Visualization of SIGLE in the selected model
============================================
The example shows the way SIGLE works for a composite hypothesis testing problem where working with the global null.
"""

import PSILOGIT
import numpy as np
from PSILOGIT.tools import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt


# %%
# We will consider toy examples for which the selected support is of size :math:`2`. This will allow us to visualize the way SIGLE works in the plane.


# %%
# SIGLE in the selected model under the global null
# -------------------------------------------------
# 
# We work first under the global null and we show that SIGLE is correctly calibrated. This was expected since we calibrate SIGLE by sampling under the null using the rejection sampling method.
model_size2 = PSILOGIT.PSILOGIT(truetheta=0*np.ones(5), regularization=7, n=100, p=5)
print('Size of the selection event: ', len(model_size2.M))
signull = 0.5 * np.ones(model_size2.X.shape[0])
states2 = model_size2.SEI_by_sampling(model_size2.sig, nb_ite=100000)
print(len(states2))
statesnull = model_size2.SEI_by_sampling(signull, nb_ite=100000)
tildeGN12, barpi = model_size2.params_saturated(signull, statesnull)
model_size2.ellipse_testing(states2, barpi, alpha=0.05, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':5000}, calibrated_from_samples=True, statesnull=statesnull, l2_regularization=100000)


# %%
# SIGLE in the selected model with a localized alternative
# --------------------------------------------------------
# 
# We work under the localized alternative :math:`\\vartheta^*=[1 ,0,0,\dots,0] \in \mathbb R^{10}`.
theta = np.zeros(10)
theta[0] = 0.5
model_size2 = PSILOGIT.PSILOGIT(truetheta=theta, regularization=8, n=100, p=len(theta))
print('Size of the selection event: ', len(model_size2.M))
states2 = model_size2.SEI_by_sampling(model_size2.sig, nb_ite=100000)
print(len(states2))
statesnull = model_size2.SEI_by_sampling(signull, nb_ite=100000)
tildeGN12, barpi = model_size2.params_saturated(signull, statesnull)
model_size2.ellipse_testing(states2, barpi, alpha=0.05, grad_descent={'lr':0.01,'return_gaps':True,'max_ite':5000}, calibrated_from_samples=True, statesnull=statesnull, l2_regularization=100000)