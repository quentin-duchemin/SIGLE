"""
==========================================
Sampling states from the SEI-SLR algorithm
==========================================
The example runs the SEI-SLR algorithm to generate states uniformly distributed on the selection event.
"""

import PSILOGIT
import numpy as np
from PSILOGIT.tools import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt

# %%
# Model built with a randomly generated dataset
n = 15
p = 10
theta = np.zeros(p)
lamb = 2
model = PSILOGIT.PSILOGIT(truetheta=theta, regularization=lamb,  n=n, p=p)


# %%
# We run the SEI-SLR algorithm.
states, ls_FNR, energies = model.SEI_SLR(total_length_SEISLR_path=100000, backup_start_time=2000, temperature=linear_temperature, repulsing_force=True, random_start=True, conditioning_signs=False, seed=0)


# %%
# The size of the observed vector is small enough to compute exactly the states belonging to the selection event via an exhaustive search.
nbM_admissibles, ls_states_admissibles = model.compute_selection_event(compare_with_energy=True)
print('Size of the selection event: ', len(ls_states_admissibles))



# %%
# By running over the~$2^{10}$ possible vectors :math:`Y \in \{ 0,1\}^n`, we have previously computed exactly the selection event.
# In the following, we identify each vector :math:`Y \in \{0, 1\}^N` with the number between~$0$ and~$2^N-1=1024$ that it represents in the base-2 numeral system.
# The following figure shows the last :math:`500,000` visited states for our simulated annealing path. On the vertical axis, we have the integers encoded by all possible vectors :math:`Y \in \{0,1\}^N`. 
# The red dashed lines represent the states that belong to the selection event :math:`E_M`. While crosses are showing the visited states on the last :math:`500,000` time steps of the path, green crosses are emphasizing the ones that belong to the selection event.
model.last_visited_states(states, ls_states_admissibles)
plt.show()

# %%
# The following figure shows that the SEI-SLR algorithm covers properly the selection event without being stuck in one specific state of :math:`E_{M}`. 
#The simulated annealing path is jumping from one state of :math:`E_{M}` to another, ending up with an asymptotic distribution of the visited states that approximates the uniform distribution on :math:`E_{M}`.
model.histo_time_in_selection_event(states, ls_states_admissibles, rotation_angle=80)
plt.show()