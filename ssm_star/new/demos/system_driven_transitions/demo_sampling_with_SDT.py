import autograd.numpy as np 

import ssm_star

from ssm_star.new.plotting import plot_sample

###
# Configs 
###

K_true = 2 
D_true = 3
N =  4
T = 200
L_true = 6  # currently must be even multiple of K.
seed = 10

np.random.seed(seed)

###
# Make system regimes, hard-coded 
###
import numpy as np 
SYSTEM_REGIMES_INDICES = np.tile(range(L_true), int(T))[:T]
from lds.util import one_hot_encoded_array_from_categorical_indices
SYSTEM_REGIMES_ONE_HOT = one_hot_encoded_array_from_categorical_indices(SYSTEM_REGIMES_INDICES, L_true)



###
# Sampling with strong lambda_ (influence of system-level regimes)
###
print("Sampling SLDS with system-driven transitions")
slds = ssm_star.SLDS(N, K_true, D_true, L=L_true, emissions="gaussian", transitions="system_driven")

# insight into system level regimes 
print(f"Xi, the KxL t.p.m governing how current system regime influences current entity regime is: \n {slds.transitions.Xis}")
print(f"lambda_, the strength of influence of system level regimes on transitions between entity regimes is: {slds.transitions.lambda_}")

z,x,y=slds.sample(T, system_input=SYSTEM_REGIMES_ONE_HOT)
plot_sample(x,y,z)

###
# Sampling with weak lambda_ (influence of system-level regimes)
###
print("Sampling SLDS with system-driven transitions")
slds = ssm_star.SLDS(N, K_true, D_true, emissions="gaussian", transitions="system_driven")
slds.transitions.lambda_=0.0
slds.transitions.Xis *= 0.0

# insight into system level regimes 
print(f"Xi, the KxL t.p.m governing how current system regime influences current entity regime is: \n {slds.transitions.Xis}")
print(f"lambda_, the strength of influence of system level regimes on transitions between entity regimes is: {slds.transitions.lambda_}")

z,x,y=slds.sample(T)
plot_sample(x,y,z)
