import autograd.numpy as np 

import ssm_star

from ssm_star.new.generate import (
    generate_multi_dim_data_with_multi_dim_states_and_two_regimes
)

"""
SDT = "System Driven Transitions"

This demo just surfaces the parameters at three points:
1) After default initialization
2) After smart initialization
3) After inference

We can inspect whether/how the parameters change. This is especially
important for the new parameters introduced by the system-level module.
"""

###
# Configs 
###

# Data generation
L_true = 6 # currently must be even multiple of K 
K_true = 2 
D_true = 3 # state_dim 
N =  4 # obs_dim 
T = 200
seed = 10

# Inference 
num_iters_laplace_em = 100
smart_initialize = True 
num_init_ar_hmms = 1



###
# Make system regimes, hard-coded 
###
import numpy as np 
SYSTEM_REGIMES_INDICES = np.tile(range(L_true), int(T))[:T]
from lds.util import one_hot_encoded_array_from_categorical_indices
SYSTEM_REGIMES_ONE_HOT = one_hot_encoded_array_from_categorical_indices(SYSTEM_REGIMES_INDICES, L_true)



###
# Generate Data
####

# TODO: allow data generation with system inputs as well

y, x_true, z_true = generate_multi_dim_data_with_multi_dim_states_and_two_regimes(N, D_true)

###
# Inference 
###

print("Fitting SLDS using Linderman's SSM repo")
slds = ssm_star.SLDS(N, K_true, D_true, L=L_true, emissions="gaussian", transitions="system_driven")

# get default initializations
e1, t1, d1 = slds.emissions.params, slds.transitions.params, slds.dynamics.params 

### Warning!  Linderman's initialization seems to assume that the obs dim exceeds the state dim!
# And if initialization is not done, results are very poor.
# See: https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/lds.py#L161
if smart_initialize:
    slds.initialize(y, num_init_restarts=num_init_ar_hmms, system_inputs = SYSTEM_REGIMES_ONE_HOT)

# get smart initializations
e2, t2, d2 = slds.emissions.params, slds.transitions.params, slds.dynamics.params 


q_elbos, q = slds.fit(
    y,
    system_inputs = SYSTEM_REGIMES_ONE_HOT,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    initialize=False,
    num_iters=num_iters_laplace_em,
)

# get final values 
e3, t3, d3 = slds.emissions.params, slds.transitions.params, slds.dynamics.params 

