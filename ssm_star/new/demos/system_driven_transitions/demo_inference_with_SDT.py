import autograd.numpy as np 

import ssm_star

from ssm_star.new.generate import (
    generate_multi_dim_data_with_multi_dim_states_and_two_regimes
)
from ssm_star.new.plotting import (
    plot_sample,
    plot_elbos,
    plot_results_for_one_entity,
)

"""
SDT = "System Driven Transitions"
he

We check on the quality of inference.
"""

###
# Configs 
###

# Data generation
L_true = 6 # currently must be even multiple of K 
K_true = 2 # TODO: extract this from generative mechanism; even better would be to update generator to be flexible to this.
D_true = 3 # state_dim 
N =  4 # obs_dim 
T = 200
seed = 10
observed_time_series_is_influenced_by_system = True 
lambda_=100.0 # strength of influence of system on observed time series.
 
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

if observed_time_series_is_influenced_by_system:
    print("Sampling SLDS with system-driven transitions")
    slds_for_generation = ssm_star.SLDS(N, K_true, D_true, L=L_true, emissions="gaussian", transitions="system_driven")
    slds_for_generation.transitions.Xis *= lambda_

    # insight into system level regimes 
    print(f"lambda_, the strength of influence of system level regimes on transitions between entity regimes is: {lambda_}")
    print(f"Xi, the KxL t.p.m governing how current system regime influences current entity regime is: \n {slds.transitions.Xis}")
    z_true, x_true, y = slds_for_generation.sample(T, system_input=SYSTEM_REGIMES_ONE_HOT)
    plot_sample(x_true, y, z_true)
else:
    # TODO: allow data generation with system inputs as well
    print("Generating data from a SLDS with no-system driven transitions")
    y, x_true, z_true = generate_multi_dim_data_with_multi_dim_states_and_two_regimes(N, D_true)
    plot_sample(x_true,y,z_true)


###
# Inference 
###

print("Fitting SLDS.")
slds = ssm_star.SLDS(N, K_true, D_true, L=L_true, emissions="gaussian", transitions="system_driven")

### Warning!  Linderman's initialization seems to assume that the obs dim exceeds the state dim!
# And if initialization is not done, results are very poor.
# See: https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/lds.py#L161
if smart_initialize:
    slds.initialize(y, num_init_restarts=num_init_ar_hmms, system_inputs = SYSTEM_REGIMES_ONE_HOT)


q_elbos, q = slds.fit(
    y,
    system_inputs = SYSTEM_REGIMES_ONE_HOT,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    initialize=False,
    num_iters=num_iters_laplace_em,
)

###
# Postmortem
###
plot_elbos(q_elbos)
plot_results_for_one_entity(
    q, slds, y, x_true, z_true, system_input = SYSTEM_REGIMES_ONE_HOT
)
