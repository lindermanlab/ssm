import autograd.numpy as np 

import ssm_star

from ssm_star.new.generate import (
    generate_multi_dim_data_with_multi_dim_states_and_two_regimes
)

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

###
# Postmortem
###

from lds.piecewise.metrics import compute_regime_labeling_accuracy

# TODO: Make `method` and `variational_posterior` arguments to the function
# TODO: Relabel q_x, q_y, q_z as expectations.
# TODO: Figure out what all the entries of q_EZ are.
q_Ez, q_x = q.mean[0]
q_y = slds.smooth(q_x, y)
z_most_likely = slds.most_likely_states(q_x, y)
pct_correct_regimes = compute_regime_labeling_accuracy(z_most_likely, z_true)
print(f"\n Pct correct segmentations: {pct_correct_regimes:.02f}.")


# Plot the true and inferred states
plt.figure(figsize=(8, 9))

plt.subplot(411)
plt.imshow(z_true[None, :], aspect="auto")
plt.imshow(np.row_stack((z_true, z_most_likely)), aspect="auto")
plt.yticks([0, 1], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{est}}}}$"])
plt.title("True and Most Likely Inferred States")
