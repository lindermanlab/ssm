import autograd.numpy as np 

import ssm_star

from ssm_star.demos.generate import (
    generate_1dim_data_with_1dim_states,
    generate_3dim_data_with_2dim_states,
)

###
# Configs 
###
num_iters_laplace_em = 100
smart_initialize = True 
num_init_ar_hmms = 1

###
# Generate Data
####

#y, x_true, z_true = generate_1dim_data_with_1dim_states()
y, x_true, z_true = generate_3dim_data_with_2dim_states()

K_true = len(set(z_true))
D_true =  np.shape(x_true)[1]
N =  np.shape(y)[1]

# TODO: Stop hardcoding this stuff

###
# Inference 
###

print("Fitting SLDS using Linderman's SSM repo")
slds = ssm_star.SLDS(N, K_true, D_true, emissions="gaussian")

### Warning!  Linderman's initialization seems to assume that the obs dim exceeds the state dim!
# And if initialization is not done, results are very poor.
# See: https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/lds.py#L161
if smart_initialize:
    slds.initialize(y, num_init_restarts=num_init_ar_hmms)
q_elbos, q = slds.fit(
    y,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    initialize=False,
    num_iters=num_iters_laplace_em,
)
