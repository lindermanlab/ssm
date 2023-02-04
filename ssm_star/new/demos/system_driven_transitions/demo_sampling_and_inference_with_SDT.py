import autograd.numpy as np
from lds.util import one_hot_encoded_array_from_categorical_indices

import ssm_star
from ssm_star.new import config_util
from ssm_star.new.generate import generate_regime_sequences_with_runs
from ssm_star.new.plotting import plot_elbos, plot_results_for_one_entity, plot_sample


"""
SDT = "System Driven Transitions"

We consider two scenarios - when the system DOES and DOES NOT drive entity-level regime transitions.

We check that we can 
1) successfully sample
2) successful do inference 

in each of these two scenarios.
"""

path_to_config = "configs/configs_2.yaml"
CFG = config_util.load(path_to_config)

print(f"\n---{CFG.summary_of_run}---\n")

###
# Make system regimes, hard-coded
###
SYSTEM_REGIMES_INDICES = generate_regime_sequences_with_runs(
    CFG.T, CFG.L_true, run_length=CFG.fixed_run_length_for_system_level_regimes
)
SYSTEM_REGIMES_ONE_HOT = one_hot_encoded_array_from_categorical_indices(
    SYSTEM_REGIMES_INDICES, CFG.L_true
)


###
# Generate Data
####

np.random.seed(CFG.seed)

print(f"Sampling SLDS ...")
slds_for_generation = ssm_star.SLDS(
    CFG.N,
    CFG.K_true,
    CFG.D_true,
    L=CFG.L_true,
    emissions="gaussian",
    transitions="system_driven",
)


if CFG.fixed_run_length_for_entity_level_regimes != 0:
    print(f"...with fixed cyclic regimes.")
    fixed_z = generate_regime_sequences_with_runs(
        CFG.T, CFG.K_true, run_length=CFG.fixed_run_length_for_entity_level_regimes
    )
    z_true, x_true, y = slds_for_generation.sample_with_fixed_z(fixed_z=fixed_z)
else:
    print(
        f"...with system-driven regime transitions. System influence scalar: {CFG.system_influence_scalar:.02f}"
    )
    slds_for_generation.transitions.Xis *= CFG.system_influence_scalar
    print(
        f"Xi, the KxL t.p.m governing how current system regime influences current entity regime is: \n {slds_for_generation.transitions.Xis}"
    )
    z_true, x_true, y = slds_for_generation.sample(
        CFG.T, system_input=SYSTEM_REGIMES_ONE_HOT
    )

plot_sample(x_true, y, z_true)


###
# Inference
###

print("Fitting SLDS.")
slds = ssm_star.SLDS(
    CFG.N,
    CFG.K_true,
    CFG.D_true,
    L=CFG.L_true,
    emissions="gaussian",
    transitions="system_driven",
    transition_kwargs=dict(alpha=CFG.alpha, kappa=CFG.kappa),
)


### Warning!  Linderman's initialization seems to assume that the obs dim exceeds the state dim!
# And if initialization is not done, results are very poor.
# See: https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/lds.py#L161
if CFG.smart_initialize:
    slds.initialize(
        y, num_init_restarts=CFG.num_init_ar_hmms, system_inputs=SYSTEM_REGIMES_ONE_HOT
    )


q_elbos, q = slds.fit(
    y,
    system_inputs=SYSTEM_REGIMES_ONE_HOT,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    initialize=False,
    num_iters=CFG.num_iters_laplace_em,
)

###
# Postmortem
###
plot_elbos(q_elbos)
plot_results_for_one_entity(
    q, slds, y, x_true, z_true, system_input=SYSTEM_REGIMES_ONE_HOT
)
