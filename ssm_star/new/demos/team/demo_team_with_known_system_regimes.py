"""
We demo (known)-system level regimes driving rSLDS's
for a collection of entity-level time series.  Each 
entity has unique rSLDS models, but the same driving
system-level regimes. 
"""

import autograd.numpy as np
from lds.util import one_hot_encoded_array_from_categorical_indices

import ssm_star
from ssm_star.new import config_util
from ssm_star.new.generate import generate_regime_sequences_with_runs
from ssm_star.new.plotting import plot_elbos, plot_results_for_one_entity, plot_sample


path_to_config = "configs/configs_4.yaml"
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

from ssm_star.new.team import EntityData


# TODO: Make this a standalone function?
np.random.seed(CFG.seed)

N_ENTITIES = 2
team_data = [None] * N_ENTITIES

for j in range(N_ENTITIES):
    print(f"Sampling SLDS ...")
    slds_for_generation = ssm_star.SLDS(
        CFG.N,
        CFG.K_true,
        CFG.D_true,
        L=CFG.L_true,
        emissions="gaussian",
        transitions="system_driven",
    )

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

    team_data[j] = EntityData(j, y, x_true, z_true)
    plot_sample(x_true, y, z_true)


###
#  Construct team model
###
from ssm_star.new.team import TeamSLDS


team_slds = TeamSLDS(
    N_ENTITIES,
    CFG.N,
    CFG.K_true,
    CFG.D_true,
    L=CFG.L_true,
    emissions="gaussian",
    transitions="system_driven",
    transition_kwargs=dict(alpha=CFG.alpha, kappa=CFG.kappa),
)

###
# Smart Initialization
###
y_by_entity = [entity_data.y for entity_data in team_data]
team_slds.smart_initialize(y_by_entity, SYSTEM_REGIMES_ONE_HOT, CFG.num_init_ar_hmms)

###
# Fit
###
team_results = team_slds.fit(
    y_by_entity, SYSTEM_REGIMES_ONE_HOT, CFG.num_iters_laplace_em
)


###
# Postmortem
###

# TODO: Make this a standalone function?
for j in range(N_ENTITIES):
    print(f"Now plotting results for entity {j+1}/{N_ENTITIES}.")
    plot_elbos(team_results.elbo_traces[j])
    plot_results_for_one_entity(
        team_results.qs[j],
        team_slds.entity_models[j],
        team_data[j].y,
        team_data[j].x_true,
        team_data[j].z_true,
        system_input=SYSTEM_REGIMES_ONE_HOT,
    )
