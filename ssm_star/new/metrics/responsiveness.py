from typing import Optional

import numpy as np
from lds.types import NumpyArray3D
from lds.util import one_hot_encoded_array_from_categorical_indices

from ssm_star.new.team import TeamInferenceResults, TeamSLDS


def compute_entity_responsivenesses_to_system(
    team_results: TeamInferenceResults,
    team_slds: TeamSLDS,
    num_of_system_regimes: int,
    input: Optional[np.array] = None,
) -> NumpyArray3D:
    """
    Arguments:
        input: Exogenous inputs to the entity-level regime transition matrix.

    Returns:
        entity_responsivenesses: (J, T-1, K). The variance (across the L system-level regimes)
            in the k=1,...,K destinations for transitioning to a new entity-level regime for the
            j-th entity at timestep t=2,...,T, when using the variational mean for the preceding
            state and preceding regime (i.e. E_q[x_{t-1}] and E_q[z_{t-1}].)

            Note therefore that the responsiveness to the system regime depends on the entity, the destination regime,
            and the timestep.

    Notation:
        J : number of entities
        T : number of timesteps
        K : number of entity-level states
        L : number of system-level regimes
    """

    # TODO: Generalize this function for when there is NOT a 1-to-1 mapping from system regimes to entity regimes
    # For now raise a NotImplementedError!

    J = len(team_results.qs)
    _, T, K = np.shape(team_results.qs[0].mean_discrete_states)
    L = num_of_system_regimes

    entity_responsivenesses = np.zeros((J, T - 1, K))
    for j in range(J):
        # TODO: why is there a list of discrete states and continuous states with one entry in each?!
        z_mean = team_results.qs[j].mean_discrete_states[0]
        x_mean = team_results.qs[j].mean_continuous_states[0]

        mask = np.ones_like(x_mean, dtype=bool)
        tag = None
        if input is None:
            input = np.zeros((T, 0))

        result = np.zeros((L, T - 1, K))

        for l in range(L):
            system_input_indicators = np.array([l] * T)
            system_input_one_hot = one_hot_encoded_array_from_categorical_indices(
                system_input_indicators, L
            )

            log_Ps = team_slds.entity_models[j].transitions.log_transition_matrices(
                x_mean, input, mask, tag, system_input_one_hot
            )
            Ps = np.exp(log_Ps)

            # Draw from tpm at the z_mean
            for t in range(1, T):
                prev_z = np.argmax(z_mean[t - 1])
                result[l, t - 1, :] = Ps[t - 1, prev_z, :]

        # Now compute how much variance there is across the L system-level regimes
        # in each entity-level destination k for each timestep t=2,...,T

        variance_matrix = np.var(result, axis=0)  # TxK matrix
        entity_responsivenesses[j, :, :] = variance_matrix
        # grand_mean_of_variance_matrix=np.mean(variance_matrix)
        # print(f"For entity {j}, grand mean of variance matrix : {grand_mean_of_variance_matrix:.02f}")

    return entity_responsivenesses
