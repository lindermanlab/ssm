from dataclasses import dataclass
from typing import List

import autograd.numpy as np
from lds.types import NumpyArray1D

import ssm_star


@dataclass
class EntityData:
    index: int
    y: np.array
    x_true: np.array
    z_true: np.array


TeamData = List[EntityData]


@dataclass
class TeamInferenceResults:
    elbo_traces: List[NumpyArray1D]
    qs: List[ssm_star.variational.SLDSStructuredMeanFieldVariationalPosterior]


@dataclass
class TeamSLDS:
    def __init__(
        self, num_entities, N, K, D, L, emissions, transitions, transition_kwargs
    ):
        self.num_entities = num_entities
        self.entity_models = [
            ssm_star.SLDS(
                N,
                K,
                D,
                L=L,
                emissions=emissions,
                transitions=transitions,
                transition_kwargs=transition_kwargs,
            )
            for j in range(num_entities)
        ]

    def smart_initialize(
        self,
        y_by_entity: List[np.array],
        system_inputs_one_hot: np.array,
        num_init_restarts: int,
    ):
        ### Warning!  Linderman's initialization seems to assume that the obs dim exceeds the state dim!
        # And if initialization is not done, results are very poor.
        # See: https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/lds.py#L161
        for j in range(self.num_entities):
            print(f"Now smart-initializing entity {j+1}/{self.num_entities}.")
            self.entity_models[j].initialize(
                y_by_entity[j],
                num_init_restarts=num_init_restarts,
                system_inputs=system_inputs_one_hot,
            )

    def fit(
        self,
        y_by_entity: List[np.array],
        system_inputs_one_hot: np.array,
        num_iters_laplace_em: int,
    ) -> TeamInferenceResults:
        elbo_traces, qs = [None] * self.num_entities, [None] * self.num_entities
        for j in range(self.num_entities):
            print(f"Now fitting model for entity {j+1}/{self.num_entities}.")
            elbo_trace, q = self.entity_models[j].fit(
                y_by_entity[j],
                system_inputs=system_inputs_one_hot,
                method="laplace_em",
                variational_posterior="structured_meanfield",
                initialize=False,
                num_iters=num_iters_laplace_em,
            )
            elbo_traces[j], qs[j] = elbo_trace, q
        return TeamInferenceResults(elbo_traces, qs)
