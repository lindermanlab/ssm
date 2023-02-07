from typing import List, Optional

import numpy as np
import seaborn as sns
from lds.types import NumpyArray3D
from matplotlib import pyplot as plt


def plot_entity_responsivenesses_to_system(
    entity_responsivenesses: NumpyArray3D,
    system_influence_scalars: Optional[List[float]] = None,
) -> None:
    """
    Arguments:
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

    J, T_minus_one, K = np.shape(entity_responsivenesses)

    colors = sns.color_palette("icefire", n_colors=J)

    # Plot the true and inferred states
    plt.figure(figsize=(8, 9))

    for k in range(K):
        responsivenesses_when_transitioning_to_k = entity_responsivenesses[
            :, :, k
        ]  #  J x (T-1)
        plt.subplot(K, 1, k + 1)
        for j in reversed(range(J)):
            label = (
                f"{system_influence_scalars[j]:02}"
                if system_influence_scalars
                else f"{j}"
            )
            plt.plot(
                responsivenesses_when_transitioning_to_k[j, :],
                color=colors[j],
                label=label,
            )
        plt.title(f"Entity responsiveness when transitioning to regime {k+1}")

    legend_title = (
        "System influence scalar for entity in DGP"
        if system_influence_scalars
        else "Entity ID"
    )
    # plt.legend(title=legend_title, bbox_to_anchor=(1.1, 1.05), title_fontsize=12, fontsize=12)
    plt.legend(title=legend_title, loc="best", title_fontsize=12, fontsize=12)
    plt.tight_layout()
    plt.show()
