import yaml
from pydantic import BaseSettings


class Config(BaseSettings):
    # TODO: Split up data generation and inference hierarchically
    # TODO: Separate redundant configs from data generation and inference; they need not match
    # TODO: Add other configs used in the demo (e.g. string-valued choice of transitions)

    ###
    # Meta data
    ###
    summary_of_run: str

    ###
    # Data Generation
    ###
    L_true: int  # num system level regimes, currently must be an even multiple of K_true
    K_true: int  # num entity level regimes
    D_true: int  # state dim
    N: int  # obs dim
    T: int  # num timesteps
    seed: int
    fixed_run_length_for_entity_level_regimes: int  # set to 0 if you want to sample regimes from Markov chain with tpm.
    fixed_run_length_for_system_level_regimes: int
    system_influence_scalar: float  # strength of system influence; not used if `fixed_run_length_for_entity_level_regimes` = True
    alpha: float  # Dirichlet parameter for symmetric Dirichlet on rows on t.p.m for regimes ....
    kappa: float  # ...with symmetry-breaking increment to prior to encourage self-transitions.
    ###
    # Inference
    ###
    num_iters_laplace_em: int
    smart_initialize: bool
    num_init_ar_hmms: int


def load(path_to_config):
    with open(path_to_config, "r") as f:
        return Config(**yaml.full_load(f))
