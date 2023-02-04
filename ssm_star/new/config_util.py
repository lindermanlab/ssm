from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseSettings, root_validator

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
    L_true: int # num system level regimes, currently must be an even multiple of K_true 
    K_true: int # num entity level regimes
    D_true: int # state dim 
    N: int  # obs dim 
    T: int # num timesteps 
    seed : int 
    use_fixed_cycling_through_regimes_instead_of_Markov_chain_sampling : bool # for data generation
    system_influence_scalar: float # strength of system influence; not used if `use_fixed_cycling_through_regimes_instead_of_Markov_chain_sampling` = True
    alpha : float # Dirichlet parameter for symmetric Dirichlet on rows on t.p.m for regimes ....
    kappa :float # ...with symmetry-breaking increment to prior to encourage self-transitions. 
    ###
    # Inference
    ### 
    num_iters_laplace_em : int
    smart_initialize : bool  
    num_init_ar_hmms : int 

def load(path_to_config):
    with open(path_to_config, "r") as f:
        return Config(**yaml.full_load(f))

