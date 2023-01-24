"""
Generate pLGSSM data with univariate state and emissions,
and K=2 regimes.
"""
from typing import Tuple 
import numpy as np

from lds.generate import make_params_for_univariate_state_space_model
from lds.piecewise.generate import generate_piecewise_time_invariant_LGSSM
from lds.piecewise.util import make_ssm_params_by_regime_from_list_of_dicts



np.set_printoptions(suppress=True, precision=3)

def generate_1dim_data_with_1dim_states() -> Tuple[np.array, np.array]:

    ###
    # Hyperparameters
    ###
    regime_run_length = 200
    regime_seq_true = ([0] * regime_run_length + [1] * regime_run_length) * 2
    params_dict_regime_0 = {
        "a": 0.99,
        "c": 1.0,
        "q": 1.0,
        "r": 0.1,
        "mu_0": 0.0,
        "sigma_0": 1.0,
    }
    params_dict_regime_1 = {
        "a": 0.90,
        "c": 1.0,
        "q": 10.0,
        "r": 0.1,
        "mu_0": 0.0,
        "sigma_0": 1.0,
    }
    num_regimes = 2  # to do: extract rather than hardcode
    seed = 2000

    ###
    # Generate Data
    ###
    params_regime_0 = make_params_for_univariate_state_space_model(**params_dict_regime_0)
    params_regime_1 = make_params_for_univariate_state_space_model(**params_dict_regime_1)
    ssm_params_by_regime = [params_regime_0, params_regime_1]

    y_seq, state_seq_true = generate_piecewise_time_invariant_LGSSM(
        ssm_params_by_regime,
        regime_indices_by_timestep=regime_seq_true,
        seed=seed,
    )
    return y_seq, state_seq_true, regime_seq_true

def generate_3dim_data_with_2dim_states():

    regime_run_length = 50
    regime_seq_true = ([0] * regime_run_length + [1] * regime_run_length) * 2
    params_dict_regime_0 = {
        "a_scalar": 0.99,
        "theta": np.pi / 20,
        "q_scalar_var": 1.0,
        "init_cov_eye": True,
    }
    params_dict_regime_1 = {
        "a_scalar": 0.90,
        "theta": np.pi / 2,
        "q_scalar_var": 10.0,
        "init_cov_eye": True,
    }
    params_dicts_by_regimes = [params_dict_regime_0, params_dict_regime_1]
    num_regimes = len(params_dicts_by_regimes)
    seed_for_generating_data = 2000
    seeds_for_generating_ssm_params_by_regime = [i for i in range(num_regimes)]
    state_dim = 2
    obs_dim = 3

    ssm_params_by_regime = make_ssm_params_by_regime_from_list_of_dicts(
        params_dicts_by_regimes,
        state_dim,
        obs_dim,
        seeds_for_generating_ssm_params_by_regime,
    )

    y_seq, state_seq_true = generate_piecewise_time_invariant_LGSSM(
        ssm_params_by_regime,
        regime_indices_by_timestep=regime_seq_true,
        seed=seed_for_generating_data,
    )
    return y_seq, state_seq_true, regime_seq_true