from ssm.core import _HMM, _LDS, _SwitchingLDS

from ssm.init_state_distns import InitialStateDistribution

from ssm.transitions import \
    StationaryTransitions, \
    StickyTransitions, \
    InputDrivenTransitions, \
    RecurrentTransitions, \
    RecurrentOnlyTransitions

from ssm.observations import \
    GaussianObservations, \
    BernoulliObservations, \
    PoissonObservations, \
    StudentsTObservations, \
    AutoRegressiveObservations, \
    RobustAutoRegressiveObservations, \
    RecurrentAutoRegressiveObservations, \
    RecurrentRobustAutoRegressiveObservations
    
from ssm.hierarchical import \
    HierarchicalInitialStateDistribution, \
    HierarchicalTransitions, \
    HierarchicalObservations, \
    HierarchicalEmissions

from ssm.emissions import \
    GaussianEmissions, \
    BernoulliEmissions, \
    PoissonEmissions, \
    StudentsTEmissions, \
    AutoRegressiveEmissions


def HMM(K, D, M=0,
        transitions="standard",
        hierarchical_transition_tags=None,
        observations="gaussian",
        hierarchical_observation_tags=None,
        **kwargs):
    """
    Construct an HMM object with the appropriate observations 
    and dynamics. 

    :param K: number of discrete latent states
    :param D: observation dimension
    :param M: input dimension
    :param observations: conditional distribution of the data 
    :param recurrent: whether or not past observations influence transitions probabilities.
    :param recurrent_only: if true, _only_ the past observations influence transitions. 
    """

    # Make the initial state distribution
    init_state_distn = InitialStateDistribution(K, D, M=M)

    # Make the transition model
    transition_classes = dict(
        standard=StationaryTransitions,
        sticky=StickyTransitions,
        inputdriven=InputDrivenTransitions,
        recurrent=RecurrentTransitions,
        recurrent_only=RecurrentOnlyTransitions
        )
    if transitions not in transition_classes:
        raise Exception("Invalid transition model: {}. Must be one of {}".
            format(transitions, list(transition_classes.keys())))
    
    transition_distn = \
        HierarchicalTransitions(transition_classes[transitions], K, D, M=M, 
                                tags=hierarchical_transition_tags) \
        if hierarchical_transition_tags is not None \
        else transition_classes[transitions](K, D, M=M)

    # This is the master list of observation classes.  
    # When you create a new observation class, add it here.
    is_recurrent = (transitions.lower() in ["recurrent", "recurrent_only"])
    observation_classes = dict(
        gaussian=GaussianObservations,
        studentst=StudentsTObservations,
        t=StudentsTObservations,
        poisson=PoissonObservations,
        bernoulli=BernoulliObservations,
        ar=RecurrentAutoRegressiveObservations if is_recurrent else AutoRegressiveObservations,
        autoregressive=RecurrentAutoRegressiveObservations if is_recurrent else AutoRegressiveObservations,
        robust_ar=RecurrentRobustAutoRegressiveObservations if is_recurrent else RobustAutoRegressiveObservations,
        robust_autoregressive=RecurrentRobustAutoRegressiveObservations if is_recurrent else RobustAutoRegressiveObservations,
        )

    observations = observations.lower()
    if observations not in observation_classes:
        raise Exception("Invalid observation model: {}. Must be one of {}".
            format(observations, list(observation_classes.keys())))

    observation_distn = \
        HierarchicalObservations(observation_classes[observations], K, D, M=M, 
                                 tags=hierarchical_observation_tags) \
        if hierarchical_observation_tags is not None \
        else observation_classes[observations](K, D, M=M)

    # Make the HMM
    return _HMM(K, D, M, init_state_distn, transition_distn, observation_distn)


def SLDS(N, K, D, M=0,
         transitions="standard",
         hierarchical_transition_tags=None,
         dynamics="gaussian",
         hierarchical_dynamics_tags=None,
         emissions="gaussian",
         hierarchical_emission_tags=None,
         single_subspace=True,
         **kwargs):
    """
    Construct an SLDS object with the appropriate observations, latent states, and dynamics. 

    :param N: observation dimension
    :param K: number of discrete latent states
    :param D: latent dimension
    :param M: input dimension
    :param observations: conditional distribution of the data 
    :param robust_dynamics: if true, continuous latent states have Student's t noise.
    :param recurrent: whether or not past observations influence transitions probabilities.
    :param recurrent_only: if true, _only_ the past observations influence transitions. 
    :param single_subspace: if true, all discrete states share the same mapping from 
        continuous latent states to observations.
    """
    # Make the initial state distribution
    init_state_distn = InitialStateDistribution(K, D, M=M)

    # Make the transition model
    transition_classes = dict(
        standard=StationaryTransitions,
        stationary=StationaryTransitions,
        sticky=StickyTransitions,
        inputdriven=InputDrivenTransitions,
        recurrent=RecurrentTransitions,
        recurrent_only=RecurrentOnlyTransitions
        )
    if transitions not in transition_classes:
        raise Exception("Invalid transition model: {}. Must be one of {}".
            format(transitions, list(transition_classes.keys())))
    
    transition_distn = \
        HierarchicalTransitions(transition_classes[transitions], K, D, M, 
                                tags=hierarchical_transition_tags) \
        if hierarchical_transition_tags is not None\
        else transition_classes[transitions](K, D, M=M)

    # Make the dynamics distn
    is_recurrent = (transitions.lower() in ["recurrent", "recurrent_only"])
    dynamics_classes = dict(
        none=GaussianObservations,
        gaussian=RecurrentAutoRegressiveObservations if is_recurrent else AutoRegressiveObservations,
        t=RecurrentRobustAutoRegressiveObservations if is_recurrent else RobustAutoRegressiveObservations,
        studentst=RecurrentRobustAutoRegressiveObservations if is_recurrent else RobustAutoRegressiveObservations,
        )

    dynamics = dynamics.lower()
    if dynamics not in dynamics_classes:
        raise Exception("Invalid dynamics model: {}. Must be one of {}".
            format(dynamics, list(dynamics_classes.keys())))

    dynamics_distn = \
        HierarchicalObservations(dynamics_classes[dynamics], K, D, M, 
                                 tags=hierarchical_dynamics_tags) \
        if hierarchical_dynamics_tags is not None \
        else dynamics_classes[dynamics](K, D, M=M)

    # Make the emission distn    
    emission_classes = dict(
        gaussian=GaussianEmissions,
        studentst=StudentsTEmissions,
        t=StudentsTEmissions,
        poisson=PoissonEmissions,
        bernoulli=BernoulliEmissions,
        ar=AutoRegressiveEmissions,
        autoregressive=AutoRegressiveEmissions
        )

    emissions = emissions.lower()
    if emissions not in emission_classes:
        raise Exception("Invalid emission model: {}. Must be one of {}".
            format(emissions, list(emission_classes.keys())))

    emission_distn = \
        HierarchicalEmissions(emission_classes[emissions], N, K, D, M, 
                              tags=hierarchical_emission_tags, 
                              single_subspace=single_subspace) \
        if hierarchical_emission_tags is not None \
        else emission_classes[emissions](N, K, D, M=M, single_subspace=single_subspace)

    # Make the HMM
    return _SwitchingLDS(N, K, D, M, init_state_distn, transition_distn, dynamics_distn, emission_distn)


def LDS(N, D, M=0,
        dynamics="gaussian",
        hierarchical_dynamics_tags=None,
        emissions="gaussian",
        hierarchical_emission_tags=None,
        **kwargs):
    """
    Construct an LDS object with the appropriate observations, latent states, and dynamics. 
    Currently, this uses a lot of the same code path as the SLDS.

    :param N: observation dimension
    :param D: latent dimension
    :param M: input dimension
    :param observations: conditional distribution of the data 
    :param robust_dynamics: if true, continuous latent states have Student's t noise.
    """
    # Make the dynamics distn
    is_recurrent = (transitions.lower() in ["recurrent", "recurrent_only"])
    dynamics_classes = dict(
        gaussian=AutoRegressiveObservations,
        t=RobustAutoRegressiveObservations,
        studentst=RobustAutoRegressiveObservations,
        )

    dynamics = dynamics.lower()
    if dynamics not in dynamic_classes:
        raise Exception("Invalid dynamics model: {}. Must be one of {}".
            format(dynamics, list(dynamic_classes.keys())))

    dynamics_distn = \
        HierarchicalDynamics(dynamics_classes[dynamics], 1, D, M, 
                             tags=hierarchical_dynamics_tags) \
        if hierarchical_dynamics_tags is not None \
        else dynamics_classes[dynamics](1, D, M=M)

    # Make the emission distn    
    emission_classes = dict(
        gaussian=GaussianEmissions
        )

    emissions = emissions.lower()
    if emissions not in emission_classes:
        raise Exception("Invalid emission model: {}. Must be one of {}".
            format(emissions, list(emission_classes.keys())))

    emission_distn = \
        HierarchicalEmissions(emission_classes[emissions], N, 1, D, M, 
                              tags=hierarchical_emission_tags) \
        if hierarchical_emissions is not None \
        else emission_classes[emissions](N, 1, D, M=M)


    # Make the HMM
    return _LDS(N, D, M, dynamics_distn, emission_distn)


