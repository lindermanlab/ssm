from ssm.core import \
    _StationaryHMM, \
    _InputDrivenHMM, \
    _RecurrentHMM, \
    _RecurrentOnlyHMM, \
    _LDSBase, \
    _SwitchingLDSBase

from ssm.likelihoods import \
    _GaussianHMMObservations, \
    _BernoulliHMMObservations, \
    _PoissonHMMObservations, \
    _StudentsTHMMObservations, \
    _AutoRegressiveHMMObservations, \
    _RobustAutoRegressiveHMMObservations, \
    _RecurrentAutoRegressiveHMMMixin, \
    _GaussianSLDSObservations


def HMM(K, D, M=0,
        observations="gaussian",
        recurrent=False,
        recurrent_only=False,
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
    observation_classes = dict(
        gaussian=_GaussianHMMObservations,
        studentst=_StudentsTHMMObservations,
        t=_StudentsTHMMObservations,
        poisson=_PoissonHMMObservations,
        bernoulli=_BernoulliHMMObservations,
        ar=_AutoRegressiveHMMObservations,
        autoregressive=_AutoRegressiveHMMObservations,
        robust_ar=_RobustAutoRegressiveHMMObservations,
        robust_autoregressive=_RobustAutoRegressiveHMMObservations
        )

    observations = observations.lower()
    if observations not in observation_classes:
        raise Exception("Invalid observation model: {}. Must be one of {}".
            format(observations, list(observation_classes.keys())))

    # Make a list of parent classes
    parents = (observation_classes[observations],)
    if recurrent_only:
        parents += (_RecurrentOnlyHMM,)
    elif recurrent:
        parents += (_RecurrentHMM,)
    elif M > 0:
        parents += (_InputDrivenHMM,)
    else:
        parents += (_StationaryHMM,)

    # Handle the special case of recurrent ARHMMs
    arhmm_observations = ["ar", "autoregressive", "robust_ar", "robust_autoregressive"]
    if (recurrent or recurrent_only) and (observations in arhmm_observations):
        parents = (_RecurrentAutoRegressiveHMMMixin,) + parents

    # Make the class and return a new instance of it
    cls = type("HMM", parents, {})
    return cls(K, D, M)


def SLDS(N, K, D, M=0,
         observations="gaussian",
         robust_dynamics=False,
         recurrent=False,
         recurrent_only=False,
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
    observation_classes = dict(
        gaussian=_GaussianSLDSObservations
        )

    observations = observations.lower()
    if observations not in observation_classes:
        raise Exception("Invalid observation model: {}. Must be one of {}".
            format(observations, list(observation_classes.keys())))

    # Make a list of parent classes, starting with the observation model
    parents = (observation_classes[observations],)

    # Add the SLDS base
    parents += (_SwitchingLDSBase,)
    
    # Inherit from the appropriate ARHMM
    if recurrent or recurrent_only:
        parents += (_RecurrentAutoRegressiveHMMMixin,)

    # Add the appropriate AR observations
    if robust_dynamics:
        parents += (_RobustAutoRegressiveHMMObservations,)
    else:
        parents += (_AutoRegressiveHMMObservations,)

    # Add the HMM base class
    if recurrent_only:
        parents += (_RecurrentOnlyHMM,)
    elif recurrent:
        parents += (_RecurrentHMM,)
    elif M > 0:
        parents += (_InputDrivenHMM,)
    else:
        parents += (_StationaryHMM,)

    # Make the class and return a new instance of it
    cls = type("SLDS", parents, {})
    return cls(N, K, D, M=M, single_subspace=single_subspace, **kwargs)


def LDS(N, D, M=0,
        observations="gaussian",
        robust_dynamics=False,
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
    observation_classes = dict(
        gaussian=_GaussianSLDSObservations
        )

    observations = observations.lower()
    if observations not in observation_classes:
        raise Exception("Invalid observation model: {}. Must be one of {}".
            format(observations, list(observation_classes.keys())))

    # Make a list of parent classes, starting with the observation model
    parents = (observation_classes[observations],)

    # Add the LDS base
    parents += (_LDSBase,)
    
    # Add the appropriate AR observations
    if robust_dynamics:
        parents += (_RobustAutoRegressiveHMMObservations,)
    else:
        parents += (_AutoRegressiveHMMObservations,)
    
    # Add the HMM base class
    parents += (_StationaryHMM,)

    # Make the class and return a new instance of it
    cls = type("LDS", parents, {})
    return cls(N, K=0, D=D, M=M, single_subspace=True, **kwargs)
