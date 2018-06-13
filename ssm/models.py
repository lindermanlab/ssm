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


# Gaussian HMMs
class GaussianHMM(_GaussianHMMObservations, _StationaryHMM):
    pass


class InputDrivenGaussianHMM(_GaussianHMMObservations, _InputDrivenHMM):
    pass


class RecurrentGaussianHMM(_GaussianHMMObservations, _RecurrentHMM):
    pass


class RecurrentOnlyGaussianHMM(_GaussianHMMObservations, _RecurrentOnlyHMM):
    pass

# Student's t HMMs
class StudentsTHMM(_StudentsTHMMObservations, _StationaryHMM):
    pass


class InputDrivenStudentsTHMM(_StudentsTHMMObservations, _InputDrivenHMM):
    pass


class RecurrentStudentsTHMM(_StudentsTHMMObservations, _RecurrentHMM):
    pass


class RecurrentOnlyStudentsTHMM(_StudentsTHMMObservations, _RecurrentOnlyHMM):
    pass


# Poisson HMMs
class PoissonHMM(_PoissonHMMObservations, _StationaryHMM):
    pass


class InputDrivenPoissonHMM(_PoissonHMMObservations, _InputDrivenHMM):
    pass


class RecurrentPoissonHMM(_PoissonHMMObservations, _RecurrentHMM):
    pass


class RecurrentOnlyPoissonHMM(_PoissonHMMObservations, _RecurrentOnlyHMM):
    pass


# Bernoulli HMMs
class BernoulliHMM(_BernoulliHMMObservations, _StationaryHMM):
    pass


class InputDrivenBernoulliHMM(_BernoulliHMMObservations, _InputDrivenHMM):
    pass


class RecurrentBernoulliHMM(_BernoulliHMMObservations, _RecurrentHMM):
    pass


class RecurrentOnlyBernoulliHMM(_BernoulliHMMObservations, _RecurrentOnlyHMM):
    pass


# Auto-regressive models
class AutoRegressiveHMM(_AutoRegressiveHMMObservations, _StationaryHMM):
    pass


class InputDrivenAutoRegressiveHMM(_AutoRegressiveHMMObservations, _InputDrivenHMM):
    pass



class RecurrentAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                 _AutoRegressiveHMMObservations, 
                                 _RecurrentHMM):
    pass


class RecurrentOnlyAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                     _AutoRegressiveHMMObservations, 
                                     _RecurrentOnlyHMM):
    pass

# Robust autoregressive models with Student's t noise
class RobustAutoRegressiveHMM(_RobustAutoRegressiveHMMObservations, _StationaryHMM):
    pass


class InputDrivenRobustAutoRegressiveHMM(_RobustAutoRegressiveHMMObservations, _InputDrivenHMM):
    pass


class RecurrentRobustAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                       _RobustAutoRegressiveHMMObservations, 
                                       _RecurrentHMM):
    pass


class RecurrentOnlyRobustAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                           _RobustAutoRegressiveHMMObservations, 
                                           _RecurrentOnlyHMM):
    pass


# LDS base classes
class _LDS(_LDSBase, AutoRegressiveHMM):
    pass


# Switching LDS base classes
class _SwitchingLDS(_SwitchingLDSBase, AutoRegressiveHMM):
    pass


class _RecurrentSwitchingLDS(_SwitchingLDSBase, RecurrentAutoRegressiveHMM):
    pass


class _RecurrentOnlySwitchingLDS(_SwitchingLDSBase, RecurrentOnlyAutoRegressiveHMM):
    pass


# Robust versions with Student's t dynamics noise
class _RobustLDS(_LDSBase, RobustAutoRegressiveHMM):
    pass


class _RobustSwitchingLDS(_SwitchingLDSBase, RobustAutoRegressiveHMM):
    pass


class _RecurrentRobustSwitchingLDS(_SwitchingLDSBase, RecurrentRobustAutoRegressiveHMM):
    pass


class _RecurrentOnlyRobustSwitchingLDS(_SwitchingLDSBase, RecurrentOnlyRobustAutoRegressiveHMM):
    pass


# Standard Gaussian versions
class GaussianLDS(_GaussianSLDSObservations, _LDS):
    pass


class GaussianSLDS(_GaussianSLDSObservations, _SwitchingLDS):
    pass


class GaussianRecurrentSLDS(_GaussianSLDSObservations, _RecurrentSwitchingLDS):
    pass


class GaussianRecurrentOnlySLDS(_GaussianSLDSObservations, _RecurrentOnlySwitchingLDS):
    pass

# Robust versions
class GaussianRobustLDS(_GaussianSLDSObservations, _RobustLDS):
    pass


class GaussianRobustSLDS(_GaussianSLDSObservations, _RobustSwitchingLDS):
    pass


class GaussianRecurrentRobustSLDS(_GaussianSLDSObservations, _RecurrentRobustSwitchingLDS):
    pass


class GaussianRecurrentOnlyRobustSLDS(_GaussianSLDSObservations, _RecurrentOnlyRobustSwitchingLDS):
    pass
