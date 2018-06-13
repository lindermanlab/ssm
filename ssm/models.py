from ssm.core import \
    _StationaryHMM, \
    _InputDrivenHMM, \
    _RecurrentHMM, \
    _RecurrentOnlyHMM, \
    _SwitchingLDSBase

from ssm.likelihoods import \
    _GaussianObservations, \
    _StudentsTObservations, \
    _AutoRegressiveObservations, \
    _RobustAutoRegressiveObservations, \
    _RecurrentAutoRegressiveHMMMixin, \
    _GaussianEmissions


# Gaussian observations
class GaussianHMM(_GaussianObservations, _StationaryHMM):
    pass


class InputDrivenGaussianHMM(_GaussianObservations, _InputDrivenHMM):
    pass


class RecurrentGaussianHMM(_GaussianObservations, _RecurrentHMM):
    pass


class RecurrentOnlyGaussianHMM(_GaussianObservations, _RecurrentOnlyHMM):
    pass


# Student's t observations
class StudentsTHMM(_StudentsTObservations, _StationaryHMM):
    pass


class InputDrivenStudentsTHMM(_StudentsTObservations, _InputDrivenHMM):
    pass


class RecurrentStudentsTHMM(_StudentsTObservations, _RecurrentHMM):
    pass


class RecurrentOnlyStudentsTHMM(_StudentsTObservations, _RecurrentOnlyHMM):
    pass


# Auto-regressive models
class AutoRegressiveHMM(_AutoRegressiveObservations, _StationaryHMM):
    pass


class InputDrivenAutoRegressiveHMM(_AutoRegressiveObservations, _InputDrivenHMM):
    pass



class RecurrentAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                 _AutoRegressiveObservations, 
                                 _RecurrentHMM):
    pass


class RecurrentOnlyAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                     _AutoRegressiveObservations, 
                                     _RecurrentOnlyHMM):
    pass


# Robust autoregressive models with Student's t noise
class RobustAutoRegressiveHMM(_RobustAutoRegressiveObservations, _StationaryHMM):
    pass


class InputDrivenRobustAutoRegressiveHMM(_RobustAutoRegressiveObservations, _InputDrivenHMM):
    pass


class RecurrentRobustAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                       _RobustAutoRegressiveObservations, 
                                       _RecurrentHMM):
    pass


class RecurrentOnlyRobustAutoRegressiveHMM(_RecurrentAutoRegressiveHMMMixin,
                                           _RobustAutoRegressiveObservations, 
                                           _RecurrentOnlyHMM):
    pass

# Switching LDS base classes
class _SwitchingLDS(_SwitchingLDSBase, 
                    AutoRegressiveHMM):
    pass


class _RecurrentSwitchingLDS(_SwitchingLDSBase, RecurrentAutoRegressiveHMM):
    pass


class _RecurrentOnlySwitchingLDS(_SwitchingLDSBase, RecurrentOnlyAutoRegressiveHMM):
    pass


# Robust versions with Student's t dynamics noise
class _RobustSwitchingLDS(_SwitchingLDSBase, RobustAutoRegressiveHMM):
    pass


class _RecurrentRobustSwitchingLDS(_SwitchingLDSBase, RecurrentRobustAutoRegressiveHMM):
    pass


class _RecurrentOnlyRobustSwitchingLDS(_SwitchingLDSBase, RecurrentOnlyRobustAutoRegressiveHMM):
    pass


# Standard Gaussian versions
class GaussianSLDS(_GaussianEmissions, _SwitchingLDS):
    pass


class GaussianRecurrentSLDS(_GaussianEmissions, _RecurrentSwitchingLDS):
    pass


class GaussianRecurrentOnlySLDS(_GaussianEmissions, _RecurrentOnlySwitchingLDS):
    pass


# Robust versions
class GaussianRobustSLDS(_GaussianEmissions, _RobustSwitchingLDS):
    pass


class GaussianRecurrentRobustSLDS(_GaussianEmissions, _RecurrentRobustSwitchingLDS):
    pass


class GaussianRecurrentOnlyRobustSLDS(_GaussianEmissions, _RecurrentOnlyRobustSwitchingLDS):
    pass
