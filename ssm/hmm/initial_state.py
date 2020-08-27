import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.optimizers import lbfgs
import ssm.distributions.distributions as dists
from ssm.util import format_dataset

def make_initial_state(num_states, initial_state, **initial_state_kwargs):
    """Helper function to construct initial state distribution
     of the desired type.
    """
    initial_state_names = dict(
        uniform=UniformInitialState,
    )
    if isinstance(initial_state, str):
        assert initial_state.lower() in initial_state_names, \
            "`initial_state` must be one of {}".format(initial_state_names.keys())
        return initial_state_names[initial_state.lower()](num_states, **initial_state_kwargs)
    else:
        assert isinstance(initial_state, InitialState)
        return initial_state


class InitialState(object):
    def __init__(self, num_states):
        self.num_states = num_states

    def log_initial_prob(self, **kwargs):
        raise NotImplementedError

    @format_dataset
    def initialize(self, dataset):
        pass

    def permute(self, perm):
        raise NotImplementedError

    def log_prior(self):
        return 0

    @format_dataset
    def expected_log_prob(self, dataset, posteriors):
        elp = self.log_prior()
        for data_dict, posterior in zip(dataset, posteriors):
            log_pi0 = self.log_initial_prob(**data_dict)
            elp += np.sum(posterior.expected_states[0] * log_pi0)
        return elp

    def m_step(self, dataset, posteriors,
               optimizer="lbfgs", num_iters=1000,
               **kwargs):
        """
        If M-step cannot be done in closed form for the transitions,
        default to BFGS.
        """
        # Normalize and negate for minimization
        T = sum([data_dict["data"].shape[0] for data_dict in dataset])
        def _objective(params, itr):
            self.unconstrained_params = params
            return -self.expected_log_prob(dataset, posteriors) / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            lbfgs(_objective, self.params, num_iters=num_iters,
                  state=optimizer_state, full_output=True, **kwargs)


class UniformInitialState(InitialState):
    def __init__(self, num_states):
        super(UniformInitialState, self).__init__(num_states)
        self.initial_state_dist = \
            dists.Categorical(np.ones(num_states) / num_states)

    def log_initial_prob(self, **kwargs):
        return np.log(self.initial_state_dist.probs)

    def initial_prob(self, **kwargs):
        return np.exp(self.log_initial_prob(**kwargs))

    def initialize(self, dataset):
        pass

    def permute(self, perm):
        self.initial_state_dist.probs = self.initial_state_dist.probs[perm]

    def log_prior(self):
        return 0

    def m_step(self, dataset, posteriors,
               optimizer="lbfgs", num_iters=1000,
               **kwargs):
        pass

    def stochastic_m_step(self, dataset, posteriors, step_size,
                          scale_factor=1.0,
                          **kwargs):
        pass