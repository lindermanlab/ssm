from warnings import warn

import jax.numpy as np
import numpy.random as npr

import ssm.distributions as dists
from ssm.optimizers import minimize, convex_combination
from ssm.util import format_dataset


def make_transitions(num_states, transitions, **transition_kwargs):
    """Helper function to construct transitions of the desired type.
    """
    transition_names = dict(
        standard=StationaryTransitions,
        stationary=StationaryTransitions,
    )
    if isinstance(transitions, np.ndarray):
        # Assume this is a transition matrix
        return StationaryTransitions(num_states,
                                     transition_matrix=transitions)

    elif isinstance(transitions, str):
        # String specifies class of transitions
        assert transitions.lower() in transition_names, \
            "`transitions` must be one of {}".format(transition_names.keys())
        transition_class = transition_names[transitions.lower()]
        return transition_class(num_states, **transition_kwargs)
    else:
        # Otherwise, we need a Transitions object
        assert isinstance(transitions, Transitions)
        return transitions


class Transitions(object):
    def __init__(self, num_states):
        self.num_states = num_states

    @format_dataset
    def initialize(self, dataset):
        pass

    def permute(self, perm):
        raise NotImplementedError

    def log_prior(self):
        return 0

    def get_transition_matrix(self, **kwargs):
        """
        Compute the transition matrix for a single timestep.
        Optional kwargs include a single timestep's data and
        covariates.
        """
        raise NotImplementedError

    def log_transition_matrices(self, data, **kwargs):
        raise NotImplementedError

    @format_dataset
    def expected_log_prob(self, dataset, posteriors):
        elp = self.log_prior()
        for data_dict, posterior in zip(dataset, posteriors):
            log_P = self.log_transition_matrices(**data_dict)
            elp += np.sum(posterior.expected_transitions * log_P)
        return elp

    @format_dataset
    def m_step(self, dataset, posteriors,
               num_iters=100,
               **kwargs):
        """
        By default, maximize the expected log likelihood with BFGS.
        """
        # Normalize and negate for minimization
        T = sum([data_dict["data"].shape[0] for data_dict in dataset])
        def _objective(params, itr):
            self.unconstrained_params = params
            return -self.expected_log_prob(dataset, posteriors) / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls
        # to m_step.
        result = minimize(_objective,
                          self.unconstrained_params,
                          **kwargs)

        if not result.success:
            warn("fit: minimize failed with result: {}".format(result))

        self.unconstrained_params = result['x']


class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and
    transition matrix.
    """
    def __init__(self, num_states, transition_matrix=None):
        super(StationaryTransitions, self).__init__(num_states)
        if transition_matrix is None:
            # default to a weakly sticky transition matrix
            self.set_transition_matrix(
                0.8 * np.eye(num_states) + \
                0.2 / (num_states - 1) * (1 - np.eye(num_states)))
        else:
            self.set_transition_matrix(transition_matrix)

        assert np.allclose(self.get_transition_matrix().sum(axis=1), 1.0)

        # Initialize state for stochastic EM
        self._stochastic_m_step_state = None

    def get_transition_matrix(self, **kwargs):
        return np.row_stack([d.probs for d in self.conditional_dists])

    def set_transition_matrix(self, value):
        assert value.shape == (self.num_states, self.num_states)
        assert np.allclose(np.sum(value, axis=1), 1.0)
        self.conditional_dists = [
            dists.Categorical(row) for row in value
        ]

    def log_prior(self):
        return np.sum([d.log_prior() for d in self.conditional_dists])

    def log_transition_matrices(self, data, **kwargs):
        # with np.errstate(divide='ignore'):
        #     return np.log(self.get_transition_matrix())
        return np.log(self.get_transition_matrix())

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.set_transition_matrix(
            self.get_transition_matrix()[np.ix_(perm, perm)]
        )

    @format_dataset
    def m_step(self, dataset, posteriors):
        """Exact M-step is possible for the standard transition model.
        """
        expected_transitions = sum([posterior.expected_transitions()
                                    for posterior in posteriors])

        for row, dist in zip(expected_transitions, self.conditional_dists):
            dist.fit_expfam_with_stats((row,), row.sum())

    @format_dataset
    def stochastic_m_step(self, dataset, posteriors, step_size,
                          scale_factor=1.0,
                          **kwargs):
        # Compute expected sufficient statistics (transition counts)
        # for this dataset
        expected_transitions = \
            scale_factor * sum([posterior.expected_transitions()
                                for posterior in posteriors])

        # Take a convex combination with past sufficient statistics
        if self._stochastic_m_step_state is not None:
            # Take a convex combination of sufficient statistics from
            # this batch and those accumulated thus far.
            expected_transitions = convex_combination(
                self._stochastic_m_step_state["expected_transitions"],
                expected_transitions,
                step_size)

        # Update the transition matrix using effective suff stats
        for row, dist in zip(expected_transitions, self.conditional_dists):
            dist.fit_expfam_with_stats((row,), row.sum())

        # Save the updated sufficient statistics
        self._stochastic_m_step_state = dict(
            expected_transitions=expected_transitions
        )
