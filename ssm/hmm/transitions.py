from warnings import warn

import jax.numpy as np
from jax import lax
from jax.tree_util import register_pytree_node, register_pytree_node_class
# import numpy.random as npr

import jxf.distributions as dists
from ssm.optimizers import minimize, convex_combination
from ssm.util import format_dataset, num_datapoints

class Transitions(object):
    def __init__(self, num_states):
        self.num_states = num_states

    @format_dataset
    def initialize(self, rng, dataset):
        pass

    def permute(self, perm):
        raise NotImplementedError

    def log_prior(self):
        return 0

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


@register_pytree_node_class
class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and
    transition matrix.
    """
    def __init__(self,
                 num_states,
                 transition_matrix=None,
                 prior=None):
        super(StationaryTransitions, self).__init__(num_states)

        # Initialize the condtional distributions (i.e. transition matrix)
        if transition_matrix is None:
            transition_matrix = 0.8 * np.eye(num_states) + \
                0.2 / (num_states - 1) * (1 - np.eye(num_states))
        # else:
        #     assert transition_matrix.shape == (num_states, num_states)
        #     assert np.allclose(transition_matrix.sum(axis=1), 1.0)

        self.conditional_dists = [
            dists.Categorical(probs=row) for row in transition_matrix
        ]

        # Initialize the prior distribution
        if prior is None:
            prior = dists.Dirichlet(1.01 * np.ones(num_states))
        self.prior = prior

    def tree_flatten(self):
        return ((self.conditional_dists, self.prior), self.num_states)

    @classmethod
    def tree_unflatten(cls, num_states, children):
        conditional_dists, prior = children
        transition_matrix = np.row_stack([d.probs_parameter() for d in conditional_dists])
        return cls(num_states, transition_matrix, prior=prior)

    @property
    def transition_matrix(self):
        return np.row_stack([d.probs_parameter() for d in self.conditional_dists])

    @transition_matrix.setter
    def transition_matrix(self, value):
        assert value.shape == (self.num_states, self.num_states)
        self.conditional_dists = [dists.Categorical(probs=row) for row in value]

    def log_prior(self):
        return 0.0
        # return np.sum([d.log_prior() for d in self.conditional_dists])

    def log_transition_matrices(self, data, **kwargs):
        # with np.errstate(divide='ignore'):
        #     return np.log(self.get_transition_matrix())
        return np.log(self.transition_matrix)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.transition_matrix = self.transition_matrix[np.ix_(perm, perm)]

    def sample(self, rng, curr_state):
        def _sample(d): return lambda seed: d.sample(seed=seed)
        trans_sample_funcs = [_sample(d) for d in self.conditional_dists]
        return lax.switch(curr_state, trans_sample_funcs, rng)

    @format_dataset
    def m_step(self, dataset, posteriors):
        """Exact M-step is possible for the standard transition model.
        """
        expected_transitions = sum([posterior.expected_transitions()
                                    for posterior in posteriors])

        new_conditional_dists = []
        for row, dist in zip(expected_transitions, self.conditional_dists):
            suff_stats = (row[:-1],)
            num_datapoints = row.sum()
            new_conditional_dists.append(
                dist.fit_with_stats(suff_stats, num_datapoints, prior=self.prior))
        self.conditional_dists = new_conditional_dists

    def proximal_optimizer(self, total_num_datapoints, step_size=0.75):
        """Return an optimizer triplet, like jax.experimental.optimizers,
        to perform proximal gradient ascent on the likelihood with a penalty
        on the KL divergence between distributions from one iteration to the
        next. This boils down to taking a convex combination of sufficient
        statistics from this data and those that have been accumulated from
        past data.

        Parameters:
            total_num_datapoints:
                number of data points in the entire dataset.

            step_size:
                fixed value in [0, 1] or a function mapping
                iteration (int) to step size in [0, 1]

        Returns:

            initial_state    :: initial optimizer state
            update           :: minibatch, itr, state -> state
            get_distribution :: state -> Distribution object
        """
        # get proximal optimizer funcs for each conditional distribution
        initial_state, update_fns, get_distribution_fns = \
            list(zip([dist.proximal_optimizer(prior=self.prior,
                                              step_size=step_size)
                      for dist in self.conditional_dists]))

        def update(minibatch, posteriors, itr, state):
            scale_factor = total_num_datapoints / num_datapoints(minibatch)

            expected_transitions = sum([posterior.expected_transitions()
                                        for posterior in posteriors])

            new_states = []
            for update_fn, this_state, row in zip(update_fns, state, expected_transitions):
                new_states.append(update_fn([],  # dummy dataset
                                            itr,
                                            this_state,
                                            suff_stats=(row[:-1],),
                                            num_datapoints=row.sum(),
                                            scale_factor=scale_factor))

            # Update the conditional distributions
            self.conditional_dists = \
                [f(this_state) for f, this_state in zip(get_distribution_fns, state)]

        return initial_state, update

    # @format_dataset
    # def stochastic_m_step(self, dataset, posteriors, step_size,
    #                       scale_factor=1.0,
    #                       **kwargs):
    #     # Compute expected sufficient statistics (transition counts)
    #     # for this dataset
    #     expected_transitions = \
    #         scale_factor * sum([posterior.expected_transitions()
    #                             for posterior in posteriors])

    #     # Take a convex combination with past sufficient statistics
    #     if self._stochastic_m_step_state is not None:
    #         # Take a convex combination of sufficient statistics from
    #         # this batch and those accumulated thus far.
    #         expected_transitions = convex_combination(
    #             self._stochastic_m_step_state["expected_transitions"],
    #             expected_transitions,
    #             step_size)

    #     # Update the transition matrix using effective suff stats
    #     for row, dist in zip(expected_transitions, self.conditional_dists):
    #         dist.fit_expfam_with_stats((row,), row.sum())

    #     # Save the updated sufficient statistics
    #     self._stochastic_m_step_state = dict(
    #         expected_transitions=expected_transitions
    #     )
