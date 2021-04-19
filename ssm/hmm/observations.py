import copy
import warnings

import jax.numpy as np
import jax.random as jr
from jax import jit
from jax.tree_util import register_pytree_node, register_pytree_node_class

import jxf.distributions as dists

from ssm.util import format_dataset, num_datapoints

@register_pytree_node_class
class Observations(object):
    """
    A thin wrapper for a list of distributions, one associated with
    each discrete state.  The reason for this layer of abstraction is
    that in some cases--e.g. hierarchical models--we want to share
    parameters between the observation distributions.  Wrapping them
    in a single object allows us to do that easily.

    :param: observations can be one of the following:
        - a string indicating the observation type for all states
        - a list of strings with types for each state
        - a list of jxf.ExponentialFamilyDistribution objects,
          already initialized for each state

    """
    def __init__(self,
                 num_states,
                 observations,
                 prior=None,
                 **observation_kwargs):

        self.num_states = num_states
        assert len(observations) == num_states
        self.conditional_dists = observations
        self.prior = prior
        self.observations_kwargs = observation_kwargs

    def tree_flatten(self):
        return ((self.conditional_dists,
                 self.prior,
                 self.observations_kwargs), self.num_states)

    @classmethod
    def tree_unflatten(cls, num_states, children):
        conditional_dists, prior, observation_kwargs = children
        return cls(num_states,
                   conditional_dists,
                   prior=prior,
                   **observation_kwargs)

    @format_dataset
    def initialize(self, rng, dataset, method="kmeans"):
        # initialize assignments and perform one M-step
        num_states = self.num_states
        if method.lower() == "random":
            # randomly assign datapoints to clusters
            keys = jr.split(rng, len(dataset))
            assignments = [jr.choice(key, num_states, shape=data_dict["data"].shape[0])
                           for key, data_dict in zip(keys, dataset)]
        elif method.lower() == "kmeans":
            # cluster the data with kmeans
            from sklearn.cluster import KMeans
            km = KMeans(num_states)
            ind = jr.choice(rng, len(dataset))
            km.fit(dataset[ind]["data"])
            assignments = [km.predict(data_dict["data"])
                           for data_dict in dataset]
        else:
            raise Exception("Observations.initialize: "
                "Invalid initialize method: {}".format(method))

        # Construct subsets of the data and fit the distributions
        def _initialize(idx_and_conditional_dist):
            idx, conditional_dist = idx_and_conditional_dist
            weights = []
            for assignment in assignments:
                weights.append((assignment == idx).astype(np.float32))
            return conditional_dist.fit(dataset, weights)

        self.conditional_dists = \
            list(map(_initialize, enumerate(self.conditional_dists)))

    def permute(self, perm):
        self.conditional_dists = [self.conditional_dists[i] for i in perm]

    def log_prior(self):
        if self.prior is not None:
            raise NotImplementedError
        else:
            return 0.0

    def log_likelihoods(self, data, **kwargs):
        return np.column_stack([obs.log_prob(data, **kwargs) for obs in self.conditional_dists])

    @format_dataset
    def m_step(self, dataset, posteriors):
        def _m_step(idx_and_conditional_dist):
            idx, conditional_dist = idx_and_conditional_dist
            weights = [p.expected_states()[:, idx] for p in posteriors]
            return conditional_dist.fit(dataset, weights, prior=self.prior, **self.observations_kwargs)

        self.conditional_dists = \
            list(map(_m_step, enumerate(self.conditional_dists)))

    @format_dataset
    def initialize_stochastic_em(self, dataset, step_size=0.75):
        initial_state, update_fns, get_distribution_fns = \
            list(zip(*[dist.proximal_optimizer(prior=self.prior,
                                               step_size=step_size,
                                               **self.observations_kwargs)
                       for dist in self.conditional_dists]))
        metadata = num_datapoints(dataset), update_fns, get_distribution_fns
        return metadata, initial_state

    @format_dataset
    def stochastic_m_step(self,
                          itr,
                          dataset,
                          posteriors,
                          metadata,
                          optimizer_state):
        """Perform one M step in the stochastic EM algorithm.
        """
        total_num_datapoints, update_fns, get_distribution_fns = metadata
        scale_factor = total_num_datapoints / num_datapoints(dataset)
        new_opt_state = []
        for idx, (update_fn, state) in enumerate(zip(update_fns, optimizer_state)):
            weights = [p.expected_states()[:, idx] for p in posteriors]
            new_opt_state.append(update_fn(itr,
                                           dataset,
                                           state,
                                           weights=weights,
                                           scale_factor=scale_factor))

        # Update the conditional distributions
        self.conditional_dists = [f(state) for f, state in zip(get_distribution_fns, new_opt_state)]
        return new_opt_state


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
                                              step_size=step_size,
                                              **self.observations_kwargs)
                      for dist in self.conditional_dists]))

        def update(minibatch, posteriors, itr, state):
            scale_factor = total_num_datapoints / num_datapoints(minibatch)
            new_states = []
            for idx, (update_fn, this_state) in enumerate(zip(update_fns, state)):
                weights = [p.expected_states()[:, idx] for p in posteriors]
                new_states.append(update_fn(minibatch,
                                            itr,
                                            this_state,
                                            weights=weights,
                                            scale_factor=scale_factor))

            # Update the conditional distributions
            self.conditional_dists = \
                [f(this_state) for f, this_state in zip(get_distribution_fns, state)]

            return new_states

        return initial_state, update
