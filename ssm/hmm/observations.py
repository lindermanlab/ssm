import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
import ssm.distributions as dists
from ssm.util import format_dataset

# Observations can be one of the following:
# - a string indicating the observation type for all states
# - a list of strings with types for each state
# - a list of Distribution objects, already initialized for each state
OBSERVATION_CLASSES = dict(
    auto_regression=dists.LinearAutoRegression,
    bernoulli=dists.Bernoulli,
    beta=dists.Beta,
    binomial=dists.Binomial,
    categorical=dists.Categorical,
    dirichlet=dists.Dirichlet,
    gamma=dists.Gamma,
    gaussian=dists.MultivariateNormal,
    linear_regression=dists.LinearRegression,
    multivariate_normal=dists.MultivariateNormal,
    multivariate_t=dists.MultivariateStudentsT,
    multivariate_t_regression=dists.MultivariateStudentsTLinearRegression,
    multivariate_t_auto_regression=dists.MultivariateStudentsTAutoRegression,
    mvn=dists.MultivariateNormal,
    normal=dists.Normal,
    robust_regression=dists.MultivariateStudentsTLinearRegression,
    robust_auto_regression=dists.MultivariateStudentsTAutoRegression,
    poisson=dists.Poisson,
    students_t=dists.StudentsT,
)

def make_observations(num_states, observations, **observation_kwargs):
    def _check(obs_name):
        assert obs_name in OBSERVATION_CLASSES, \
            "`observations` must be one of: {}".format(OBSERVATION_CLASSES.keys())

    def _convert(obs):
        if isinstance(obs, str):
            _check(obs)
            return OBSERVATION_CLASSES[obs.lower()]
        elif isinstance(obs, dists.Distribution):
            return obs
        else:
            raise Exception("`observations` must be either strings or "
                            "Distribution instances")

    if isinstance(observations, str):
        observations = [_convert(observations)] * num_states
    elif isinstance(observations, (tuple, list)):
        assert len(observations) == num_states
        observations = list(map(_convert, observations))
    else:
        raise Exception("Expected `observations` to be a string, a list of "
                        "strings, or a list of Distribution objects")

    return Observations(observations, **observation_kwargs)


class Observations(object):
    """
    A thin wrapper for a list of distributions, one associated with
    each discrete state.  The reason for this layer of abstraction is
    that in some cases--e.g. hierarchical models--we want to share
    parameters between the observation distributions.  Wrapping them
    in a single object allows us to do that easily.
    """
    def __init__(self, observations, **observation_kwargs):
        assert isinstance(observations, list)
        self.num_states = len(observations)
        self.observations = observations
        self.observations_kwargs = observation_kwargs

        # Initialize class variables for stochastic_m_step
        self._stochastic_m_step_state = [None] * self.num_states

    @property
    def is_built(self):
        return all([isinstance(o, dists.Distribution) for o in self.observations])

    @format_dataset
    def build_from_example(self, dataset):
        if not self.is_built:
            kwargs = dataset[0].copy()
            kwargs.update(self.observations_kwargs)
            self.observations = [
                obs_class.from_example(**kwargs)
                for obs_class in self.observations
            ]

    @format_dataset
    def initialize(self, dataset, method="kmeans"):
        # use the first datapoint to get the shape
        if not self.is_built:
            self.build_from_example(dataset)

        # initialize assignments and perform one M-step
        num_states = self.num_states
        if method.lower() == "random":
            # randomly assign datapoints to clusters
            assignments = [npr.choice(num_states,
                                    size=data_dict["data"].shape[0])
                        for data_dict in dataset]
        elif method.lower() == "kmeans":
            # cluster the data with kmeans
            from sklearn.cluster import KMeans
            km = KMeans(num_states)
            ind = npr.choice(len(dataset))
            km.fit(dataset[ind]["data"])
            assignments = [
                km.predict(data_dict["data"])
                for data_dict in dataset]
        else:
            raise Exception("Observations.initialize: "
                "Invalid initialize method: {}".format(method))

        # Construct subsets of the data and fit the distributions
        for idx, observation_dist in enumerate(self.observations):
            data_subsets = []
            for data_dict, assignment in zip(dataset, assignments):
                n = data_dict["data"].shape[0]
                subset = dict()
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                        subset[k] = v[assignment == idx]
                    else:
                        subset[k] = v
                data_subsets.append(subset)

            observation_dist.fit(data_subsets)

        # Finally, precompute sufficient statistics
        for observation_dist in self.observations:
            observation_dist.preprocess(dataset)

    def permute(self, perm):
        self.observations = [self.observations[i] for i in perm]

    def log_prior(self):
        return np.sum([observation_dist.log_prior()
                       for observation_dist in self.observations])

    def log_likelihoods(self, data, **kwargs):
        assert self.is_built
        return np.column_stack([obs.log_prob(data, **kwargs)
                                for obs in self.observations])

    @format_dataset
    def m_step(self, dataset, posteriors, **kwargs):
        assert self.is_built
        for idx, observation_dist in enumerate(self.observations):
            weighted_dataset = []
            for data_dict, posterior in zip(dataset, posteriors):
                weighted_data = data_dict.copy()
                weighted_data["weights"] = posterior.expected_states()[:, idx]
                weighted_dataset.append(weighted_data)
            observation_dist.fit(weighted_dataset)

    @format_dataset
    def stochastic_m_step(self, dataset, posteriors, step_size,
                          scale_factor=1.0, **kwargs):
        assert self.is_built

        # Update each distribution
        new_optimizer_states = []
        for idx, (observation_dist, optimizer_state) in \
            enumerate(zip(self.observations, self._stochastic_m_step_state)):
            weighted_dataset = []
            for data_dict, posterior in zip(dataset, posteriors):
                weighted_data = data_dict.copy()
                weighted_data["weights"] = posterior.expected_states()[:, idx]
                weighted_dataset.append(weighted_data)

            new_optimizer_states.append(
                observation_dist.fit_proximal(
                    weighted_dataset, step_size,
                    optimizer_state=optimizer_state,
                    scale_factor=scale_factor))

        self._stochastic_m_step_state = new_optimizer_states
