from functools import partial
from tqdm.auto import trange
from textwrap import dedent
from time import time_ns

import jax.numpy as np
import jax.random as jr
from jax import jit, lax
from jax.tree_util import register_pytree_node, register_pytree_node_class

import jxf.distributions as dists
import jxf.regressions as regrs

from ssm.hmm.initial_state import InitialState, UniformInitialState
from ssm.hmm.transitions import Transitions, StationaryTransitions
from ssm.hmm.observations import Observations
from ssm.hmm.posteriors import HMMPosterior
from ssm.util import ssm_pbar, format_dataset, num_datapoints, Verbosity


_INITIAL_STATE_CLASSES = dict(
    uniform=UniformInitialState,
)

_TRANSITION_CLASSES = dict(
    standard=StationaryTransitions,
    stationary=StationaryTransitions,
)

_OBSERVATION_CLASSES = dict(
    bernoulli=dists.Bernoulli,
    beta=dists.Beta,
    binomial=dists.Binomial,
    categorical=dists.Categorical,
    dirichlet=dists.Dirichlet,
    gamma=dists.Gamma,
    gaussian=dists.MultivariateNormalFullCovariance,
    multivariate_normal=dists.MultivariateNormalFullCovariance,
    # multivariate_t=dists.MultivariateStudentsT,
    mvn=dists.MultivariateNormalFullCovariance,
    normal=dists.Normal,
    poisson=dists.Poisson,
    students_t=dists.StudentT,
)


@register_pytree_node_class
class HMM(object):
    __doc__ = """
    A Hidden Markov model is a probabilistic state space model with
    discrete latent states that change over time.

    Dataset formatting:

    The methods of this class consume a `dataset`, which can be given
    in a number of ways:
    - A numpy array where the first dimension is the number
        of time steps.

    - A dictionary with a `data` key and numpy array value.
        The dictionary may also include other properties of the data,
        which may affect the transition or observation distributions.
        For example, it may include `covariates`.

    - A list of dictionaries, one for each "batch" of data.  Each
        dictionary must be formatted as above.

    Args:

    num_states: integer number of discrete latent states.

    initial_state: specification of initial state distribution.

        If given a string, it must be one of: \n{initial_states}

    transitions: specification of the transition distribution.  This can
        be a string, a transition matrix, or an instance of
        `ssm.hmm.transitions.Transitions`.

        If given a string, it must be one of: \n{transitions}

        If given a matrix, it must be shape `(num_states, num_states)` and
        be row-stochastic (i.e. non-negative with rows that sum to 1).

        If given an object, it must be an instance of
        `ssm.hmm.transitions.Transitions.`

    observations: specification of the transition distribution.  This can
        be a string or a list of `ssm.distributions.Distribution` objects
        of length number of states.

        If given a string, it must be one of: \n{observations}

        Otherwise, it must be a list of `ssm.distributions.Distribution`
        objects of length number of states.

    initial_state_kwargs:  keyword arguments passed to the `InitialState`
        constructor.

    transitions_kwargs:  keyword arguments passed to the `Transitions`
        constructor.

    observation_kwargs:  keyword arguments passed to the `Observations`
        constructor.
    """.format(
        initial_states='\n'.join('\t- ' + key for key in _INITIAL_STATE_CLASSES.keys()),
        transitions='\n'.join('\t- ' + key for key in _TRANSITION_CLASSES.keys()),
        observations='\n'.join('\t- ' + key for key in _OBSERVATION_CLASSES.keys())
        )

    def __init__(self, num_states,
                 initial_state="uniform",
                 initial_state_kwargs={},
                 transitions="standard",
                 transitions_prior=None,
                 transition_kwargs={},
                 observations="gaussian",
                 observations_prior=None,
                 observation_kwargs={}):
        self.num_states = num_states
        self.initial_state = self._build_initial_state(
            num_states, initial_state, **initial_state_kwargs)
        self.transitions = self._build_transitions(
            num_states, transitions, transitions_prior, **transition_kwargs)
        self.observations = self._build_observations(
            num_states, observations, observations_prior, **observation_kwargs)

    def tree_flatten(self):
        return ((self.initial_state,
                 self.transitions,
                 self.observations), self.num_states)

    @classmethod
    def tree_unflatten(cls, num_states, children):
        initial_state, transitions, observations = children
        return cls(num_states,
                   initial_state=initial_state,
                   transitions=transitions,
                   observations=observations)

    def _build_initial_state(self, num_states,
                              initial_state,
                              **initial_state_kwargs):
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

    def _build_transitions(self, num_states,
                            transitions,
                            transitions_prior,
                            **transitions_kwargs):

        """Helper function to construct transitions of the desired type.
        """
        if isinstance(transitions, np.ndarray):
            # Assume this is a transition matrix
            return StationaryTransitions(num_states,
                                        transition_matrix=transitions)

        elif isinstance(transitions, str):
            # String specifies class of transitions
            assert transitions.lower() in _TRANSITION_CLASSES, \
                "`transitions` must be one of {}".format(_TRANSITION_CLASSES.keys())
            transition_class = _TRANSITION_CLASSES[transitions.lower()]
            return transition_class(num_states, **transitions_kwargs)
        else:
            # Otherwise, we need a Transitions object
            assert isinstance(transitions, Transitions)
            return transitions

    def _build_observations(self, num_states,
                             observations,
                             observations_prior,
                             **observations_kwargs):

        # convert observations into list of Distribution classes (or instances)
        def _check(obs_name):
            assert obs_name in _OBSERVATION_CLASSES, \
                "`observations` must be one of: {}".format(_OBSERVATION_CLASSES.keys())

        def _convert(obs):
            if isinstance(obs, str):
                _check(obs)
                return _OBSERVATION_CLASSES[obs.lower()]
            elif isinstance(obs, (dists.ExponentialFamilyDistribution, dists.CompoundDistribution)):
                # It's a JXF distribution
                return obs
            else:
                raise Exception("`observations` must be either strings or "
                                "Distribution instances")

        if isinstance(observations, str):
            observations = [_convert(observations)] * num_states
            return Observations(num_states,
                                observations,
                                observations_prior,
                                **observations_kwargs)

        elif isinstance(observations, (tuple, list)):
            assert len(observations) == num_states
            observations = list(map(_convert, observations))
            return Observations(num_states,
                                observations,
                                observations_prior,
                                **observations_kwargs)
        else:
            assert isinstance(observations, Observations)
            return observations

    @property
    def transition_matrix(self):
        """The transition matrix of the HMM.  This only works if the
        transition distribution is stationary or "standard."  Otherwise,
        use `hmm.transitions.get_transition_matrix(data, **kwargs)`
        """
        if not isinstance(self.transitions, StationaryTransitions):
            raise Exception(
                "Can only get transition matrix for \"standard\" "
                "(aka \"stationary\") transitions.  Otherwise, use"
                "`hmm.transitions.get_transition_matrix(data, **kwargs).")
        return self.transitions.transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value):
        if not isinstance(self.transitions, StationaryTransitions):
            raise Exception(
                "Can only set transition matrix for \"standard\" "
                "(aka \"stationary\") transitions.  Otherwise, use"
                "`hmm.transitions.set_transition_matrix(data, **kwargs).")
        self.transitions.transition_matrix = value

    @property
    def observation_distributions(self):
        """
        A list of distribution objects specifying the conditional
        probability of the data under each discrete latent state.
        """
        return self.observations.conditional_dists

    @format_dataset
    def initialize(self, rng, dataset):
        """Initialize parameters based on the given dataset.

        Args:

        dataset: see help(HMM) for details
        """
        keys = jr.split(rng, 3)
        components = [self.initial_state, self.transitions, self.observations]
        for key, component in zip(keys, components):
            component.initialize(key, dataset)

    def permute(self, perm):
        """
        Permute the discrete latent states.

        Args:
        perm: a numpy array of integers {0, ..., num_states - 1}
            in the new desired order.
        """
        assert np.all(np.sort(perm) == np.arange(self.num_states))
        self.initial_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def log_prior(self):
        return self.initial_state.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @format_dataset
    def average_log_likelihood(self, dataset):
        """Compute the average log likelihood of data in this dataset.
        This does not include the prior probability of the model parameters.

        Note: This requires performing posterior inference of the latent states,
        so in addition to the log probability we return the posterior distributions.

        Args:

        dataset: see help(HMM) for details
        """
        posteriors = [HMMPosterior(self, data_dict) for data_dict in dataset]
        lp = sum([p.marginal_likelihood() for p in posteriors])
        return lp / num_datapoints(dataset), posteriors

    @format_dataset
    def average_log_prob(self, dataset):
        """Compute the average log probability of data in this dataset.

        Note: This requires performing posterior inference of the latent states,
        so in addition to the log probability we return the posterior distributions.

        Args:

        dataset: see help(HMM) for details
        """
        posteriors = [HMMPosterior(self, data_dict) for data_dict in dataset]
        lp = self.log_prior()
        lp += sum([p.marginal_likelihood() for p in posteriors])
        return lp / num_datapoints(dataset), posteriors

    def sample(self, rng, num_timesteps, prefix=None, covariates=None, **kwargs):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Args
        num_timesteps: integer number of time steps to sample

        prefix: optional tuple of (state_prefix, data_prefix)
        - `state_prefix` must be an array of integers taking values
           {0...num_states-1.
        - `data_prefix` must be an array of the same length that has preceding
          observations.

        covariates: optional array with leading dimension `num_timesteps` that
        specifies the covariates necessary for sampling.

        Returns:

        states: a numpy array with the sampled discrete states

        data: a numpy array of sampled data
        """
        # num_states = self.num_states

        # # make dummy covariates if necessary
        # if covariates is not None:
        #     assert covariates.shape[0] == num_timesteps
        # else:
        #     covariates = np.zeros((num_timesteps, 1))

        # # sample the initial state if necessary
        # if prefix is None:
        #     # Sample the first state and data from the initial distribution
        #     key1, key2, rng = jax.random.split(rng, 3)
        #     initial_state = jax.random.choice(key1, num_states,
        #                                       p=self.initial_state.initial_prob())
        #     initial_data = self.observation_distributions[initial_state]\
        #         .sample(seed=key2, preceding_data=None, covariates=covariates[0])
        #     prefix = (initial_state, initial_data)

        # TODO: write fast code for general transitions
        # assert isinstance(self.transitions, StationaryTransitions)
        # transition_matrix = self.transition_matrix

        # # Sample one step at a time with lax.scan
        # keys = jax.random.split(rng, num_timesteps-1)
        # def sample_next(history, prms):
        #     prev_state, prev_data = history
        #     key, covariate = prms
        #     key1, key2 = jax.random.split(key, 2)
        #     next_state = self.transitions.sample(key1, prev_state)

        #     sample_funcs = [
        #         partial(dist.sample, preceding_data=prev_data, covariates=covariate)
        #         for dist in self.observation_distributions]
        #     next_data = lax.switch(next_state, sample_funcs, key2)
        #     return (next_state, next_data), (next_state, next_data)

        # # Sample the data
        # _, (states, data) = lax.scan(sample_next, prefix, (keys, covariates[1:]))

        # # Append the prefix before returning
        # states = np.concatenate([np.array([prefix[0]]), states])
        # data = np.row_stack([prefix[1], data])
        # return states, data

        # Sample initial state
        rng_init, rng = jr.split(rng, 2)
        initial_state = jr.choice(rng_init, self.num_states)

        # Precompute sample functions for each observation and transition distribution
        def _sample(d): return lambda seed: d.sample(seed=seed)
        trans_sample_funcs = [_sample(d) for d in self.transitions.conditional_dists]
        obs_sample_funcs = [_sample(d) for d in self.observations.conditional_dists]

        # Sample one step at a time with lax.scan
        keys = jr.split(rng, num_timesteps)
        def sample_next(curr_state, key):
            key1, key2 = jr.split(key, 2)

            # Sample observation
            curr_obs = lax.switch(curr_state, obs_sample_funcs, key1)

            # Sample next state
            next_state = lax.switch(curr_state, trans_sample_funcs, key2)
            return next_state, (curr_state, curr_obs)

        _, (states, data) = lax.scan(sample_next, initial_state, keys)
        return states, data

    @format_dataset
    def infer_posterior(self, dataset):
        """Compute the posterior distribution for a given dataset using
        this model's parameters.

        Args:

        dataset: see help(HMM) for details
        """
        posteriors = [HMMPosterior(self, data) for data in dataset]
        return posteriors[0] if len(posteriors) == 1 else posteriors

    @format_dataset
    def _fit_em(self, rng, dataset, num_iters=100, tol=1e-4, verbosity=Verbosity.LOUD):
        """
        Fit the HMM with expectation-maximization (EM).
        """
        @jit
        def step(model):
            # E Step
            posteriors = [HMMPosterior(model, data) for data in dataset]

            # Compute log probability
            lp = model.log_prior()
            lp += sum([p.marginal_likelihood() for p in posteriors])

            # M Step
            model.initial_state.m_step(dataset, posteriors)
            model.transitions.m_step(dataset, posteriors)
            model.observations.m_step(dataset, posteriors)

            return model, lp / num_datapoints(dataset)

        # Run the EM algorithm to convergence
        model = self
        log_probs = [np.nan]
        pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, log_probs[-1])
        for itr in pbar:
            model, lp = step(model)
            log_probs.append(lp)
            assert np.isfinite(lp) #and (log_probs[-1] >= log_probs[-2] or np.isnan(log_probs[-2]))

            # Update progress bar
            if verbosity >= Verbosity.LOUD:
                pbar.set_description("LP: {:.3f}".format(lp))
                pbar.update(1)

            # Check for convergence
            if abs(log_probs[-1] - log_probs[-2]) < tol and itr > 1:
                break

        # Copy over the final model parameters
        self.initial_state = model.initial_state
        self.transitions = model.transitions
        self.observations = model.observations

        # Compute the posterior distribution(s) with the optimized parameters
        posteriors = [HMMPosterior(self, data) for data in dataset] \
            if len(dataset) > 1 else HMMPosterior(self, dataset[0])

        return np.array(log_probs), posteriors

    @format_dataset
    def _fit_stochastic_em(self, rng, dataset,
                           validation_dataset=None,
                           num_iters=100,
                           step_size_delay=1.0,
                           step_size_forgetting_rate=0.5,
                           verbosity=Verbosity.LOUD):

        # Initialize the step size schedule
        step_size = lambda itr: itr ** (-step_size_forgetting_rate)

        # Initialize the stochastic EM states for each component of the model
        components = [self.transitions, self.observations]
        metadata, optimizer_state = \
            list(zip(*[component.initialize_stochastic_em(dataset, step_size)
                       for component in components]))

        # @jit
        def validation_log_prob(model):
            if validation_dataset is None:
                return np.nan

            lp = self.log_prior()
            lp += sum([HMMPosterior(model, data).marginal_likelihood()
                       for data in validation_dataset])
            return lp / num_datapoints(validation_dataset)

        # @jit
        def step(model, itr, minibatch, optimizer_state):
            # E Step
            posteriors = [HMMPosterior(model, data) for data in minibatch]

            # Compute log probability on this batch
            lp = model.log_prior()
            lp += sum([p.marginal_likelihood() for p in posteriors])

            # M Step
            components = [model.transitions, model.observations]
            new_optimizer_state = []
            for component, meta, state in zip(components, metadata, optimizer_state):
                new_optimizer_state.append(
                    component.stochastic_m_step(itr,
                                                minibatch,
                                                posteriors,
                                                meta,
                                                state))

            return model, lp / num_datapoints(minibatch), new_optimizer_state

        # Run stochastic EM algorithm
        model = self
        batch_log_probs = []
        validation_log_probs = []
        pbar = ssm_pbar(num_iters, verbosity, "Batch LP {:.2f}", np.nan)
        for epoch in pbar:
            rng, this_rng = jr.split(rng, 2)
            perm = jr.permutation(this_rng, len(dataset))
            for batch_idx in range(len(dataset)):
                itr = epoch * len(dataset) + batch_idx

                # grab minibatch for this iteration and perform one em update
                minibatch = [dataset[perm[batch_idx]]]
                model, lp, optimizer_state = step(model, itr, minibatch, optimizer_state)
                batch_log_probs.append(lp)
                if not np.isfinite(lp):
                    print("WARNING: lp not finite!")
                # assert np.isfinite(lp), f"lp not finite: {lp}"

                # Compute complete log prob and update pbar
                if verbosity >= Verbosity.LOUD:
                    pbar.set_description("Batch LP: {:.2f} ({:d}/{:d})"\
                        .format(lp, batch_idx+1, len(dataset)))

            # Each epoch, compute the likelihood of the validation data
            validation_log_probs.append(validation_log_prob(model))
            pbar.update(1)

        # Copy over the final model parameters
        self.initial_state = model.initial_state
        self.transitions = model.transitions
        self.observations = model.observations

        # Finally, compute the posteriors and return
        posteriors = [HMMPosterior(self, data) for data in dataset]
        if len(posteriors) == 1:
            posteriors = posteriors[0]

        if validation_dataset is not None:
            return np.array(batch_log_probs), np.array(validation_log_probs), posteriors
        else:
            return np.array(batch_log_probs), posteriors

    @format_dataset
    def fit(self,
            dataset,
            method="em",
            rng=None,
            initialize=True,
            verbosity=Verbosity.LOUD,
            **kwargs):
        """
        Fit the parameters of the HMM using the specified method.

        Args:

        dataset: see `help(HMM)` for details.

        method: specification of how to fit the data.  Must be one
        of the following strings:
        - em
        - stochastic_em

        initialize: boolean flag indicating whether to initialize the
        model before running the specified method.

        rng: jax.PRNGKey for random initialization and/or fitting

        verbosity: specify how verbose the print-outs should be.  See
        `ssm.util.Verbosity`.

        **kwargs: keyword arguments are passed to the given fitting method.
        """
        make_rng = (rng is None)
        if make_rng:
            rng = jr.PRNGKey(time_ns())

        _fitting_methods = dict(
            em=self._fit_em,
            stochastic_em=self._fit_stochastic_em,
            )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            # TODO: allow for kwargs to initialize
            rng_init, rng = jr.split(rng, 2)
            if verbosity >= Verbosity.LOUD : print("Initializing...")
            self.initialize(rng_init, dataset)
            if verbosity >= Verbosity.LOUD: print("Done.")

        # Run the fitting algorithm
        results = _fitting_methods[method](rng, dataset, **kwargs)
        return (rng, results) if make_rng else results


_ARHMM_OBSERVATION_CLASSES = dict(
    gaussian=regrs.GaussianLinearRegression,
)

@register_pytree_node_class
class AutoregressiveHMM(HMM):
    r"""A hidden Markov model in which the observations at time :math:`t` depends
    on past observations at times :math:`t - \ell` for :math:`\ell=1,\ldots,L`,
    where :math:`L` is the number of lags.

    In addition to the arguments for the regular HMM, you must specify the number
    of past timesteps that the ARHMM depends on.

    We need to write a slightly different `sample` function for this model.
    """
    def __init__(self,
                 num_states,
                 num_lags,
                 initial_state="uniform",
                 initial_state_kwargs={},
                 transitions="standard",
                 transitions_prior=None,
                 transition_kwargs={},
                 observations="gaussian",
                 observations_prior=None,
                 observation_kwargs={}):
        self.num_lags = num_lags
        super(AutoregressiveHMM, self).__init__(
            num_states,
            initial_state=initial_state,
            initial_state_kwargs=initial_state_kwargs,
            transitions=transitions,
            transition_kwargs=transition_kwargs,
            observations=observations,
            observation_kwargs=observation_kwargs)

    def tree_flatten(self):
        aux_data = (self.num_states, self.num_lags)
        children = (self.initial_state, self.transitions, self.observations)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, num_lags = aux_data
        initial_state, transitions, observations = children
        return cls(num_states, num_lags,
                   initial_state=initial_state,
                   transitions=transitions,
                   observations=observations)

    def _build_observations(self, num_states,
                             observations,
                             observations_prior,
                             **observations_kwargs):

        # convert observations into list of Distribution classes (or instances)
        def _check(obs_name):
            assert obs_name in _ARHMM_OBSERVATION_CLASSES, \
                "`observations` must be one of: {}".format(_ARHMM_OBSERVATION_CLASSES.keys())

        def _convert(obs):
            if isinstance(obs, str):
                _check(obs)
                return _ARHMM_OBSERVATION_CLASSES[obs.lower()]
            elif isinstance(obs, regrs.GaussianLinearRegression):
                # It's a JXF distribution
                return obs
            else:
                raise Exception("`observations` must be either strings or "
                                "Distribution instances")

        if isinstance(observations, str):
            observations = [_convert(observations)] * num_states
            return Observations(num_states,
                                observations,
                                observations_prior,
                                **observations_kwargs)

        elif isinstance(observations, (tuple, list)):
            assert len(observations) == num_states
            observations = list(map(_convert, observations))
            return Observations(num_states,
                                observations,
                                observations_prior,
                                **observations_kwargs)
        else:
            assert isinstance(observations, Observations)
            return observations

    def sample(self, rng, num_timesteps, prefix=None, covariates=None, **kwargs):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Args
            rng: jax.random.PRNGKey
            num_timesteps: integer number of time steps to sample

        Returns:
            states: a numpy array with the sampled discrete states
            data: a numpy array of sampled data
        """
        # Sample initial state
        rng_init, rng = jr.split(rng, 2)
        initial_state = jr.choice(rng_init, self.num_states)

        # Precompute sample functions for each observation and transition distribution
        def _sample_trans(d): return lambda seed: d.sample(seed=seed)
        trans_sample_funcs = [_sample_trans(d) for d in self.transitions.conditional_dists]

        def _sample_obs(d): return lambda args: d.sample(*args)
        obs_sample_funcs = [_sample_obs(d) for d in self.observations.conditional_dists]

        # Sample one step at a time with lax.scan
        keys = jr.split(rng, num_timesteps)
        def sample_next(carry, key):
            # unpack the history
            curr_state, past_data = carry
            # initialize rng keys
            key1, key2 = jr.split(key, 2)
            # Sample observation
            covariates = regrs.make_next_autoregression_covariate(past_data,
                                                                 self.num_lags,
                                                                 fit_intercept=True)
            curr_obs = lax.switch(curr_state, obs_sample_funcs, (covariates, key1))
            # Sample next state
            next_state = lax.switch(curr_state, trans_sample_funcs, key2)
            # update the carry
            new_carry = (next_state, np.row_stack([past_data[1:], curr_obs]))
            return new_carry, (curr_state, curr_obs)


        carry = (initial_state, np.zeros((self.num_lags, self.observation_distributions[0].data_dimension)))
        _, (states, data) = lax.scan(sample_next, carry, keys)
        return states, data

    @format_dataset
    def preprocess(self, dataset):
        """Preprocess the dataset to construct features for the autoregressive model.
        """
        return list(map(
            lambda data_dict: \
                regrs.preprocess_autoregression_data(**data_dict,
                                                     num_lags=self.num_lags,
                                                     fit_intercept=True),
            dataset))
