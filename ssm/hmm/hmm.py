from functools import partial
from tqdm.auto import trange
from textwrap import dedent

import jax.numpy as np
import jax.random
from jax import jit, lax
import numpy.random as npr

from ssm.util import ssm_pbar, format_dataset, num_datapoints, Verbosity

from ssm.hmm.initial_state import make_initial_state
from ssm.hmm.transitions import make_transitions
from ssm.hmm.observations import make_observations, OBSERVATION_CLASSES
from ssm.hmm.posteriors import HMMPosterior

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

    initial_state: string specification of initial state distribution.
        Currently, we support the following values:
        - uniform

    transitions: specification of the transition distribution.  This can
        be a string, a transition matrix, or an instance of
        `ssm.hmm.transitions.Transitions`.

        If given a string, it must be one of:
        - "standard" or "stationary": the standard setup with a
        transition matrix that does not change over time.

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
    """.format(observations='\n'.join('\t- ' + key for key in
                                      OBSERVATION_CLASSES.keys()))

    def __init__(self, num_states,
                 initial_state="uniform",
                 transitions="standard",
                 observations="gaussian",
                 initial_state_kwargs={},
                 transition_kwargs={},
                 observation_kwargs={}):
        self.num_states = num_states
        self.initial_state = make_initial_state(
            num_states, initial_state, **initial_state_kwargs)
        self.transitions = make_transitions(
            num_states, transitions, **transition_kwargs)
        self.observations = make_observations(
            num_states, observations, **observation_kwargs)


    @property
    def transition_matrix(self):
        """The transition matrix of the HMM.  This only works if the
        transition distribution is stationary or "standard."  Otherwise,
        use `hmm.transitions.get_transition_matrix(data, **kwargs)`
        """
        from ssm.hmm.transitions import StationaryTransitions
        if not isinstance(self.transitions, StationaryTransitions):
            raise Exception(
                "Can only get transition matrix for \"standard\" "
                "(aka \"stationary\") transitions.  Otherwise, use"
                "`hmm.transitions.get_transition_matrix(data, **kwargs).")
        return self.transitions.get_transition_matrix()

    @transition_matrix.setter
    def transition_matrix(self, value):
        from ssm.hmm.transitions import StationaryTransitions
        if not isinstance(self.transitions, StationaryTransitions):
            raise Exception(
                "Can only set transition matrix for \"standard\" "
                "(aka \"stationary\") transitions.  Otherwise, use"
                "`hmm.transitions.set_transition_matrix(data, **kwargs).")
        return self.transitions.set_transition_matrix(value)

    @property
    def observation_distributions(self):
        """
        A list of distribution objects specifying the conditional
        probability of the data under each discrete latent state.
        """
        return self.observations.observations

    @format_dataset
    def initialize(self, dataset):
        """Initialize parameters based on the given dataset.

        Args:

        dataset: see help(HMM) for details
        """
        self.initial_state.initialize(dataset)
        self.transitions.initialize(dataset)
        self.observations.initialize(dataset)

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
        lp = np.sum([p.marginal_likelihood() for p in posteriors])
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
        lp += np.sum([p.marginal_likelihood() for p in posteriors])
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
        num_states = self.num_states

        # Check the covariates
        if covariates is not None:
            assert covariates.shape[0] == num_timesteps
        else:
            covariates = np.zeros((num_timesteps, 0))

        if prefix is None:
            # Sample the first state and data from the initial distribution
            key1, key2, rng = jax.random.split(rng, 3)
            initial_state = jax.random.choice(key1, num_states,
                                              p=self.initial_state.initial_prob())
            initial_data = self.observation_distributions[initial_state]\
                .sample(key2, preceding_data=None, covariates=covariates[0])
            prefix = (initial_state, initial_data)

        # TODO: write fast code for general transitions
        from ssm.hmm.transitions import StationaryTransitions
        assert isinstance(self.transitions, StationaryTransitions)
        transition_matrix = self.transition_matrix

        # Sample the data and states
        keys = jax.random.split(rng, num_timesteps-1)
        states_and_data = [prefix]
        for t in trange(1, num_timesteps):
            key1, key2 = jax.random.split(keys[t])
            prev_state, prev_data = states_and_data[-1]

            # Sample next latente state
            next_state = jax.random.choice(
                key1, num_states, p=transition_matrix[prev_state])

            # Sample the next data point
            data_dist = self.observation_distributions[next_state]
            next_data = data_dist.sample(
                key2, preceding_data=prev_data, covariates=covariates[t], **kwargs)

            states_and_data.append((next_state, next_data))

        states, data = list(zip(*states_and_data))
        states = np.stack(states)
        data = np.row_stack(data)
        return states, data

        # TODO: Make this work with scan
        # def sample_next(history, prms):
        #     key, covariate = prms
        #     prev_state, prev_data = history
        #     key1, key2 = jax.random.split(key, 2)
        #     next_state = jax.random.choice(key1, num_states,
        #                                    p=transition_matrix[prev_state])
        #     next_data = self.observation_distributions[next_state]\
        #         .sample(key2, preceding_data=prev_data, covariates=covariate)
        #     return (next_state, next_data), None

        # _, (states, data) = lax.scan(sample_next,
        #                              prefix,
        #                              (keys, covariates[1:]))

        # # Append the prefix before returning
        # states = np.concatenate([np.array(prefix[0]), states])
        # data = np.concatenate([np.array(prefix[1]), data])
        # return states, data

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
    def _fit_em(self, dataset, num_iters=100, tol=1e-4, verbosity=2):
        """
        Fit the HMM with expectation-maximization (EM).
        """
        def log_prob(posteriors):
            lp = self.log_prior()
            for p in posteriors:
                lp += p.marginal_likelihood()
            assert np.isfinite(lp)
            return lp / num_datapoints(dataset)

        def e_step(posteriors):
            for p in posteriors:
                p.update()
            return sum([p.marginal_likelihood() for p in posteriors])

        def m_step(posteriors):
            self.initial_state.m_step(dataset, posteriors)
            self.transitions.m_step(dataset, posteriors)
            self.observations.m_step(dataset, posteriors)

        # Run the EM algorithm to convergence
        posteriors = [HMMPosterior(self, data) for data in dataset]
        log_probs = [log_prob(posteriors)]
        pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, log_probs[-1])
        for itr in pbar:
            # Update the posteriors and parameters
            m_step(posteriors)
            e_step(posteriors)
            log_probs.append(log_prob(posteriors))

            # Update progress bar
            if verbosity == 2:
                pbar.set_description("LP: {:.3f}".format(log_probs[-1]))
                pbar.update(1)

            # Check for convergence
            if abs(log_probs[-1] - log_probs[-2]) < tol and itr > 10:
                break

        # Clean up the output before returning
        if len(posteriors) == 1:
            posteriors = posteriors[0]
        return np.array(log_probs), posteriors

    @format_dataset
    def _fit_stochastic_em(self, dataset,
                           validation_dataset=None,
                           num_iters=100,
                           step_size_delay=1.0,
                           step_size_forgetting_rate=0.5,
                           verbosity=Verbosity.LOUD):
        # Initialize the step size schedule
        step_sizes = np.power(np.arange(num_iters * len(dataset)) + step_size_delay,
                              -step_size_forgetting_rate)

        # Make sure the first step size is 1!
        step_sizes = np.concatenate((np.array([0]), step_sizes))

        # Choose random data for each iteration
        total_num_datapoints = num_datapoints(dataset)
        data_indices = npr.choice(len(dataset), size=num_iters)

        def validation_log_prob():
            if validation_dataset is None:
                return np.nan

            lp = self.log_prior()
            for batch in validation_dataset:
                posterior = HMMPosterior(self, batch)
                lp += posterior.marginal_likelihood()
            return lp / num_datapoints(validation_dataset)

        def batch_log_prob(batch, posterior):
            scale_factor = total_num_datapoints / len(batch["data"])
            lp = self.log_prior() + scale_factor * posterior.marginal_likelihood()
            # lp = scale_factor * posterior.marginal_likelihood()
            return lp / total_num_datapoints

        def e_step(batch):
            return HMMPosterior(self, batch)

        def m_step(batch, posterior, step_size):
            scale_factor = total_num_datapoints / len(batch["data"])
            args = batch, [posterior], step_size
            kwargs = dict(scale_factor=scale_factor)
            self.initial_state.stochastic_m_step(*args, **kwargs)
            self.transitions.stochastic_m_step(*args, **kwargs)
            self.observations.stochastic_m_step(*args,  **kwargs)

        # Run stochastic EM algorithm
        batch_log_probs = []
        if verbosity >= Verbosity.LOUD: print("Computing initial log probability...")
        validation_log_probs = [validation_log_prob()]
        if verbosity >= Verbosity.LOUD: print("Done.")

        pbar = ssm_pbar(num_iters, verbosity,
                        "Validation LP: {:.2f} Batch LP {:.2f}",
                        validation_log_probs[0], np.nan)
        for epoch in pbar:
            perm = npr.permutation(len(dataset))
            for batch_idx in range(len(dataset)):
                itr = epoch * len(dataset) + batch_idx

                # grab minibatch for this iteration and perform one em update
                batch = dataset[perm[batch_idx]]
                posterior = e_step(batch)
                m_step(batch, posterior, step_sizes[itr])
                batch_log_probs.append(batch_log_prob(batch, posterior))

                # Update per-batch progress bar
                if verbosity >= 2:
                    pbar.set_description("Validation LP: {:.2f} Batch LP: {:.2f}"\
                        .format(validation_log_probs[-1], batch_log_probs[-1]))

            # Compute complete log prob and update pbar
            validation_log_probs.append(validation_log_prob())
            if verbosity >= 2:
                pbar.set_description("Validation LP: {:2f} Batch LP: {:.2f}"\
                        .format(validation_log_probs[-1], batch_log_probs[-1]))
                pbar.update(1)

        # Finally, compute the posteriors and return
        posteriors = [HMMPosterior(self, batch) for batch in dataset]
        if len(posteriors) == 1:
            posteriors = posteriors[0]

        return np.array(validation_log_probs), posteriors

    @format_dataset
    def fit(self, dataset, method="em", initialize=True,
            verbose=Verbosity.LOUD, **kwargs):
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

        verbose: specify how verbose the print-outs should be.  See
        `ssm.util.Verbosity`.

        **kwargs: keyword arguments are passed to the given fitting method.
        """
        _fitting_methods = dict(
            em=self._fit_em,
            stochastic_em=self._fit_stochastic_em,
            )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            # TODO: allow for kwargs to initialize
            if verbose >= Verbosity.LOUD : print("Initializing...")
            self.initialize(dataset)
            if verbose >= Verbosity.LOUD: print("Done.")

        return _fitting_methods[method](dataset, **kwargs)
