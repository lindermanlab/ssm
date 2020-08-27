from functools import partial
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.util import ssm_pbar, format_dataset, num_datapoints

from ssm.hmm.initial_state import make_initial_state
from ssm.hmm.transitions import make_transitions
from ssm.hmm.observations import make_observations
from ssm.hmm.posteriors import HMMPosterior

class HMM(object):
    """
    Hidden Markov model.
    """
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
        """Compute the transition matrix (or transition matrices, if
        the model is nonstationary).  In some cases, the transition matrix
        may depend on the data; e.g. the data may contain covariates that
        modulate the transition probabilities, or the transition probabilities
        may be a function of past observations.
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
        """Compute the transition matrix (or transition matrices, if
        the model is nonstationary).  In some cases, the transition matrix
        may depend on the data; e.g. the data may contain covariates that
        modulate the transition probabilities, or the transition probabilities
        may be a function of past observations.
        """
        from ssm.hmm.transitions import StationaryTransitions
        if not isinstance(self.transitions, StationaryTransitions):
            raise Exception(
                "Can only set transition matrix for \"standard\" "
                "(aka \"stationary\") transitions.  Otherwise, use"
                "`hmm.transitions.set_transition_matrix(data, **kwargs).")
        return self.transitions.set_transition_matrix(value)


    @property
    def observation_distributions(self):
        return self.observations.observations

    @format_dataset
    def initialize(self, dataset):
        """Initialize parameters given data.
        """
        self.initial_state.initialize(dataset)
        self.transitions.initialize(dataset)
        self.observations.initialize(dataset)

    def permute(self, perm):
        """
        Permute the discrete latent states.
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
    def average_log_prob(self, dataset):
        """Compute the log probability of a dataset.  This requires
        performing posterior inference of the latent states.
        """
        posteriors = [HMMPosterior(self, data_dict) for data_dict in dataset]
        lp = self.log_prior()
        lp += np.sum([p.marginal_likelihood() for p in posteriors])
        return lp / num_datapoints(dataset), posteriors

    def sample(self, num_timesteps,
               prefix=None,
               covariates=None,
               **kwargs):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        num_timesteps : int
            number of time steps to sample

        prefix : (state_prefix, data_prefix)
            Optional prefix of discrete states and data
            `state_prefix` must be an array of integers taking values 0...num_states-1.
            `data_prefix` must be an array of the same length that has preceding observations.

        covariates : (T, ...) array_like
            Optional inputs to specify for sampling

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """

        # Check the covariates
        if covariates is not None:
            assert covariates.shape[0] == num_timesteps
        else:
            covariates = np.zeros((num_timesteps, 0))

        # Use prefix if it's given
        if prefix is not None:
            states, data = list(prefix[0]), list(prefix[1])
            assert len(data) == len(states)
        else:
            states, data = [], []
        prefix_len = len(states)

        # Fill in the rest of the data
        for t in range(num_timesteps):
            covariate = covariates[t] if covariates else None

            # Sample the next latent state
            if len(states) == 0:
                p = self.initial_state.initial_prob()
            else:
                p = self.transitions.get_transition_matrix(
                    data=data, covariates=covariate, **kwargs)[states[-1]]
            next_state = npr.choice(self.num_states, p=p)
            states.append(next_state)

            # Sample the next data point
            data_dist = self.observation_distributions[next_state]
            next_data = data_dist.sample(preceding_data=data,
                                         covariates=covariate,
                                         **kwargs)
            data.append(next_data)

        return np.array(states[prefix_len:]), np.row_stack(data[prefix_len:])

    @format_dataset
    def infer_posterior(self, dataset):
        """Compute the posterior distribution for a given dataset using
        this model's parameters.
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
            if abs(log_probs[-1] - log_probs[-2]) < tol:
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
                           verbosity=2):
        """
        Fit the HMM with stochastic expectation-maximization (EM).
        """
        # Initialize the step size schedule
        step_sizes = np.power(np.arange(num_iters) + step_size_delay,
                              -step_size_forgetting_rate)

        # Make sure the first step size is 1!
        step_sizes = np.concatenate(([0], step_sizes))

        # Choose random data for each iteration
        total_num_datapoints = num_datapoints(dataset)
        data_indices = npr.choice(len(dataset), size=num_iters)

        def validation_log_prob():
            if validation_dataset is None:
                return 0

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
        if verbosity >= 2: print("Computing initial log probability...")
        validation_log_probs = [validation_log_prob()]
        if verbosity >= 2: print("Done.")

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
                    pbar.set_description("Epoch LP: {:.2f} Batch LP: {:.2f}"\
                        .format(validation_log_probs[-1], batch_log_probs[-1]))

            # Compute complete log prob and update pbar
            validation_log_probs.append(validation_log_prob())
            if verbosity >= 2:
                pbar.set_description("Validation LP: {:.2f} Batch LP: {:.2f}"\
                        .format(validation_log_probs[-1], batch_log_probs[-1]))
                pbar.update(1)

        # Finally, compute the posteriors and return
        posteriors = [HMMPosterior(self, batch) for batch in dataset]
        if len(posteriors) == 1:
            posteriors = posteriors[0]

        return np.array(validation_log_probs), posteriors

    @format_dataset
    def fit(self, dataset,
            method="em",
            initialize=True,
            verbose=2,
            **kwargs):
        _fitting_methods = dict(
            em=self._fit_em,
            stochastic_em=self._fit_stochastic_em,
            )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            if verbose >= 2: print("Initializing...")
            self.initialize(dataset)
            if verbose >= 2: print("Done.")

        return _fitting_methods[method](dataset, **kwargs)
