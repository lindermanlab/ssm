import copy
import warnings
from functools import partial
import tqdm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.primitives import hmm_normalizer, hmm_expected_states, hmm_filter
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_elbo_args_are_lists, adam_with_convergence_check

class _HMM(object):
    """
    Base class for hidden Markov models.

    Notation:
    K: number of discrete latent states
    D: dimensionality of observations
    M: dimensionality of inputs

    In the code we will sometimes refer to the discrete
    latent state sequence as z and the data as x.
    """
    def __init__(self, K, D, M, init_state_distn, transitions, observations):
        self.K, self.D, self.M = K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.observations = observations
        
        self._fitting_methods = \
            dict(sgd=partial(self._fit_sgd, "sgd"),
                 adam=partial(self._fit_sgd, "adam"),
                 em=self._fit_em)

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.observations.params
    
    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """
        Initialize parameters given data.
        """
        self.init_state_distn.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.transitions.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.observations.initialize(datas, inputs=inputs, masks=masks, tags=tags)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        K, D = self.K, self.D

        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            data = np.zeros((T+1, D))
            input = np.zeros((T+1, self.M)) if input is None else input
            mask = np.ones((T+1, D), dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data, input, mask, tag))
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observations.sample_x(z[0], data[:0], with_noise=with_noise)
        
        else:
            zhist, xhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            assert xhist.shape == (pad, D)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            data = np.concatenate((xhist, np.zeros((T, D))))
            input = np.zeros((T+pad, self.M)) if input is None else input
            mask = np.ones((T+pad, D), dtype=bool)

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            Pt = np.exp(self.transitions.log_transition_matrices(data[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            data[t] = self.observations.sample_x(z[t], data[:t], input=input[t], tag=tag, with_noise=with_noise)

        return z[pad:], data[pad:]

    @ensure_args_not_none
    def expected_states(self, data, input=None, mask=None, tag=None):
        log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
        log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @ensure_args_not_none
    def most_likely_states(self, data, input=None, mask=None, tag=None):
        Ez, _, _ = self.expected_states(data, input, mask, tag)
        return np.argmax(Ez, axis=1)

    @ensure_args_not_none
    def filter(self, data, input=None, mask=None, tag=None):
        log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
        log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return hmm_filter(log_pi0, log_Ps, log_likes)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(data, input, mask)
        return self.observations.smooth(Ez, data, input, tag)
        
    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """  
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current 
        model parameters.
        
        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
            log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)
            ll += hmm_normalizer(log_pi0, log_Ps, log_likes)
            assert np.isfinite(ll)
        return ll

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        return self.log_likelihood(datas, inputs, masks, tags) + self.log_prior()

    def expected_log_probability(self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current 
        model parameters.
        
        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        elp = self.log_prior()
        for (Ez, Ezzp1), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
            log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)

            # Compute the expected log probability 
            elp += np.sum(Ez[0] * log_pi0)
            elp += np.sum(Ezzp1 * log_Ps)
            elp += np.sum(Ez * log_likes)
            assert np.isfinite(elp)
        return elp
    
    # Model fitting
    def _fit_sgd(self, optimizer, datas, inputs, masks, tags, print_intvl=10, **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data.shape[0] for data in datas])
        
        def _objective(params, itr):
            self.params = params
            obj = self.log_probability(datas, inputs, masks, tags)
            return -obj / T

        lls = []
        def _print_progress(params, itr, g):
            lls.append(self.log_probability(datas, inputs, masks, tags)._value)
            if itr % print_intvl == 0:
                print("Iteration {}.  LL: {}".format(itr, lls[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam, adam_with_convergence_check=adam_with_convergence_check)
        self.params = \
            optimizers[optimizer](grad(_objective), self.params, callback=_print_progress, **kwargs)

        return lls

    def _fit_em(self, datas, inputs, masks, tags, num_em_iters=100, **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        pbar = tqdm.trange(num_em_iters)

        lls = [self.log_probability(datas, inputs, masks, tags)]
        pbar.set_description("LP: {:.1f}".format(lls[-1]))
        
        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask, tag) 
                            for data, input, mask, tag in zip(datas, inputs, masks, tags)]

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step(expectations, datas, inputs, masks, tags, **kwargs)
            self.transitions.m_step(expectations, datas, inputs, masks, tags, **kwargs)
            self.observations.m_step(expectations, datas, inputs, masks, tags, **kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))
            pbar.set_description("LP: {:.1f}".format(lls[-1]))

        return lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None, method="sgd", initialize=True, **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        return self._fitting_methods[method](datas, inputs=inputs, masks=masks, tags=tags, **kwargs)


class _SwitchingLDS(object):
    """
    Switching linear dynamical system fit with 
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, K, D, M, init_state_distn, transitions, dynamics, emissions):
        self.N, self.K, self.D, self.M = N, K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions
        
        # Only allow fitting by SVI
        self._fitting_methods = dict(svi=self._fit_svi)

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params, \
               self.emissions.params
    
    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.dynamics.params = value[2]
        self.emissions.params = value[3]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25, verbose=False):
        # First initialize the observation model
        self.emissions.initialize(datas, inputs, masks, tags)

        # Get the initialized variational mean for the data
        xs = [self.emissions.initialize_variational_params(data, input, mask, tag)[0]
              for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        xmasks = [np.ones_like(x, dtype=bool) for x in xs]

        # Now run a few iterations of EM on a ARHMM with the variational mean
        print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
        arhmm = _HMM(self.K, self.D, self.M, 
                     copy.deepcopy(self.init_state_distn),
                     copy.deepcopy(self.transitions),
                     copy.deepcopy(self.dynamics))

        arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags, 
                  method="em", num_em_iters=num_em_iters, num_iters=10, verbose=verbose)

        self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        self.transitions = copy.deepcopy(arhmm.transitions)
        self.dynamics = copy.deepcopy(arhmm.observations)
        print("Done")

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.dynamics.permute(perm)
        self.emissions.permute(perm)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """  
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.dynamics.log_prior() + \
               self.emissions.log_prior()

    def sample(self, T, input=None, tag=None):
        K, D = self.K, self.D
        input = np.zeros((T, self.M)) if input is None else input
        mask = np.ones((T, D), dtype=bool)
        
        # Initialize outputs
        z = np.zeros(T, dtype=int)
        x = np.zeros((T, D))
        
        # Sample discrete and continuous latent states
        pi0 = np.exp(self.init_state_distn.log_initial_state_distn(x, input, mask, tag))
        z[0] = npr.choice(self.K, p=pi0)
        x[0] = self.dynamics.sample_x(z[0], x[:0], tag=tag)

        for t in range(1, T):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x(z[t], x[:t], input=input[t], tag=tag)

        # Sample observations given latent states
        y = self.emissions.sample_y(z, x, input=input, tag=tag)
        return z, x, y

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        log_pi0 = self.init_state_distn.log_initial_state_distn(variational_mean, input, mask, tag)
        log_Ps = self.transitions.log_transition_matrices(variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        Ez, _ = self.expected_states(variational_mean, data, input, mask, tag)
        return np.argmax(Ez, axis=1)

    @ensure_slds_args_not_none
    def smooth(self, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _ = self.expected_states(variational_mean, data, input, mask, tag)
        return self.emissions.smooth(Ez, variational_mean, data, input, tag)

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Cannot compute exact marginal log probability for the SLDS. "
                      "the ELBO instead.")
        return np.nan

    @ensure_elbo_args_are_lists
    def elbo(self, variational_params, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for data, input, mask, tag, (q_mu, q_sigma_inv) in \
            zip(datas, inputs, masks, tags, variational_params):

            q_sigma = np.exp(q_sigma_inv)
            for sample in range(n_samples):
                # log p(theta)
                elbo += self.log_prior()

                # Sample x from the variational posterior
                x = q_mu + np.sqrt(q_sigma) * npr.randn(data.shape[0], self.D)

                # Compute log p(x | theta) = log \sum_z p(x, z | theta)
                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
                log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)
                elbo += hmm_normalizer(log_pi0, log_Ps, log_likes)

                # -log q(x)
                elbo -= np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
                elbo -= np.sum(-0.5 * (x - q_mu)**2 / q_sigma)

                assert np.isfinite(elbo)
        
        return elbo / n_samples

    def _fit_svi(self, datas, inputs, masks, tags, learning=True, optimizer="adam", print_intvl=1, **kwargs):
        """
        Fit with stochastic variational inference using a 
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        T = sum([data.shape[0] for data in datas])

        # Initialize the variational posterior parameters
        variational_params = [self.emissions.initialize_variational_params(data, input, mask, tag) 
                              for data, input, mask, tag in zip(datas, inputs, masks, tags)]

        def _objective(params, itr):
            if learning:
                self.params, variational_params = params
            else:
                variational_params = params

            obj = self.elbo(variational_params, datas, inputs, masks, tags)
            return -obj / T

        elbos = []
        def _print_progress(params, itr, g):
            elbos.append(-_objective(params, itr) * T)
            if itr % print_intvl == 0:
                print("Iteration {}.  ELBO: {:.1f}".format(itr, elbos[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam, adam_with_convergence_check=adam_with_convergence_check)
        initial_params = (self.params, variational_params) if learning else variational_params
        results = \
            optimizers[optimizer](grad(_objective), 
                initial_params,
                callback=_print_progress,
                **kwargs)

        if learning:
            self.params, variational_params = results
        else:
            variational_params = results

        # unpack outputs as necessary
        variational_params = variational_params[0] if len(variational_params) == 1 else variational_params
        return elbos, variational_params

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None, method="svi", initialize=True, **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs, masks, tags)

        return self._fitting_methods[method](datas, inputs, masks, tags, learning=True, **kwargs)

    @ensure_args_are_lists
    def approximate_posterior(self, datas, inputs=None, masks=None, tags=None, method="svi", **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        return self._fitting_methods[method](datas, inputs, masks, tags, learning=False, **kwargs)


class _LDS(_SwitchingLDS):
    """
    Switching linear dynamical system fit with 
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, D, M, dynamics, emissions):
        from ssm.init_state_distns import InitialStateDistribution
        from ssm.transitions import StationaryTransitions
        init_state_distn = InitialStateDistribution(1, D, M)
        transitions = StationaryTransitions(1, D, M)
        super(_LDS, self).__init__(N, 1, D, M, init_state_distn, transitions, dynamics, emissions)
    
    @ensure_slds_args_not_none    
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        return np.ones((variational_mean.shape[0], 1)), \
               np.ones((variational_mean.shape[0], 1, 1)), 

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def log_prior(self):
        return self.dynamics.log_prior() + self.emissions.log_prior()

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Log probability of LDS is not yet implemented.")
        return np.nan

    @ensure_elbo_args_are_lists
    def elbo(self, variational_params, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for data, input, mask, tag, (q_mu, q_sigma_inv) in \
            zip(datas, inputs, masks, tags, variational_params):

            q_sigma = np.exp(q_sigma_inv)
            for sample in range(n_samples):
                # log p(theta)
                elbo += self.log_prior()

                # Sample x from the variational posterior
                x = q_mu + np.sqrt(q_sigma) * npr.randn(data.shape[0], self.D)
                x_mask = np.ones_like(x, dtype=bool)
                # Compute log p(y, x | theta) 
                elbo += np.sum(self.dynamics.log_likelihoods(x, input, x_mask, tag))
                elbo += np.sum(self.emissions.log_likelihoods(data, input, mask, tag, x))
                
                # -log q(x)
                elbo -= np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
                elbo -= np.sum(-0.5 * (x - q_mu)**2 / q_sigma)

                assert np.isfinite(elbo)
        
        return elbo / n_samples

