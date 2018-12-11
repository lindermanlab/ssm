import copy
import warnings
from functools import partial
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from ssm.primitives import hmm_normalizer, hmm_expected_states, hmm_filter, viterbi
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists

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
        log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
        log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return viterbi(log_pi0, log_Ps, log_likes)

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
        for (Ez, Ezzp1, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
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
    def _fit_sgd(self, optimizer, datas, inputs, masks, tags, num_iters=1000, **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = self.log_probability(datas, inputs, masks, tags)
            return -obj / T
        
        # Set up the progress bar
        lls = [-_objective(self.params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            lls.append(-val * T)
            pbar.set_description("LP: {:.1f}".format(lls[-1]))
            pbar.update(1)
        
        return lls

    def _fit_stochastic_em(self, optimizer, datas, inputs, masks, tags, num_epochs=100, **kwargs):
        """
        Replace the M-step of EM with a stochastic gradient update using the ELBO computed
        on a minibatch of data. 
        """
        M = len(datas)
        T = sum([data.shape[0] for data in datas])
        
        # A helper to grab a minibatch of data
        perm = [np.random.permutation(M) for _ in range(num_epochs)]
        def _get_minibatch(itr):
            epoch = itr // M
            m = itr % M
            i = perm[epoch][m]
            return datas[i], inputs[i], masks[i], tags[i]

        # Define the objective (negative ELBO)
        def _objective(params, itr):
            # Grab a minibatch of data
            data, input, mask, tag = _get_minibatch(itr)
            Ti = data.shape[0]

            # E step: compute expected latent states with current parameters
            Ez, Ezzp1, _ = self.expected_states(data, input, mask, tag) 

            # M step: set the parameter and compute the (normalized) objective function
            self.params = params
            log_pi0 = self.init_state_distn.log_initial_state_distn(data, input, mask, tag)
            log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)

            # Compute the expected log probability 
            # (Scale by number of length of this minibatch.)
            obj = self.log_prior()
            obj += np.sum(Ez[0] * log_pi0) * M
            obj += np.sum(Ezzp1 * log_Ps) * (T - M) / (Ti - 1)
            obj += np.sum(Ez * log_likes) * T / Ti
            assert np.isfinite(obj)

            return -obj / T

        # Set up the progress bar
        lls = [-_objective(self.params, 0) * T]
        pbar = trange(num_epochs * M)
        pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            epoch = itr // M
            m = itr % M
            lls.append(-val * T)
            pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(epoch, m, lls[-1]))
            pbar.update(1)
        
        return lls

    def _fit_em(self, datas, inputs, masks, tags, num_em_iters=100, **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = [self.log_probability(datas, inputs, masks, tags)]

        pbar = trange(num_em_iters)
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
        _fitting_methods = \
            dict(sgd=partial(self._fit_sgd, "sgd"),
                 adam=partial(self._fit_sgd, "adam"),
                 em=self._fit_em,
                 stochastic_em=partial(self._fit_stochastic_em, "adam"),
                 stochastic_em_sgd=partial(self._fit_stochastic_em, "sgd"),
                 )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        return _fitting_methods[method](datas, inputs=inputs, masks=masks, tags=tags, **kwargs)


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
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # First initialize the observation model
        self.emissions.initialize(datas, inputs, masks, tags)

        # Get the initialized variational mean for the data
        xs = [self.emissions.invert(data, input, mask, tag)
              for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        xmasks = [np.ones_like(x, dtype=bool) for x in xs]

        # Now run a few iterations of EM on a ARHMM with the variational mean
        print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
        arhmm = _HMM(self.K, self.D, self.M, 
                     copy.deepcopy(self.init_state_distn),
                     copy.deepcopy(self.transitions),
                     copy.deepcopy(self.dynamics))

        arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags, 
                  method="em", num_em_iters=num_em_iters, num_iters=10)

        self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        self.transitions = copy.deepcopy(arhmm.transitions)
        self.dynamics = copy.deepcopy(arhmm.observations)
        
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

    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        N, K, D = self.N, self.K, self.D
        
        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            x = np.zeros((T+1, D))
            data = np.zeros((T+1, D))
            input = np.zeros((T+1, self.M)) if input is None else input
            xmask = np.ones((T+1, D), dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data, input, xmask, tag))
            z[0] = npr.choice(self.K, p=pi0)
            x[0] = self.dynamics.sample_x(z[0], x[:0], tag=tag, with_noise=with_noise)
        
        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            assert xhist.shape == (pad, D)
            assert yhist.shape == (pad, N)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            x = np.concatenate((xhist, np.zeros((T, D))))
            input = np.zeros((T+pad, self.M)) if input is None else input
            xmask = np.ones((T+pad, D), dtype=bool)
        
        # Sample z and x 
        for t in range(pad, T+pad):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x(z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Sample observations given latent states
        # TODO: sample in the loop above? 
        y = self.emissions.sample(z, x, input=input, tag=tag)
        return z[pad:], x[pad:], y[pad:]

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        x_mask = np.ones_like(variational_mean, dtype=bool)
        log_pi0 = self.init_state_distn.log_initial_state_distn(variational_mean, input, x_mask, tag)
        log_Ps = self.transitions.log_transition_matrices(variational_mean, input, x_mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, x_mask, tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        log_pi0 = self.init_state_distn.log_initial_state_distn(variational_mean, input, mask, tag)
        log_Ps = self.transitions.log_transition_matrices(variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return viterbi(log_pi0, log_Ps, log_likes)

    @ensure_slds_args_not_none
    def smooth(self, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(variational_mean, data, input, mask, tag)
        return self.emissions.smooth(Ez, variational_mean, data, input, tag)

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Cannot compute exact marginal log probability for the SLDS. "
                      "the ELBO instead.")
        return np.nan

    @ensure_variational_args_are_lists
    def elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            xs = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # log p(x, y | theta) = log \sum_z p(x, y, z | theta)            
            for x, data, input, mask, tag in zip(xs, datas, inputs, masks, tags):

                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
                log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)
                elbo += hmm_normalizer(log_pi0, log_Ps, log_likes)

            # -log q(x)
            elbo -= variational_posterior.log_density(xs)
            assert np.isfinite(elbo)
        
        return elbo / n_samples

    @ensure_variational_args_are_lists
    def _surrogate_elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None, 
        alpha=0.75, **kwargs):
        """
        Lower bound on the marginal likelihood p(y | gamma) 
        using variational posterior q(x; phi) where phi = variational_params
        and gamma = emission parameters.  As part of computing this objective, 
        we optimize q(z | x) and take a natural gradient step wrt theta, the 
        parameters of the dynamics model.

        Note that the surrogate ELBO is a lower bound on the ELBO above. 
           E_p(z | x, y)[log p(z, x, y)]
           = E_p(z | x, y)[log p(z, x, y) - log p(z | x, y) + log p(z | x, y)]
           = E_p(z | x, y)[log p(x, y) + log p(z | x, y)]
           = log p(x, y) + E_p(z | x, y)[log p(z | x, y)]
           = log p(x, y) -H[p(z | x, y)]
          <= log p(x, y) 
        with equality only when p(z | x, y) is atomic.  The gap equals the
        entropy of the posterior on z. 
        """
        # log p(theta)
        elbo = self.log_prior()

        # Sample x from the variational posterior
        xs = variational_posterior.sample()

        # Inner optimization: find the true posterior p(z | x, y; theta).
        # Then maximize the inner ELBO wrt theta,
        # 
        #    E_p(z | x, y; theta_fixed)[log p(z, x, y; theta).
        # 
        # This can be seen as a natural gradient step in theta
        # space.  Note: we do not want to compute gradients wrt x or the
        # emissions parameters backward throgh this optimization step, 
        # so we unbox them first.
        xs_unboxed = [getval(x) for x in xs]
        emission_params_boxed = self.emissions.params
        flat_emission_params_boxed, unflatten = flatten(emission_params_boxed)
        self.emissions.params = unflatten(getval(flat_emission_params_boxed))

        # E step: compute the true posterior p(z | x, y, theta_fixed) and 
        # the necessary expectations under this posterior.
        expectations = [self.expected_states(x, data, input, mask, tag) 
                        for x, data, input, mask, tag 
                        in zip(xs_unboxed, datas, inputs, masks, tags)]

        # M step: maximize expected log joint wrt parameters
        # Note: Only do a partial update toward the M step for this sample of xs
        x_masks = [np.ones_like(x, dtype=bool) for x in xs_unboxed]
        for distn in [self.init_state_distn, self.transitions, self.dynamics]:
            curr_prms = copy.deepcopy(distn.params)
            distn.m_step(expectations, xs_unboxed, inputs, x_masks, tags, **kwargs)
            distn.params = convex_combination(curr_prms, distn.params, alpha)

        # Box up the emission parameters again before computing the ELBO
        self.emissions.params = emission_params_boxed

        # Compute expected log likelihood E_q(z | x, y) [log p(z, x, y; theta)]
        for (Ez, Ezzp1, _), x, x_mask, data, mask, input, tag in \
            zip(expectations, xs, x_masks, datas, masks, inputs, tags):
            
            # Compute expected log likelihood (inner ELBO)
            log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
            log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
            log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
            log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

            elbo += np.sum(Ez[0] * log_pi0)
            elbo += np.sum(Ezzp1 * log_Ps)
            elbo += np.sum(Ez * log_likes)

        # -log q(x)
        elbo -= variational_posterior.log_density(xs)
        assert np.isfinite(elbo)
        
        return elbo

    def _fit_svi(self, variational_posterior, datas, inputs, masks, tags, 
                 learning=True, optimizer="adam", num_iters=100, **kwargs):
        """
        Fit with stochastic variational inference using a 
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        # Define the objective (negative ELBO)
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            if learning:
                self.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self.elbo(variational_posterior, datas, inputs, masks, tags)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.params, variational_posterior.params) 
        else:
            params = variational_posterior.params
        
        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("ELBO: {:.1f}".format(elbos[0]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # TODO: Check for convergence -- early stopping

            # Update progress bar
            pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
            pbar.update()
        
        # Save the final parameters
        if learning:
            self.params, variational_posterior.params = params
        else:
            variational_posterior.params = params
        
        return elbos

    def _fit_variational_em(self, variational_posterior, datas, inputs, masks, tags, 
                 learning=True, alpha=.75, optimizer="adam", num_iters=100, **kwargs):
        """
        Let gamma denote the emission parameters and theta denote the transition
        and initial discrete state parameters. This is a mix of EM and SVI:
            1. Sample x ~ q(x; phi)
            2. Compute L(x, theta') E_p(z | x, theta)[log p(x, z; theta')]
            3. Set theta = (1 - alpha) theta + alpha * argmax L(x, theta')
            4. Set gamma = gamma + eps * nabla log p(y | x; gamma)
            5. Set phi = phi + eps * dx/dphi * d/dx [L(x, theta) + log p(y | x; gamma) - log q(x; phi)]
        """
        # Optimize the standard ELBO when updating gamma (emissions params) 
        # and phi (variational params)
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            if learning:
                self.emissions.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self._surrogate_elbo(variational_posterior, datas, inputs, masks, tags, **kwargs)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.emissions.params, variational_posterior.params) 
        else:
            params = variational_posterior.params
        
        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("Surrogate ELBO: {:.1f}".format(elbos[0]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            # Update the emission and variational posterior parameters 
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # Update progress bar
            pbar.set_description("Surrogate ELBO: {:.1f}".format(elbos[-1]))
            pbar.update()
        
        # Save the final emission and variational parameters
        if learning:
            self.emissions.params, variational_posterior.params = params
        else:
            variational_posterior.params = params
        
        return elbos

    @ensure_variational_args_are_lists
    def fit(self, variational_posterior, datas, 
            inputs=None, masks=None, tags=None, method="svi", 
            initialize=True, **kwargs):

        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                vem=self._fit_variational_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs, masks, tags)

        return _fitting_methods[method](variational_posterior, datas, inputs, masks, tags, 
            learning=True, **kwargs)

    @ensure_variational_args_are_lists
    def approximate_posterior(self, variational_posterior, datas, inputs=None, masks=None, tags=None, 
                              method="svi", **kwargs):
        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                vem=self._fit_variational_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        return _fitting_methods[method](variational_posterior, datas, inputs, masks, tags, 
            learning=False, **kwargs)


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
               np.ones((variational_mean.shape[0], 1, 1)), \
               0

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def log_prior(self):
        return self.dynamics.log_prior() + self.emissions.log_prior()

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Log probability of LDS is not yet implemented.")
        return np.nan

    @ensure_variational_args_are_lists
    def elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            xs = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # Compute log p(y, x | theta) 
            for x, data, input, mask, tag in zip(xs, datas, inputs, masks, tags):
                x_mask = np.ones_like(x, dtype=bool)    
                elbo += np.sum(self.dynamics.log_likelihoods(x, input, x_mask, tag))
                elbo += np.sum(self.emissions.log_likelihoods(data, input, mask, tag, x))
                
            # -log q(x)
            elbo -= variational_posterior.log_density(xs)
            assert np.isfinite(elbo)
    
        return elbo / n_samples

