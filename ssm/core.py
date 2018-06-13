from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.primitives import hmm_normalizer, hmm_expected_states
from ssm.util import ensure_args_are_lists, ensure_args_not_none, ensure_elbo_args_are_lists

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
    def __init__(self, K, D, M):
        self.K, self.D, self.M = K, D, M
        self._fitting_methods = \
            dict(sgd=partial(self._fit_mle, "sgd"),
                 adam=partial(self._fit_mle, "adam"),
                 em=self._fit_em,
                 stochastic_em=self._fit_stochastic_em)

    @property
    def params(self):
        return ()
    
    @params.setter
    def params(self, value):
        # Note that calling the parent's setter is a pain. See this discussion:
        # https://stackoverflow.com/questions/10810369/python-super-and-setting-parent-class-property
        pass

    # Private methods: override in base classes    
    def _log_initial_state_distn(self, data, input, mask):
        """
        Compute the initial state distribution.

        :return K array of log initial state probabilities.
        """
        raise NotImplementedError
    
    def _log_transition_matrices(self, data, input, mask):
        """
        Compute the transition matrices for each time step.

        :return TxKxK array of log normalized transition matrices.
        """
        raise NotImplementedError
        
    def _log_likelihoods(self, data, input, mask):
        """
        Compute the log likelihood for each time step
        under each of the K latent states.

        Override this in base classes. 

        :return TxK matrix of log likelihoods.
        """
        raise NotImplementedError

    # External facing methods
    def _sample_x(self, z, xhist):
        raise NotImplementedError

    def sample(self, T, input=None):
        K, D = self.K, self.D
        data = np.zeros((T, D))
        input = np.zeros((T, self.M)) if input is None else input
        mask = np.ones((T, D), dtype=bool)
        
        # Initialize outputs
        z = np.zeros(T, dtype=int)
        
        pi0 = np.exp(self._log_initial_state_distn(data, input, mask))
        z[0] = npr.choice(self.K, p=pi0)
        data[0] = self._sample_x(z[0], data[:0])

        for t in range(1, T):
            Pt = np.exp(self._log_transition_matrices(
                data[t-1:t+1], 
                input[t-1:t+1],
                mask=mask[t-1:t+1]))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            data[t] = self._sample_x(z[t], data[:t])
        return z, data

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        """
        Optional: initialize parameters given data.
        """
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))

    @ensure_args_not_none
    def expected_states(self, data, input=None, mask=None):
        log_pi0 = self._log_initial_state_distn(data, input, mask)
        log_Ps = self._log_transition_matrices(data, input, mask)
        log_likes = self._log_likelihoods(data, input, mask)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @ensure_args_not_none
    def most_likely_states(self, data, input=None, mask=None):
        Ez, _ = self.expected_states(data, input, mask)
        return np.argmax(Ez, axis=1)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError
        
    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None):
        """
        Compute the log probability of the data under the current 
        model parameters.
        
        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, input, mask in zip(datas, inputs, masks):
            log_pi0 = self._log_initial_state_distn(data, input, mask)
            log_Ps = self._log_transition_matrices(data, input, mask)
            log_likes = self._log_likelihoods(data, input, mask)
            ll += hmm_normalizer(log_pi0, log_Ps, log_likes)
            assert np.isfinite(ll)
        return ll
    
    # Model fitting
    def _fit_mle(self, optimizer, datas, inputs, masks, print_intvl=10, **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data.shape[0] for data in datas])
        
        def _objective(params, itr):
            self.params = params
            obj = self.log_likelihood(datas, inputs, masks)
            return -obj / T

        lls = []
        def _print_progress(params, itr, g):
            lls.append(self.log_likelihood(datas, inputs, masks)._value)
            if itr % print_intvl == 0:
                print("Iteration {}.  LL: {}".format(itr, lls[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam)
        self.params = \
            optimizers[optimizer](grad(_objective), self.params,
                                  callback=_print_progress,
                                  **kwargs)

        return lls

    def _stochastic_m_step(self, expectations, datas, inputs, masks, optimizer="adam", **kwargs):
        """
        The default M step implementation does SGD on the expected log joint. 
        Base classes can override this with closed form updates if available.
        """
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]
        
        # expected log joint
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, expectations):

                log_pi0 = self._log_initial_state_distn(data, input, mask)
                elbo += np.sum(expected_states[0] * log_pi0)
                log_Ps = self._log_transition_matrices(data, input, mask)
                elbo += np.sum(expected_joints * log_Ps)
                log_likes = self._log_likelihoods(data, input, mask)
                elbo += np.sum(expected_states * log_likes)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, **kwargs)

    def _fit_stochastic_em(self, datas, inputs, masks, num_em_iters=100, optimizer="adam", **kwargs):
        """
        Fit the parameters with expectation maximization.  

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: stochastic gradient ascent on E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = []
        for itr in range(num_em_iters):
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask) 
                            for data, input, mask in zip(datas, inputs, masks)]

            # M step: maximize expected log joint wrt parameters
            self._stochastic_m_step(expectations, datas, inputs, masks, optimizer=optimizer, **kwargs)

            # Store progress
            lls.append(self.log_likelihood(datas, inputs, masks))
            print("Iteration {}.  LL: {}".format(itr, lls[-1]))

        return lls

    def _m_step_initial_state(self, expectations, datas, inputs, masks, **kwargs):
        raise NotImplementedError

    def _m_step_transitions(self, expectations, datas, inputs, masks, **kwargs):
        raise NotImplementedError

    def _m_step_observations(self, expectations, datas, inputs, masks, **kwargs):
        raise NotImplementedError

    def _fit_em(self, datas, inputs, masks, num_em_iters=100, verbose=True, **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = []
        for itr in range(num_em_iters):
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask) 
                            for data, input, mask in zip(datas, inputs, masks)]

            # M step: maximize expected log joint wrt parameters
            self._m_step_initial_state(expectations, datas, inputs, masks, **kwargs)
            self._m_step_transitions(expectations, datas, inputs, masks, **kwargs)
            self._m_step_observations(expectations, datas, inputs, masks, **kwargs)

            # Store progress
            lls.append(self.log_likelihood(datas, inputs, masks))

            if verbose:
                print("Iteration {}.  LL: {}".format(itr, lls[-1]))

        return lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, method="sgd", initialize=True, **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs=inputs, masks=masks)

        return self._fitting_methods[method](datas, inputs=inputs, masks=masks, **kwargs)


class _StationaryHMM(_HMM):
    """
    Standard Hidden Markov Model with fixed initial 
    distribution and transition matrix. 
    """
    def __init__(self, K, D, M=0):
        super(_StationaryHMM, self).__init__(K, D, M)
        
        # Initialize params
        self.log_pi0 = -np.log(K) * np.ones(K)
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return super(_StationaryHMM, self).params + (self.log_pi0, self.log_Ps)
    
    @params.setter
    def params(self, value):
        self.log_pi0, self.log_Ps = value[-2:]
        super(_StationaryHMM, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(_StationaryHMM, self).permute(perm)
        self.log_pi0 = self.log_pi0[perm]
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def init_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))
    
    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def _log_initial_state_distn(self, data, input, mask):
        return self.log_pi0 - logsumexp(self.log_pi0)
    
    def _log_transition_matrices(self, data, input, mask):
        T = data.shape[0]
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def _m_step_initial_state(self, expectations, datas, inputs, masks, **kwargs):
        pi0 = sum([Ez[0] for Ez, _ in expectations]) + 1e-8
        self.log_pi0 = np.log(pi0 / pi0.sum())

    def _m_step_transitions(self, expectations, datas, inputs, masks, **kwargs):
        expected_joints = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1 in expectations]) + 1e-8
        P = expected_joints / expected_joints.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(P)
    

class _InputDrivenHMM(_HMM):
    """
    Hidden Markov Model whose transition probabilities are 
    determined by a generalized linear model applied to the
    exogenous input. 
    """
    def __init__(self, K, D, M):
        super(_InputDrivenHMM, self).__init__(K, D, M)

        # Initial state distribution
        self.log_pi0 = -np.log(K) * np.ones(K)

        # Baseline transition probabilities
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(K, M)

    @property
    def params(self):
        return super(_InputDrivenHMM, self).params + \
               (self.log_pi0, self.log_Ps, self.Ws)
    
    @params.setter
    def params(self, value):
        self.log_pi0, self.log_Ps, self.Ws = value[-3], value[-2], value[-1]
        super(_InputDrivenHMM, self.__class__).params.fset(self, value[:-3])
        
    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(_InputDrivenHMM, self).permute(perm)
        self.log_pi0 = self.log_pi0[perm]
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    @property
    def init_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    def _log_initial_state_distn(self, data, input, mask):
        return self.log_pi0 - logsumexp(self.log_pi0)
    
    def _log_transition_matrices(self, data, input, mask):
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1)) 
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def _m_step_initial_state(self, expectations, datas, inputs, masks, **kwargs):
        pi0 = sum([Ez[0] for Ez, _ in expectations]) + 1e-8
        self.log_pi0 = np.log(pi0 / pi0.sum())

    def _m_step_transitions(self, expectations, datas, inputs, masks, optimizer="adam", num_iters=10, **kwargs):
        """
        M-step cannot be done in closed form for the transitions. Default to SGD.
        """
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]
        
        # expected log joint
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, expectations):
                log_Ps = self._log_transition_matrices(data, input, mask)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)


class _RecurrentHMM(_InputDrivenHMM):
    """
    Generalization of the input driven HMM in which 
    the observations serve as future inputs
    """
    def __init__(self, K, D, M):
        super(_RecurrentHMM, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Rs = npr.randn(K, D)

    @property
    def params(self):
        return super(_RecurrentHMM, self).params + (self.Rs,)
    
    @params.setter
    def params(self, value):
        self.Rs = value[-1]
        super(_RecurrentHMM, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(_RecurrentHMM, self).permute(perm)
        self.Rs = self.Rs[perm]

    def _log_transition_matrices(self, data, input, mask):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1)) 
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class _RecurrentOnlyHMM(_InputDrivenHMM):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M):
        super(_RecurrentOnlyHMM, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Rs = npr.randn(K, D)
        self.r = npr.randn(K)

        # Remove the transition matrix component
        del(self.log_Ps)

    @property
    def params(self):
        return (self.log_pi0, self.Ws, self.Rs, self.r)
    
    @params.setter
    def params(self, value):
        self.log_pi0, self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_pi0 = self.log_pi0[perm]
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def _log_transition_matrices(self, data, input, mask):
        T, D = data.shape
        log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


class _SwitchingLDSBase(object):
    """
    Switching linear dynamical system fit with 
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, K, D, *args, **kwargs):
        super(_SwitchingLDSBase, self).__init__(K, D, *args, **kwargs)
        self.N = N

        # Only allow fitting by SVI
        self._fitting_methods = dict(svi=self._fit_svi)

    def _emission_log_likelihoods(self, data, input, mask, x):
        raise NotImplementedError

    def _sample_y(self, z, x, input=None):
        raise NotImplementedError

    def sample(self, T, input=None):
        z, x = super(_SwitchingLDSBase, self).sample(T)
        y = self._sample_y(z, x, input=input)
        return z, x, y

    def expected_states(self, variational_mean, input=None, mask=None):
        input = np.zeros((variational_mean.shape[0], self.M)) if input is None else input
        mask = np.ones_like(variational_mean, dtype=bool) if mask is None else mask
        log_pi0 = self._log_initial_state_distn(variational_mean, input, mask)
        log_Ps = self._log_transition_matrices(variational_mean, input, mask)
        log_likes = self._log_likelihoods(variational_mean, input, mask)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    def most_likely_states(self, variational_mean, input=None, mask=None):
        Ez, _ = self.expected_states(variational_mean, input, mask)
        return np.argmax(Ez, axis=1)

    def log_likelihood(self, datas, inputs=None, masks=None):
        warn("Cannot compute exact marginal log likelihood for the SLDS. "
             "the ELBO instead.")
        return np.nan

    @ensure_elbo_args_are_lists
    def elbo(self, variational_params, datas, inputs=None, masks=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for data, input, mask, (q_mu, q_sigma_inv) in \
            zip(datas, inputs, masks, variational_params):

            q_sigma = np.exp(q_sigma_inv)
            for sample in range(n_samples):
                # Sample x from the variational posterior
                x = q_mu + np.sqrt(q_sigma) * npr.randn(data.shape[0], self.D)

                # Compute log p(x | theta) = log \sum_z p(x, z | theta)
                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                log_pi0 = self._log_initial_state_distn(x, input, x_mask)
                log_Ps = self._log_transition_matrices(x, input, x_mask)
                log_likes = self._log_likelihoods(x, input, x_mask)

                # Add the log likelihood of the data p(y | x, z, theta)
                log_likes = log_likes + self._emission_log_likelihoods(data, input, mask, x)
                elbo += hmm_normalizer(log_pi0, log_Ps, log_likes)

                # -log q(x)
                elbo -= np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
                elbo -= np.sum(-0.5 * (x - q_mu)**2 / q_sigma)

                assert np.isfinite(elbo)
        
        return elbo / n_samples

    def _initialize_variational_params(self, data, input, mask):
        T = data.shape[0]
        q_mu = np.zeros((T, self.D))
        q_sigma_inv = np.zeros((T, self.D))
        return q_mu, q_sigma_inv

    def _fit_svi(self, datas, inputs, masks, learning=True, optimizer="adam", print_intvl=1, **kwargs):
        """
        Fit with stochastic variational inference using a 
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        T = sum([data.shape[0] for data in datas])

        # Initialize the variational posterior parameters
        variational_params = [self._initialize_variational_params(data, input, mask) 
                              for data, input, mask in zip(datas, inputs, masks)]

        def _objective(params, itr):
            if learning:
                self.params, variational_params = params
            else:
                variational_params = params

            obj = self.elbo(variational_params, datas, inputs, masks)
            return -obj / T

        elbos = []
        def _print_progress(params, itr, g):
            elbos.append(-_objective(params, itr) * T)
            if itr % print_intvl == 0:
                print("Iteration {}.  ELBO: {}".format(itr, elbos[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam)
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
        variational_params = variational_params[0] if \
            len(variational_params) == 1 else variational_params
        return elbos, variational_params

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, method="svi", initialize=True, **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs, masks)

        return self._fitting_methods[method](datas, inputs, masks, learning=True, **kwargs)

    @ensure_args_are_lists
    def approximate_posterior(self, datas, inputs=None, masks=None, method="svi", **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        return self._fitting_methods[method](datas, inputs, masks, learning=False, **kwargs)


class _LDSBase(_SwitchingLDSBase):
    """
    Switching linear dynamical system fit with 
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, K, D, *args, **kwargs):
        assert K == 1
        super(_LDSBase, self).__init__(N, K, D, *args, **kwargs)
        self.N = N

        # Only allow fitting by SVI
        self._fitting_methods = dict(svi=self._fit_svi)

    def _emission_log_likelihoods(self, data, input, mask, x):
        raise NotImplementedError

    def _sample_y(self, z, x, input=None):
        raise NotImplementedError

    def sample(self, T, input=None):
        z, x = super(_LDSBase, self).sample(T)
        y = self._sample_y(z, x, input=input)
        return z, x, y

    def expected_states(self, variational_mean, input=None, mask=None):
        return np.ones((variational_mean.shape[0], 1)), \
               np.ones((variational_mean.shape[0], 1, 1)), 

    def most_likely_states(self, variational_mean, input=None, mask=None):
        raise NotImplementedError

    def log_likelihood(self, datas, inputs=None, masks=None):
        warn("Log likelihood of LDS is not yet implemented.")
        return np.nan

    @ensure_elbo_args_are_lists
    def elbo(self, variational_params, datas, inputs=None, masks=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for data, input, mask, (q_mu, q_sigma_inv) in \
            zip(datas, inputs, masks, variational_params):

            q_sigma = np.exp(q_sigma_inv)
            for sample in range(n_samples):
                # Sample x from the variational posterior
                x = q_mu + np.sqrt(q_sigma) * npr.randn(data.shape[0], self.D)
                x_mask = np.ones_like(x, dtype=bool)
                # Compute log p(x | theta) 
                elbo += np.sum(self._log_likelihoods(x, input, x_mask))
                # Compute the log likelihood of the data p(y | x, theta)
                elbo += np.sum(self._emission_log_likelihoods(data, input, mask, x))
                
                # -log q(x)
                elbo -= np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
                elbo -= np.sum(-0.5 * (x - q_mu)**2 / q_sigma)

                assert np.isfinite(elbo)
        
        return elbo / n_samples

