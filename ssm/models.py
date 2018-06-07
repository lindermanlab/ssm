from functools import partial

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.primitives import hmm_normalizer, hmm_expected_states
from ssm.util import random_rotation, ensure_args_are_lists, \
    ensure_args_not_none, ensure_elbo_args_are_lists

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
                em=self._fit_em)

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
        # import ipdb; ipdb.set_trace()
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

    def _m_step(self, expectations, datas, inputs, masks, optimizer="sgd", **kwargs):
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


    def _fit_em(self, datas, inputs, masks, num_em_iters=100, optimizer="sgd", **kwargs):
        """
        Fit the parameters with expectation maximization.

        Compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        Optimize the objective function E_{p(z | x)} [log p(x, z; theta)]
        with respect to theta. 
        """
        lls = []
        for itr in range(num_em_iters):
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask) 
                            for data, input, mask in zip(datas, inputs, masks)]

            # M step: maximize expected log joint wrt parameters
            self._m_step(expectations, datas, inputs, masks, optimizer=optimizer, **kwargs)

            # Store progress
            lls.append(self.log_likelihood(datas, inputs, masks))
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

    def _log_transition_matrices(self, data, input, mask):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1)) 
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class _GaussianObservations(object):
    def __init__(self, K, D, M=0):
        super(_GaussianObservations, self).__init__(K, D, M)

        self.D = D
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)

    @property
    def params(self):
        return super(_GaussianObservations, self).params + (self.mus, self.inv_sigmas)
    
    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas = value[-2:]
        super(_GaussianObservations, self.__class__).params.fset(self, value[:-2])

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self.inv_sigmas = np.log(sigmas)
        
    def _log_likelihoods(self, data, input, mask):
        mus, sigmas = self.mus, np.exp(self.inv_sigmas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas) 
            * mask[:, None, :], axis=2)

    def _sample_x(self, z, xhist):
        D, mus, sigmas = self.D, self.mus, np.exp(self.inv_sigmas)
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        E_z, _ = self.expected_states(data, input, mask)
        return E_z.dot(self.mus)


class GaussianHMM(_GaussianObservations, _StationaryHMM):
    pass


class InputDrivenGaussianHMM(_GaussianObservations, _InputDrivenHMM):
    pass


class RecurrentGaussianHMM(_GaussianObservations, _RecurrentHMM):
    pass


class _AutoRegressiveObservations(object):
    def __init__(self, K, D, M=0):
        super(_AutoRegressiveObservations, self).__init__(K, D, M)

        self.D = D
        # Distribution over initial point
        self.mu_init = np.zeros(D)
        self.inv_sigma_init = np.zeros(D)
        # AR parameters
        self.As = .95 * np.array([random_rotation(D) for _ in range(K)])
        self.bs = npr.randn(K, D)
        self.Vs = npr.randn(K, D, M)
        self.inv_sigmas = -4 + npr.randn(K, D)

    @property
    def params(self):
        return super(_AutoRegressiveObservations, self).params + \
               (self.As, self.bs, self.Vs, self.inv_sigmas)
        
    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas = value[-4:]
        super(_AutoRegressiveObservations, self.__class__).params.fset(self, value[:-4])

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]

        for k in range(self.K):
            ts = npr.choice(T-1, replace=False, size=T//2)
            x, y = np.column_stack((data[ts], input[ts])), data[ts+1]
            lr = LinearRegression().fit(x, y)
            self.As[k] = lr.coef_[:, :self.D]
            self.Vs[k] = lr.coef_[:, self.D:]
            self.bs[k] = lr.intercept_
            
            resid = y - lr.predict(x)
            sigmas = np.var(resid, axis=0)
            self.inv_sigmas[k] = np.log(sigmas)

    def _compute_mus(self, data, input, mask):
        assert np.all(mask), "ARHMM cannot handle missing data"

        As, bs, Vs = self.As, self.bs, self.Vs

        # linear function of preceding data, current input, and bias
        mus = np.matmul(As[None, ...], data[:-1, None, :, None])[:, :, :, 0]
        mus = mus + np.matmul(Vs[None, ...], input[1:, None, :, None])[:, :, :, 0]
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((1, self.K, self.D)), mus))
        return mus

    def _log_likelihoods(self, data, input, mask):
        mus = self._compute_mus(data, input, mask)
        sigmas = np.exp(self.inv_sigmas)
        return -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas) 
            * mask[:, None, :], axis=2)

    def _sample_x(self, z, xhist):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] == 0:
            mu_init = self.mu_init
            sigma_init = np.exp(self.inv_sigma_init)
            return mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            return As[z].dot(xhist[-1]) + bs[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        E_z, _ = self.expected_states(data, input, mask)
        mus = self._compute_mus(data, input, mask)
        return (E_z[:, :, None] * mus).sum(1)


class AutoRegressiveHMM(_AutoRegressiveObservations, _StationaryHMM):
    pass


class InputDriveAutoRegressiveHMM(_AutoRegressiveObservations, _InputDrivenHMM):
    pass


class RecurrentAutoRegressiveHMM(_AutoRegressiveObservations, _RecurrentHMM):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        """
        Smarter initialization by clustering the data
        """
        data = np.concatenate(datas) 
        input = np.concatenate(inputs)
        T = data.shape[0]
    
        from sklearn.cluster import KMeans
        km = KMeans(self.K)
        km.fit(data)
        z = km.labels_[:-1]

        # Cluster the data before initializing
        from sklearn.linear_model import LinearRegression
        
        for k in range(self.K):
            ts = np.where(z == k)[0]
            x, y = np.column_stack((data[ts], input[ts])), data[ts+1]
            lr = LinearRegression().fit(x, y)
            self.As[k] = lr.coef_[:, :self.D]
            self.Vs[k] = lr.coef_[:, self.D:]
            self.bs[k] = lr.intercept_
            
            resid = y - lr.predict(x)
            sigmas = np.var(resid, axis=0)
            self.inv_sigmas[k] = np.log(sigmas)


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

    def expected_states(self, variational_param, input=None, mask=None):
        x_mu, x_var = variational_param
        input = np.zeros((x_mu.shape[0], self.M)) if input is None else input
        mask = np.ones_like(x_mu, dtype=bool) if mask is None else mask
        log_pi0 = self._log_initial_state_distn(x_mu, input, mask)
        log_Ps = self._log_transition_matrices(x_mu, input, mask)
        log_likes = self._log_likelihoods(x_mu, input, mask)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    def most_likely_states(self, variational_params, input=None, mask=None):
        Ez, _ = self.expected_states(variational_params, input, mask)
        return np.argmax(Ez, axis=1)

    def log_likelihood(self, datas, inputs=None, masks=None):
        raise Exception("Cannot compute exact marginal log likelihood for the SLDS. "
                         "Compute the ELBO instead")

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

    def _fit_svi(self, datas, inputs, masks, optimizer="sgd", print_intvl=1, **kwargs):
        """
        Fit with stochastic variational inference using a 
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        T = sum([data.shape[0] for data in datas])

        # Initialize the variational posterior parameters
        variational_params = [self._initialize_variational_params(data, input, mask) 
                              for data, input, mask in zip(datas, inputs, masks)]

        def _objective(params, itr):
            self.params, variational_params = params
            obj = self.elbo(variational_params, datas, inputs, masks)
            return -obj / T

        elbos = []
        def _print_progress(params, itr, g):
            elbos.append(-_objective(params, itr) * T)
            if itr % print_intvl == 0:
                print("Iteration {}.  ELBO: {}".format(itr, elbos[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam)
        self.params, variational_params = \
            optimizers[optimizer](grad(_objective), 
                (self.params, variational_params),
                callback=_print_progress,
                **kwargs)

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

        return self._fitting_methods[method](datas, inputs, masks, **kwargs)



class _SwitchingLDS(_SwitchingLDSBase, AutoRegressiveHMM):
    pass


class _RecurrentSwitchingLDS(_SwitchingLDSBase, RecurrentAutoRegressiveHMM):
    pass


class _GaussianEmissions(object):
    def __init__(self, N, K, D, *args, single_subspace=True):
        super(_GaussianEmissions, self).__init__(N, K, D, *args)

        # Initialize observation model
        self.single_subspace = single_subspace
        self.Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_GaussianEmissions, self).params + \
               (self.Cs, self.ds, self.inv_etas)
        
    @params.setter
    def params(self, value):
        self.Cs, self.ds, self.inv_etas = value[-3:]
        super(_GaussianEmissions, self.__class__).params.fset(self, value[:-3])

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        data = np.concatenate(datas)
        pca = PCA(self.D)
        pca.fit(data)

        # Transform the data and compute residuals
        x = pca.transform(data)
        resid = data - pca.inverse_transform(x)
        etas = np.var(resid, axis=0)
        
        self.Cs[:,...] = pca.components_.T
        self.ds[:,...] = pca.mean_
        self.inv_etas[:,...] = np.log(etas)

        # Initialize the dynamics parameters with the pca embedded data
        xs = np.split(x, [len(data) for data in datas]) if isinstance(data, list) else x
        super(_GaussianEmissions, self).initialize(xs, inputs, masks)

    def _initialize_variational_params(self, data, input, mask):
        # y = Cx + d + noise; C orthogonal.  xhat = C^T (y-d)
        T = data.shape[0]
        C, d = self.Cs[0], self.ds[0]
        q_mu = (data-d).dot(C)
        q_sigma_inv = -2 * np.ones((T, self.D))
        return q_mu, q_sigma_inv
        
    def _emission_log_likelihoods(self, data, input, mask, x):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        Cs, ds = self.Cs, self.ds
        mus = np.matmul(Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + ds
        etas = np.exp(self.inv_etas)

        return -0.5 * np.sum(
            (np.log(2 * np.pi * etas) + (data[:, None, :] - mus)**2 / etas) 
            * mask[:, None, :], axis=2)

    def _sample_y(self, z, x, input=None):
        T = z.shape[0]
        Cs, ds = self.Cs, self.ds
        mus = np.matmul(Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + ds
        etas = np.exp(self.inv_etas)

        mu = mus[:,0,:] if self.single_subspace else mus[:, z, :] 
        eta = etas[0] if self.single_subspace else etas[z]
        y = mu + np.sqrt(eta) * npr.randn(T, self.N)
        return y

    @ensure_args_not_none
    def smooth(self, x, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        E_z, _ = self.expected_states(variational_param, input, mask)

        Cs, ds = self.Cs, self.ds
        mus = np.matmul(Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + ds
        return mus[:,0,:] if self.single_subspace else np.sum(mus * E_z, axis=1)
        

class GaussianSLDS(_GaussianEmissions, _SwitchingLDS):
    pass


class GaussianRecurrentSLDS(_GaussianEmissions, _RecurrentSwitchingLDS):
    pass

