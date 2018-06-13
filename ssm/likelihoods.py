import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none


class _GaussianHMMObservations(object):
    def __init__(self, K, D, M=0):
        super(_GaussianHMMObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)

    @property
    def params(self):
        return super(_GaussianHMMObservations, self).params + (self.mus, self.inv_sigmas)
    
    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas = value[-2:]
        super(_GaussianHMMObservations, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_GaussianHMMObservations, self).permute(perm)
        self.mus = self.mus[perm]
        self.inv_sigmas = self.inv_sigmas[perm]
        
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

    def _m_step_observations(self, expectations, datas, inputs, masks, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _ in expectations])
        for k in range(self.K):
            self.mus[k] = np.average(x, axis=0, weights=weights[:,k])
            sqerr = (x - self.mus[k])**2
            self.inv_sigmas[k] = np.log(np.average(sqerr, weights=weights[:,k], axis=0))

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        E_z, _ = self.expected_states(data, input, mask)
        return E_z.dot(self.mus)


class _StudentsTHMMObservations(object):
    def __init__(self, K, D, M=0):
        super(_StudentsTHMMObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)
        # Student's t distribution also has a degrees of freedom parameter
        self.inv_nus = np.log(4) * np.ones(K)

    @property
    def params(self):
        return super(_StudentsTHMMObservations, self).params + \
            (self.mus, self.inv_sigmas, self.inv_nus)
    
    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas, self.inv_nus = value[-3:]
        super(_StudentsTHMMObservations, self.__class__).params.fset(self, value[:-3])

    def permute(self, perm):
        super(_StudentsTHMMObservations, self).permute(perm)
        self.mus = self.mus[perm]
        self.inv_sigmas = self.inv_sigmas[perm]
        self.inv_nus = self.inv_nus[perm] 
        
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
        self.inv_nus = np.log(4) * np.ones(self.K)
        
    def _log_likelihoods(self, data, input, mask):
        D, mus, sigmas, nus = self.D, self.mus, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask

        resid = data[:, None, :] - mus
        z = resid / sigmas
        return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=1)

    def _sample_x(self, z, xhist):
        D, mus, sigmas, nus = self.D, self.mus, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[z] + np.sqrt(sigmas[z] / tau) * npr.randn(D)

    def _m_step_observations(self, expectations, datas, inputs, masks, 
                             optimizer="adam", num_iters=10, **kwargs):
        """
        Max likelihood is not available in closed form. Default to SGD.
        """
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]
        
        # expected log joint
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, expectations):
                lls = self._log_likelihoods(data, input, mask)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.mus, self.inv_sigmas, self.inv_nus = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.mus, self.inv_sigmas, self.inv_nus = \
            optimizer(grad(_objective), 
                (self.mus, self.inv_sigmas, self.inv_nus), 
                num_iters=num_iters, **kwargs)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        E_z, _ = self.expected_states(data, input, mask)
        return E_z.dot(self.mus)


class _BernoulliHMMObservations(object):
    def __init__(self, K, D, M=0):
        super(_BernoulliHMMObservations, self).__init__(K, D, M)
        self.logit_ps = npr.randn(K, D)
        
    @property
    def params(self):
        return super(_BernoulliHMMObservations, self).params + (self.logit_ps,)
    
    @params.setter
    def params(self, value):
        self.logit_ps = value[-1:]
        super(_BernoulliHMMObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_BernoulliHMMObservations, self).permute(perm)
        self.logit_ps = self.logit_ps[perm]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        ps = km.cluster_centers_
        assert np.all((ps > 0) & (ps < 1))
        self.logit_ps = np.log(ps / (1-ps))
        
    def _log_likelihoods(self, data, input, mask):
        assert data.dtype == int and data.min() >= 0 and data.max() <= 1
        ps = 1 / (1 + np.exp(self.logit_ps))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)

    def _sample_x(self, z, xhist):
        ps = 1 / (1 + np.exp(self.logit_ps))
        return npr.rand(self.D) < ps[z]

    def _m_step_observations(self, expectations, datas, inputs, masks, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _ in expectations])
        for k in range(self.K):
            ps = np.average(x, axis=0, weights=weights[:,k])
            self.logit_ps[k] = np.log((ps + 1e-8) / (1 - ps + 1e-8))

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        E_z, _ = self.expected_states(data, input, mask)
        ps = 1 / (1 + np.exp(self.logit_ps))
        return E_z.dot(ps)


class _PoissonHMMObservations(object):
    def __init__(self, K, D, M=0):
        super(_PoissonHMMObservations, self).__init__(K, D, M)
        self.log_lambdas = npr.randn(K, D)
        
    @property
    def params(self):
        return super(_PoissonHMMObservations, self).params + (self.log_lambdas,)
    
    @params.setter
    def params(self, value):
        self.log_lambdas = value[-1:]
        super(_PoissonHMMObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_PoissonHMMObservations, self).permute(perm)
        self.log_lambdas = self.log_lambdas[perm]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
        
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.log_lambdas = np.log(km.cluster_centers_)
        
    def _log_likelihoods(self, data, input, mask):
        assert data.dtype == int
        lambdas = np.exp(self.inv_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def _sample_x(self, z, xhist):
        lambdas = np.exp(self.inv_lambdas)
        return npr.poisson(lambdas[z])

    def _m_step_observations(self, expectations, datas, inputs, masks, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _ in expectations])
        for k in range(self.K):
            self.inv_lambdas = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-8)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        E_z, _ = self.expected_states(data, input, mask)
        return E_z.dot(np.exp(self.inv_lambdas))


class _AutoRegressiveHMMObservations(object):
    def __init__(self, K, D, M=0):
        super(_AutoRegressiveHMMObservations, self).__init__(K, D, M)
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
        return super(_AutoRegressiveHMMObservations, self).params + \
               (self.As, self.bs, self.Vs, self.inv_sigmas)
        
    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas = value[-4:]
        super(_AutoRegressiveHMMObservations, self.__class__).params.fset(self, value[:-4])

    def permute(self, perm):
        super(_AutoRegressiveHMMObservations, self).permute(perm)
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self.inv_sigmas = self.inv_sigmas[perm]

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

    def _m_step_observations(self, expectations, datas, inputs, masks, **kwargs):
        from sklearn.linear_model import LinearRegression
        D, M = self.D, self.M

        for k in range(self.K):
            xs, ys, weights = [], [], []
            for (Ez, _), data, input in zip(expectations, datas, inputs):
                xs.append(np.hstack((data[:-1], input[:-1])))
                ys.append(data[1:])
                weights.append(Ez[1:,k])
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            weights = np.concatenate(weights)

            # Fit a weighted linear regression
            lr = LinearRegression()
            lr.fit(xs, ys, sample_weight=weights)
            self.As[k], self.Vs[k], self.bs[k] = lr.coef_[:,:D], lr.coef_[:,D:], lr.intercept_

            # Update the variances
            yhats = lr.predict(xs)
            sqerr = (ys - yhats)**2
            self.inv_sigmas[k] = np.log(np.average(sqerr, weights=weights, axis=0))

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


class _RecurrentAutoRegressiveHMMMixin(object):
    """
    A simple mixin to allow for smarter initialization.
    """
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None):
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


# Robust autoregressive models with Student's t noise
class _RobustAutoRegressiveHMMObservations(_AutoRegressiveHMMObservations):
    def __init__(self, K, D, M=0):
        super(_RobustAutoRegressiveHMMObservations, self).__init__(K, D, M)
        self.inv_nus = np.log(4) * np.ones(K)

    @property
    def params(self):
        return super(_RobustAutoRegressiveHMMObservations, self).params + \
               (self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus)
        
    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus = value[-5:]
        super(_RobustAutoRegressiveHMMObservations, self.__class__).params.fset(self, value[:-5])

    def permute(self, perm):
        super(_RobustAutoRegressiveHMMObservations, self).permute(perm)
        self.inv_nus = self.inv_nus[perm]

    def _log_likelihoods(self, data, input, mask):
        D = self.D
        mus = self._compute_mus(data, input, mask)
        sigmas = np.exp(self.inv_sigmas)
        nus = np.exp(self.inv_nus)

        resid = data[:, None, :] - mus
        z = resid / sigmas
        return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=1)

    def _m_step_observations(self, expectations, datas, inputs, masks, 
                             optimizer="adam", num_iters=10, **kwargs):
        """
        Max likelihood is not available in closed form. Default to SGD.
        """
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]
        
        # expected log joint
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, expectations):
                lls = self._log_likelihoods(data, input, mask)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus = \
            optimizer(grad(_objective), 
                (self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus), 
                num_iters=num_iters, **kwargs)

    def _sample_x(self, z, xhist):
        D, As, bs, sigmas, nus = self.D, self.As, self.bs, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        if xhist.shape[0] == 0:
            mu_init = self.mu_init
            sigma_init = np.exp(self.inv_sigma_init)
            return mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            return As[z].dot(xhist[-1]) + bs[z] + np.sqrt(sigmas[z] / tau) * npr.randn(D)

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


# Observations models for SLDS
class _GaussianSLDSObservations(object):
    def __init__(self, N, K, D, *args, single_subspace=True):
        super(_GaussianSLDSObservations, self).__init__(N, K, D, *args)

        # Initialize observation model
        self.single_subspace = single_subspace
        self.Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_GaussianSLDSObservations, self).params + \
               (self.Cs, self.ds, self.inv_etas)
        
    @params.setter
    def params(self, value):
        self.Cs, self.ds, self.inv_etas = value[-3:]
        super(_GaussianSLDSObservations, self.__class__).params.fset(self, value[:-3])

    def permute(self, perm):
        super(_GaussianSLDSObservations, self).permute(perm)

        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.ds = self.ds[perm]
            # self.Vs = self.Vs[perm]
            self.inv_eta = self.inv_etas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        data = np.concatenate(datas)
        pca = PCA(self.D)
        x = pca.fit_transform(data)
        resid = data - pca.inverse_transform(x)
        etas = np.var(resid, axis=0)

        self.Cs[:,...] = pca.components_.T
        self.ds[:,...] = pca.mean_
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

        # Initialize the dynamics parameters with the pca embedded data
        xs = np.split(x, [len(data) for data in datas])
        xmasks = [np.ones_like(x, dtype=bool) for x in xs]
        super(_GaussianSLDSObservations, self).initialize(xs, inputs, masks)
        
        print("Initializing with an ARHMM fit via ", num_em_iters, " iterations of EM.")
        super(_GaussianSLDSObservations, self)._fit_em(xs, inputs, xmasks, 
            num_em_iters=num_em_iters, step_size=1e-2, num_iters=10, verbose=False)
        print("Done.")

    def initialize_from_arhmm(self, arhmm, pca):
        for attr in ['As', 'bs', 'inv_sigmas', 'inv_nus',
                     'log_pi0', 'log_Ps', 
                     'Ws', 'Rs', 'r']:
            if hasattr(self, attr) and hasattr(arhmm, attr):
                setattr(self, attr, getattr(arhmm, attr).copy())

        
        self.Cs[:,...] = pca.components_.T
        self.ds[:,...] = pca.mean_
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def _initialize_variational_params(self, data, input, mask):
        # y = Cx + d + noise; C orthogonal.  xhat = (C^T C)^{-1} C^T (y-d)
        T = data.shape[0]
        C, d = self.Cs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T
        q_mu = (data-d).dot(C_pseudoinv)
        q_sigma_inv = -4 * np.ones((T, self.D))
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

    def smooth(self, variational_mean, input=None, mask=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Cs, ds = self.Cs, self.ds
        E_z, _ = self.expected_states(variational_mean, input, mask)
        mus = np.matmul(Cs[None, ...], variational_mean[:, None, :, None])[:, :, :, 0] + ds
        return mus[:,0,:] if self.single_subspace else np.sum(mus * E_z, axis=1)
        