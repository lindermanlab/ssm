import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none, interpolate_data


class _Observations(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError
    
    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass
        
    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample_x(self, z, xhist, input=None, tag=None):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags, 
               optimizer="adam", num_iters=10, **kwargs):
        """
        Max likelihood is not available in closed form. Default to SGD.
        """
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]
        
        # expected log joint
        def _expected_log_joint(expectations):
            elbo = 0
            for data, input, mask, tag, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self._log_likelihoods(data, input, mask, tag)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = \
            optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError


class GaussianObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(GaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)

    @property
    def params(self):
        return self.mus, self.inv_sigmas
    
    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.inv_sigmas = self.inv_sigmas[perm]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self.inv_sigmas = np.log(sigmas)
        
    def log_likelihoods(self, data, input, mask, tag):
        mus, sigmas = self.mus, np.exp(self.inv_sigmas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas) 
            * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None):
        D, mus, sigmas = self.D, self.mus, np.exp(self.inv_sigmas)
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _ in expectations])
        for k in range(self.K):
            self.mus[k] = np.average(x, axis=0, weights=weights[:,k])
            sqerr = (x - self.mus[k])**2
            self.inv_sigmas[k] = np.log(np.average(sqerr, weights=weights[:,k], axis=0))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class StudentsTObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(StudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)
        # Student's t distribution also has a degrees of freedom parameter
        self.inv_nus = np.log(4) * np.ones(K)

    @property
    def params(self):
        return self.mus, self.inv_sigmas, self.inv_nus
    
    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas, self.inv_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.inv_sigmas = self.inv_sigmas[perm]
        self.inv_nus = self.inv_nus[perm] 
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self.inv_sigmas = np.log(sigmas)
        self.inv_nus = np.log(4) * np.ones(self.K)
        
    def log_likelihoods(self, data, input, mask, tag):
        D, mus, sigmas, nus = self.D, self.mus, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        # mask = np.ones_like(data, dtype=bool) if mask is None else mask

        resid = data[:, None, :] - mus
        z = resid / sigmas
        return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=1)

    def sample_x(self, z, xhist, input=None, tag=None):
        D, mus, sigmas, nus = self.D, self.mus, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[z] + np.sqrt(sigmas[z] / tau) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class BernoulliObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(BernoulliObservations, self).__init__(K, D, M)
        self.logit_ps = npr.randn(K, D)
        
    @property
    def params(self):
        return (self.logit_ps,)
    
    @params.setter
    def params(self, value):
        self.logit_ps = value[0]
        
    def permute(self, perm):
        self.logit_ps = self.logit_ps[perm]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        ps = km.cluster_centers_
        assert np.all((ps > 0) & (ps < 1))
        self.logit_ps = np.log(ps / (1-ps))
        
    def log_likelihoods(self, data, input, mask, tag):
        assert data.dtype == int and data.min() >= 0 and data.max() <= 1
        ps = 1 / (1 + np.exp(self.logit_ps))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None):
        ps = 1 / (1 + np.exp(self.logit_ps))
        return npr.rand(self.D) < ps[z]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _ in expectations])
        for k in range(self.K):
            ps = np.average(x, axis=0, weights=weights[:,k])
            self.logit_ps[k] = np.log((ps + 1e-8) / (1 - ps + 1e-8))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        ps = 1 / (1 + np.exp(self.logit_ps))
        return expectations.dot(ps)


class PoissonObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(PoissonObservations, self).__init__(K, D, M)
        self.log_lambdas = npr.randn(K, D)
        
    @property
    def params(self):
        return (self.log_lambdas,)
    
    @params.setter
    def params(self, value):
        self.log_lambdas = value[0]
        
    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.log_lambdas = np.log(km.cluster_centers_)
        
    def log_likelihoods(self, data, input, mask, tag):
        assert data.dtype == int
        lambdas = np.exp(self.inv_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None):
        lambdas = np.exp(self.inv_lambdas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _ in expectations])
        for k in range(self.K):
            self.inv_lambdas = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-8)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.inv_lambdas))


class AutoRegressiveObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(AutoRegressiveObservations, self).__init__(K, D, M)
        
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
        return self.As, self.bs, self.Vs, self.inv_sigmas
        
    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas = value

    def permute(self, perm):
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self.inv_sigmas = self.inv_sigmas[perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
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

    def _compute_mus(self, data, input, mask, tag):
        assert np.all(mask), "ARHMM cannot handle missing data"

        As, bs, Vs = self.As, self.bs, self.Vs

        # linear function of preceding data, current input, and bias
        mus = np.matmul(As[None, ...], data[:-1, None, :, None])[:, :, :, 0]
        mus = mus + np.matmul(Vs[None, ...], input[1:, None, :, None])[:, :, :, 0]
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((1, self.K, self.D)), mus))
        return mus

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = np.exp(self.inv_sigmas)
        return -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas) 
            * mask[:, None, :], axis=2)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
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

    def sample_x(self, z, xhist, input=None, tag=None):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] == 0:
            mu_init = self.mu_init
            sigma_init = np.exp(self.inv_sigma_init)
            return mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            return As[z].dot(xhist[-1]) + bs[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool) 
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)


# Robust autoregressive models with Student's t noise
class RobustAutoRegressiveObservations(AutoRegressiveObservations):
    def __init__(self, K, D, M=0):
        super(RobustAutoRegressiveObservations, self).__init__(K, D, M)
        self.inv_nus = np.log(4) * np.ones(K)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus
        
    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus = value

    def permute(self, perm):
        super(RobustAutoRegressiveObservations, self).permute(perm)
        self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        D = self.D
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = np.exp(self.inv_sigmas)
        nus = np.exp(self.inv_nus)

        resid = data[:, None, :] - mus
        z = resid / sigmas
        return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=1)

    def sample_x(self, z, xhist, input=None, tag=None):
        D, As, bs, sigmas, nus = self.D, self.As, self.bs, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        if xhist.shape[0] == 0:
            mu_init = self.mu_init
            sigma_init = np.exp(self.inv_sigma_init)
            return mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            return As[z].dot(xhist[-1]) + bs[z] + np.sqrt(sigmas[z] / tau) * npr.randn(D)


class _RecurrentAutoRegressiveObservationsMixin(AutoRegressiveObservations):
    """
    A simple mixin to allow for smarter initialization.
    """
    def initialize(self, datas, inputs=None, masks=None, tags=None):
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


class RecurrentAutoRegressiveObservations(
    _RecurrentAutoRegressiveObservationsMixin, 
    AutoRegressiveObservations):
    pass


class RecurrentRobustAutoRegressiveObservations(
    _RecurrentAutoRegressiveObservationsMixin, 
    RobustAutoRegressiveObservations):
    pass


# class _HierarchicalAutoRegressiveHMMObservations(object):
#     def __init__(self, K, D, M=0, tags=(None,), lmbda=0.01):
#         super(_HierarchicalAutoRegressiveHMMObservations, self).__init__(K, D, M)
        
#         # Top-level AR parameters (global mean)
#         self.As = .95 * np.array([random_rotation(D) for _ in range(K)])
#         self.bs = npr.randn(K, D)
#         self.Vs = npr.randn(K, D, M)
#         self.inv_sigmas = -4 + npr.randn(K, D)
#         self.lmbda = lmbda

#         # Make AR models for each tag
#         self.tags = tags
#         self.ars = dict()
#         for tag in tags:
#             self.ars[tag] = _AutoRegressiveHMMObservations(K, D, M=M)
#             self.ars[tag].As = self.As + np.sqrt(lmbda) * npr.randn(self.As.shape)
#             self.ars[tag].bs = self.bs + np.sqrt(lmbda) * npr.randn(self.bs.shape)
#             self.ars[tag].Vs = self.Vs + np.sqrt(lmbda) * npr.randn(self.Vs.shape)
#             self.ars[tag].inv_sigmas = self.inv_sigmas + np.sqrt(lmbda) * npr.randn(self.inv_sigmas.shape)

#     @property
#     def params(self):
#         prms = (self.As, self.bs, self.Vs, self.inv_sigmas) 
#         for tag in self.tags:
#             prms += self.ars[tag].params

#         return super(_HierarchicalAutoRegressiveHMMObservations, self).params + prms
               
        
#     @params.setter
#     def params(self, value):
#         # Unpack parameters of the sub-models in reverse order
#         for tag in self.tags[::-1]:
#             ar = self.ars[tag]
#             ar.As, ar.bs, ar.Vs, ar.inv_sigmas = value[:-4]
#             value = value[:-4]

#         # Set global parameters
#         self.As, self.bs, self.Vs, self.inv_sigmas = value[:-4]

#         # Call super method
#         super(_HierarchicalAutoRegressiveHMMObservations, self.__class__).params.fset(self, value[:-4])

#     def permute(self, perm):
#         super(_HierarchicalAutoRegressiveHMMObservations, self).permute(perm)
#         self.As = self.As[perm]
#         self.bs = self.bs[perm]
#         self.Vs = self.Vs[perm]
#         self.inv_sigmas = self.inv_sigmas[perm]

#         for ar in self.ars:
#             ar.permute(perm)

#     @ensure_args_are_lists
#     def initialize(self, datas, inputs=None, masks=None, tags=None):
#         # Initialize with linear regressions
#         from sklearn.linear_model import LinearRegression
#         data = np.concatenate(datas) 
#         input = np.concatenate(inputs)
#         T = data.shape[0]

#         for k in range(self.K):
#             ts = npr.choice(T-1, replace=False, size=T//2)
#             x, y = np.column_stack((data[ts], input[ts])), data[ts+1]
#             lr = LinearRegression().fit(x, y)
#             self.As[k] = lr.coef_[:, :self.D]
#             self.Vs[k] = lr.coef_[:, self.D:]
#             self.bs[k] = lr.intercept_
            
#             resid = y - lr.predict(x)
#             sigmas = np.var(resid, axis=0)
#             self.inv_sigmas[k] = np.log(sigmas)

#         for ar in self.ars:
#             ar.As = self.As + np.sqrt(lmbda) * npr.randn(self.As.shape)
#             ar.bs = self.bs + np.sqrt(lmbda) * npr.randn(self.bs.shape)
#             ar.Vs = self.Vs + np.sqrt(lmbda) * npr.randn(self.Vs.shape)
#             ar.inv_sigmas = self.inv_sigmas + np.sqrt(lmbda) * npr.randn(self.inv_sigmas.shape)



#     def _log_prior(self):
#         lp = 0

#         # Gaussian likelihood on each AR param given global AR param
#         for ar in self.ars:
#             lp += -0.5 * np.sum(np.log(2 * np.pi * self.lmbda) + (ar.As - self.As)**2 / self.lmbda, axis=2)
#             lp += -0.5 * np.sum(np.log(2 * np.pi * self.lmbda) + (ar.bs - self.bs)**2 / self.lmbda, axis=2)
#             lp += -0.5 * np.sum(np.log(2 * np.pi * self.lmbda) + (ar.Vs - self.Vs)**2 / self.lmbda, axis=2)
#             lp += -0.5 * np.sum(np.log(2 * np.pi * self.lmbda) + (ar.inv_sigmas - self.inv_sigmas)**2 / self.lmbda, axis=2)

#         return lp

#     def _log_likelihoods(self, data, input, mask, tag):
#         return self.ars[tag]._log_likelihoods(data, input, mask, tag)

#     def _m_step_observations(self, expectations, datas, inputs, masks, tags, **kwargs):
#         warnings.warn("_m_step_observations for _HierarchicalAutoRegressiveHMMObservations "
#                       "does not include the global prior. We still need to implement this feature.")
        
#         for tag in tags:
#             if not tag in self.tags:
#                 raise Exception("Invalid tag: ".format(tag))

#         for tag in self.tags:
#             self.ars[tag]._m_step_observations(
#                 [e for e,t in zip(expectations, tags) if t == tag],
#                 [d for d,t in zip(datas, tags) if t == tag],
#                 [i for i,t in zip(inputs, tags) if t == tag],
#                 [m for m,t in zip(masks, tags) if t == tag],
#                 [t for t in tags if t == tag],
#                 **kwargs)

#     def sample_x(self, z, xhist, input=None, tag=None):
#         return self.ars[tag].sample_x(z, xhist, input=input, tag=tag)

#     @ensure_args_not_none
#     def smooth(self, data, input=None, mask=None, tag=None):
#         return self.ars[tag].smooth(data, input=input, mask=mask, tag=tag)


class _HierarchicalObservations(object):

    __base_class = None     ## This must be set in the inheriting class!

    def __init__(self, base_class, K, D, M=0, tags=(None,), lmbda=0.01):
        super(_HierarchicalObservations, self).__init__(K, D, M)

        # How similar should parent and child params be
        self.lmbda = lmbda

        # Top-level AR parameters (parent mean)
        self.parent = self.__base_class(K, D, M=M)

        # Make AR models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            ch = self.children[tag] = self.__base_class(K, D, M=M)
            ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

    @property
    def params(self):
        prms = self.parent.params
        for tag in self.tags:
            prms += self.children[tag].params
        return super(_HierarchicalObservations, self).params + prms

    @params.setter
    def params(self, value):
        # Unpack parameters of the sub-models in reverse order
        nprms = len(self.parent.params)
        for tag in self.tags[::-1]:
            ch = self.children[tag]
            ch.params = value[:-nprms]
            value = value[:-nprms]

        # Set global parameters
        self.parent.params = value[:-nprms]
        value = value[:-nprms]

        # Call super method
        super(_HierarchicalObservations, self.__class__).params.fset(self, value)

    def permute(self, perm):
        super(_HierarchicalObservations, self).permute(perm)
        
        self.parent.permute(perm)
        for ch in self.children:
            ch.permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        for ch in self.children:
            cprm = copy.deepcopy(self.parent.params)
            for prm in cprm:
                prm += np.sqrt(lmbda) + npr.randn(*prm.shape)
            ch.params = cprm

    def _log_prior(self):
        lp = 0

        # Gaussian likelihood on each AR param given global AR param
        for ch in self.children:
            for pprm, cprm in zip(self.parent.params, ch.params):
                lp += -0.5 * np.sum(np.log(2 * np.pi * self.lmbda) + (cprm - pprm)**2 / self.lmbda, axis=2)
        return lp

    def _log_likelihoods(self, data, input, mask, tag):
        return self.children[tag]._log_likelihoods(data, input, mask, tag)

    def _m_step_observations(self, expectations, datas, inputs, masks, tags, **kwargs):
        warnings.warn("_m_step_observations for _HierarchicalAutoRegressiveHMMObservations "
                      "does not include the global prior. We still need to implement this feature.")
        
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        for tag in self.tags:
            self.children[tag]._m_step_observations(
                [e for e,t in zip(expectations, tags) if t == tag],
                [d for d,t in zip(datas, tags) if t == tag],
                [i for i,t in zip(inputs, tags) if t == tag],
                [m for m,t in zip(masks, tags) if t == tag],
                [t for t in tags if t == tag],
                **kwargs)

    def sample_x(self, z, xhist, input=None, tag=None):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(data, input=input, mask=mask, tag=tag)


class _HierarchicalAutoRegressiveHMMObservations(_HierarchicalObservations):
    __base_class = AutoRegressiveObservations


