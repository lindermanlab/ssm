import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import gammaln, digamma, logsumexp
from autograd.scipy.special import logsumexp

from ssm_star.util import random_rotation, ensure_args_are_lists, \
    logistic, logit, one_hot
from ssm_star.regression import fit_linear_regression, generalized_newton_studentst_dof
from ssm_star.preprocessing import interpolate_data
from ssm_star.cstats import robust_ar_statistics
from ssm_star.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm_star.stats as stats

class Observations(object):
    # K = number of discrete states
    # D = number of observed dimensions
    # M = exogenous input dimensions (the inputs modulate the probability of discrete state transitions via a multiclass logistic regression)

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
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        Ts = [data.shape[0] for data in datas]

        # Get initial discrete states
        if init_method.lower() == 'kmeans':
            # KMeans clustering
            from sklearn.cluster import KMeans
            km = KMeans(self.K)
            km.fit(np.vstack(datas))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

        elif init_method.lower() =='random':
            # Random assignment
            zs = [npr.choice(self.K, size=T) for T in Ts]

        else:
            raise Exception('Not an accepted initialization type: {}'.format(init_method))

        # Make a one-hot encoding of z and treat it as HMM expectations
        Ezs = [one_hot(z, self.K) for z in zs]
        expectations = [(Ez, None, None) for Ez in Ezs]

        # Set the variances all at once to use the setter
        self.m_step(expectations, datas, inputs, masks, tags)

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the observations, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, _, _) \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(data, input, mask, tag)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        raise NotImplementedError


class GaussianObservations(Observations):
    def __init__(self, K, D, M=0):
        super(GaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    def log_likelihoods(self, data, input, mask, tag):
        mus, Sigmas = self.mus, self.Sigmas
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        # stats.multivariate_normal_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), and (D,D)
        # arrays as inputs
        return np.column_stack([stats.multivariate_normal_logpdf(data, mu, Sigma)
                               for mu, Sigma in zip(mus, Sigmas)])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus = self.D, self.mus
        sqrt_Sigmas = self._sqrt_Sigmas if with_noise else np.zeros((self.K, self.D, self.D))
        return mus[z] + np.dot(sqrt_Sigmas[z], npr.randn(D))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K, D = self.K, self.D
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for (Ez, _, _), y in zip(expectations, datas):
            J += np.sum(Ez[:, :, None], axis=0)
            h += np.sum(Ez[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for (Ez, _, _), y in zip(expectations, datas):
            resid = y[:, None, :] - self.mus
            sqerr += np.sum(Ez[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(Ez, axis=0)
        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(self.D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)

class ExponentialObservations(Observations):
    def __init__(self, K, D, M=0):
        super(ExponentialObservations, self).__init__(K, D, M)
        self.log_lambdas = npr.randn(K, D)

    @property
    def params(self):
        return self.log_lambdas

    @params.setter
    def params(self, value):
        self.log_lambdas = value

    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.exponential_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        lambdas = np.exp(self.log_lambdas)
        return npr.exponential(1/lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.log_lambdas[k] = -np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(1/np.exp(self.log_lambdas))

class DiagonalGaussianObservations(Observations):
    def __init__(self, K, D, M=0):
        super(DiagonalGaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._log_sigmasq = -2 + npr.randn(K, D)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert np.all(value > 0) and value.shape == (self.K, self.D)
        self._log_sigmasq = np.log(value)

    @property
    def params(self):
        return self.mus, self._log_sigmasq

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    def log_likelihoods(self, data, input, mask, tag):
        mus, sigmas = self.mus, np.exp(self._log_sigmasq) + 1e-16
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.diagonal_gaussian_logpdf(data[:, None, :], mus, sigmas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus = self.D, self.mus
        sigmas = np.exp(self._log_sigmasq) if with_noise else np.zeros((self.K, self.D))
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.mus[k] = np.average(x, axis=0, weights=weights[:, k])
            sqerr = (x - self.mus[k])**2
            self._log_sigmasq[k] = np.log(np.average(sqerr, weights=weights[:, k], axis=0))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class StudentsTObservations(Observations):
    def __init__(self, K, D, M=0):
        super(StudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._log_sigmasq = -2 + npr.randn(K, D)
        # Student's t distribution also has a degrees of freedom parameter
        self._log_nus = np.log(4) * np.ones((K, D))

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]
        self._log_nus = self._log_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        D, mus, sigmas, nus = self.D, self.mus, self.sigmasq, self.nus
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.independent_studentst_logpdf(data[:, None, :], mus, sigmas, nus, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, sigmas, nus = self.D, self.mus, self.sigmasq, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sigma = sigmas[z] / tau if with_noise else 0
        return mus[z] + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, datas, inputs, masks, tags)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_mu_sigma(self, expectations, datas, inputs, masks, tags):
        K, D = self.K, self.D

        # Estimate the precisions w for each data point
        E_taus = []
        for y in datas:
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / self.sigmasq
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            J += np.sum(Ez[:, :, None] * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau * y[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D))
        weight = np.zeros((K, D))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            weight += np.sum(Ez[:, :, None], axis=0)
        self._log_sigmasq = np.log(sqerr / weight + 1e-16)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, sigma^2 / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.K, self.D

        # Compute the precisions w for each data point
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for y, (Ez, _, _) in zip(datas, expectations):
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> alpha/beta: (T, K, D)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / self.sigmasq

            E_taus += np.sum(Ez[:, :, None] * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))


class MultivariateStudentsTObservations(Observations):
    def __init__(self, K, D, M=0):
        super(MultivariateStudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)
        self._log_nus = np.log(4) * np.ones((K,))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]
        self._log_nus = self._log_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "MultivariateStudentsTObservations does not support missing data"
        D, mus, Sigmas, nus = self.D, self.mus, self.Sigmas, self.nus

        # stats.multivariate_studentst_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), (D,D), and (,)
        # arrays as inputs
        return np.column_stack([stats.multivariate_studentst_logpdf(data, mu, Sigma, nu)
                               for mu, Sigma, nu in zip(mus, Sigmas, nus)])

        # return stats.multivariate_studentst_logpdf(data[:, None, :], mus, Sigmas, nus)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, datas, inputs, masks, tags)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_mu_sigma(self, expectations, datas, inputs, masks, tags):
        K, D = self.K, self.D

        # Estimate the precisions w for each data point
        E_taus = []
        for y in datas:
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K,))
        h = np.zeros((K, D))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            J += np.sum(Ez * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J[:, None]

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            # sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            resid = y[:, None, :] - self.mus
            sqerr += np.einsum('tk,tk,tki,tkj->kij', Ez, E_tau, resid, resid)
            weight += np.sum(Ez, axis=0)

        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(D))

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, Sigma / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.K, self.D

        # Compute the precisions w for each data point
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for y, (Ez, _, _) in zip(datas, expectations):
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> alpha/beta: (T, K)
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)

            E_taus += np.sum(Ez * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, Sigmas, nus = self.D, self.mus, self.Sigmas, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sqrt_Sigma = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
        return mus[z] + np.dot(sqrt_Sigma, npr.randn(D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class BernoulliObservations(Observations):

    def __init__(self, K, D, M=0):
        super(BernoulliObservations, self).__init__(K, D, M)
        self.logit_ps = npr.randn(K, D)

    @property
    def params(self):
        return self.logit_ps

    @params.setter
    def params(self, value):
        self.logit_ps = value

    def permute(self, perm):
        self.logit_ps = self.logit_ps[perm]

    def log_likelihoods(self, data, input, mask, tag):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.bernoulli_logpdf(data[:, None, :], self.logit_ps, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        ps = 1 / (1 + np.exp(self.logit_ps))
        return npr.rand(self.D) < ps[z]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            ps = np.clip(np.average(x, axis=0, weights=weights[:,k]), 1e-3, 1-1e-3)
            self.logit_ps[k] = logit(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        ps = 1 / (1 + np.exp(self.logit_ps))
        return expectations.dot(ps)


class PoissonObservations(Observations):

    def __init__(self, K, D, M=0):
        super(PoissonObservations, self).__init__(K, D, M)
        self.log_lambdas = npr.randn(K, D)

    @property
    def params(self):
        return self.log_lambdas

    @params.setter
    def params(self, value):
        self.log_lambdas = value

    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        lambdas = np.exp(self.log_lambdas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.log_lambdas))


class CategoricalObservations(Observations):

    def __init__(self, K, D, M=0, C=2):
        """
        @param C:  number of classes in the categorical observations
        """
        super(CategoricalObservations, self).__init__(K, D, M)
        self.C = C
        self.logits = npr.randn(K, D, C)

    @property
    def params(self):
        return self.logits

    @params.setter
    def params(self, value):
        self.logits = value

    def permute(self, perm):
        self.logits = self.logits[perm]

    def log_likelihoods(self, data, input, mask, tag):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :], self.logits, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        ps = np.exp(self.logits - logsumexp(self.logits, axis=2, keepdims=True))
        return np.array([npr.choice(self.C, p=ps[z, d]) for d in range(self.D)])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            # compute weighted histogram of the class assignments
            xoh = one_hot(x, self.C)                                          # T x D x C
            ps = np.average(xoh, axis=0, weights=weights[:, k]) + 1e-3        # D x C
            ps /= np.sum(ps, axis=-1, keepdims=True)
            self.logits[k] = np.log(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

class InputDrivenObservations(Observations):

    def __init__(self, K, D, M=0, C=2, prior_mean = 0, prior_sigma=1000):
        """
        @param K: number of states
        @param D: dimensionality of output
        @param C: number of distinct classes for each dimension of output
        @param prior_sigma: parameter governing strength of prior. Prior on GLM weights is multivariate
        normal distribution with mean 'prior_mean' and diagonal covariance matrix (prior_sigma is on diagonal)
        """
        super(InputDrivenObservations, self).__init__(K, D, M)
        self.C = C
        self.M = M
        self.D = D
        self.K = K
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma
        # Parameters linking input to distribution over output classes
        self.Wk = npr.randn(K, C - 1, M)

    @property
    def params(self):
        return self.Wk

    @params.setter
    def params(self, value):
        self.Wk = value

    def permute(self, perm):
        self.Wk = self.Wk[perm]

    def log_prior(self):
        lp = 0
        for k in range(self.K):
            for c in range(self.C - 1):
                weights = self.Wk[k][c]
                lp += stats.multivariate_normal_logpdf(weights, mus=np.repeat(self.prior_mean, (self.M)),
                                                 Sigmas=((self.prior_sigma) ** 2) * np.identity(self.M))
        return lp

    # Calculate time dependent logits - output is matrix of size TxKxC
    # Input is size TxM
    def calculate_logits(self, input):
        """
        Return array of size TxKxC containing log(pr(yt=C|zt=k))
        :param input: input array of covariates of size TxM
        :return: array of size TxKxC containing log(pr(yt=c|zt=k, ut)) for all c in {1, ..., C} and k in {1, ..., K}
        """
        # Transpose array dimensions, so that array is now of shape ((C-1)xKx(M+1))
        Wk_tranpose = np.transpose(self.Wk, (1, 0, 2))
        # Stack column of zeros to transform array from size ((C-1)xKx(M+1)) to ((C)xKx(M+1)) and then transform shape back to (KxCx(M+1))
        Wk = np.transpose(np.vstack([Wk_tranpose, np.zeros((1, Wk_tranpose.shape[1], Wk_tranpose.shape[2]))]),
                          (1, 0, 2))
        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(Wk, input.T), (2, 0, 1)) #Note: this has an unexpected effect when both input (and thus Wk) are empty arrays and returns an array of zeros
        time_dependent_logits = time_dependent_logits - logsumexp(time_dependent_logits, axis=2, keepdims=True)
        return time_dependent_logits

    def log_likelihoods(self, data, input, mask, tag):
        time_dependent_logits = self.calculate_logits(input)
        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :], time_dependent_logits[:, :, None, :], mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        if input.ndim == 1 and input.shape == (self.M,): # if input is vector of size self.M (one time point), expand dims to be (1, M)
            input = np.expand_dims(input, axis=0)
        time_dependent_logits = self.calculate_logits(input)  # size TxKxC
        ps = np.exp(time_dependent_logits)
        T = time_dependent_logits.shape[0]
        if T == 1:
            sample = np.array([npr.choice(self.C, p=ps[t, z]) for t in range(T)])
        elif T > 1:
            sample = np.array([npr.choice(self.C, p=ps[t, z[t]]) for t in range(T)])
        return sample

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer = "bfgs", **kwargs):

        T = sum([data.shape[0] for data in datas]) #total number of datapoints

        def _multisoftplus(X):
            '''
            computes f(X) = log(1+sum(exp(X), axis =1)) and its first derivative
            :param X: array of size Tx(C-1)
            :return f(X) of size T and df of size (Tx(C-1))
            '''
            X_augmented = np.append(X, np.zeros((X.shape[0], 1)), 1) # append a column of zeros to X for rowmax calculation
            rowmax = np.max(X_augmented, axis = 1, keepdims=1) #get max along column for log-sum-exp trick, rowmax is size T
            # compute f:
            f = np.log(np.exp(-rowmax[:,0]) + np.sum(np.exp(X - rowmax), axis = 1)) + rowmax[:,0]
            # compute df
            df = np.exp(X - rowmax)/np.expand_dims((np.exp(-rowmax[:,0]) + np.sum(np.exp(X - rowmax), axis = 1)), axis = 1)
            return f, df

        def _objective(params, k):
            '''
            computes term in negative expected complete loglikelihood that depends on weights for state k
            :param params: vector of size (C-1)xM
            :return term in negative expected complete LL that depends on weights for state k; scalar value
            '''
            W = np.reshape(params, (self.C - 1, self.M))
            obj = 0
            for data, input, mask, tag, (expected_states, _, _) \
                    in zip(datas, inputs, masks, tags, expectations):
                xproj = input @ W.T  # projection of input onto weight matrix for particular state, size is Tx(C-1)
                f, _ = _multisoftplus(xproj)
                assert data.shape[1] == 1, "InputDrivenObservations written for D = 1!"
                data_one_hot = one_hot(data[:, 0], self.C)  # convert to one-hot representation of size TxC
                temp_obj = (-np.sum(data_one_hot[:,:-1]*xproj, axis = 1) + f)@expected_states[:,k]
                obj += temp_obj

            # add contribution of prior:
            if self.prior_sigma != 0:
                obj += 1/(2*self.prior_sigma**2)*np.sum(W**2)
            return obj / T

        def _gradient(params, k):
            '''
            Explicit calculation of gradient of _objective w.r.t weight matrix for state k, W_{k}
            :param params: vector of size (C-1)xM
            :param k: state whose parameters we are currently optimizing
            :return gradient of objective with respect to parameters; vector of size (C-1)xM
            '''
            W = np.reshape(params, (self.C-1, self.M))
            grad = np.zeros((self.C-1, self.M))
            for data, input, mask, tag, (expected_states, _, _) \
                    in zip(datas, inputs, masks, tags, expectations):
                xproj = input@W.T #projection of input onto weight matrix for particular state, size is Tx(C-1)
                _, df = _multisoftplus(xproj)
                assert data.shape[1] == 1, "InputDrivenObservations written for D = 1!"
                data_one_hot = one_hot(data[:, 0], self.C) #convert to one-hot representation of size TxC
                grad  += (df - data_one_hot[:,:-1]).T@(expected_states[:, [k]]*input) #gradient is shape (C-1,M)
            # Add contribution to gradient from prior:
            if self.prior_sigma != 0:
                grad += (1/(self.prior_sigma)**2)*W
            # Now flatten grad into a vector:
            grad = grad.flatten()
            return grad/T

        def _hess(params, k):
            '''
            Explicit calculation of hessian of _objective w.r.t weight matrix for state k, W_{k}
            :param params: vector of size (C-1)xM
            :param k: state whose parameters we are currently optimizing
            :return hessian of objective with respect to parameters; matrix of size ((C-1)xM) x ((C-1)xM)
            '''
            W = np.reshape(params, (self.C - 1, self.M))
            hess = np.zeros(((self.C - 1)*self.M, (self.C - 1)*self.M))
            for data, input, mask, tag, (expected_states, _, _) \
                    in zip(datas, inputs, masks, tags, expectations):
                xproj = input @ W.T  # projection of input onto weight matrix for particular state
                _, df = _multisoftplus(xproj)
                # center blocks:
                dftensor = np.expand_dims(df, axis = 2) # dims are now (T,  (C-1), 1)
                Xdf = np.expand_dims(input, axis = 1) * dftensor # multiply every input covariate term with every class derivative term for a given time step; dims are now (T, (C-1), M)
                # reshape Xdf to (T, (C-1)*M)
                Xdf = np.reshape(Xdf, (Xdf.shape[0], -1))
                # weight Xdf by posterior state probabilities
                pXdf = expected_states[:, [k]]*Xdf # output is size (T, (C-1)*M)
                # outer product with input vector, size (M, (C-1)*M)
                XXdf = input.T @ pXdf
                # center blocks of hessian:
                temp_hess = np.zeros(((self.C - 1) * self.M, (self.C - 1) * self.M))
                for c in range(1, self.C):
                    inds = range((c - 1)*self.M,c*self.M)
                    temp_hess[np.ix_(inds, inds)] = XXdf[:, inds]
                # off diagonal entries:
                hess += temp_hess - Xdf.T@pXdf
            # add contribution of prior to hessian
            if self.prior_sigma != 0:
                hess += (1 / (self.prior_sigma) ** 2)
            return hess/T

        from scipy.optimize import minimize
        # Optimize weights for each state separately:
        for k in range(self.K):
            def _objective_k(params):
                return _objective(params, k)
            def _gradient_k(params):
                return _gradient(params, k)
            def _hess_k(params):
                return _hess(params, k)
            sol = minimize(_objective_k, self.params[k].reshape(((self.C-1) * self.M)), hess=_hess_k, jac=_gradient_k, method="trust-ncg")
            self.params[k] = np.reshape(sol.x, (self.C-1, self.M))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


class _AutoRegressiveObservationsBase(Observations):
    """
    Base class for autoregressive observations of the form,

    E[x_t | x_{t-1}, z_t=k, u_t]
        = \sum_{l=1}^{L} A_k^{(l)} x_{t-l} + b_k + V_k u_t.

    where L is the number of lags and u_t is the input.
    """
    def __init__(self, K, D, M=0, lags=1):
        super(_AutoRegressiveObservationsBase, self).__init__(K, D, M)

        # Distribution over initial point
        self.mu_init = np.zeros((K, D))

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.bs = npr.randn(K, D)
        self.Vs = npr.randn(K, D, M)

        # Inheriting classes may treat _As differently
        self._As = None

    @property
    def As(self):
        return self._As

    @As.setter
    def As(self, value):
        self._As = value

    @property
    def params(self):
        return self.As, self.bs, self.Vs

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs = value

    def permute(self, perm):
        self.mu_init = self.mu_init[perm]
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]

    def _compute_mus(self, data, input, mask, tag):
        # assert np.all(mask), "ARHMM cannot handle missing data"
        K, M = self.K, self.M
        T, D = data.shape
        As, bs, Vs, mu0s = self.As, self.bs, self.Vs, self.mu_init

        # Instantaneous inputs
        mus = np.empty((K, T, D))
        mus = []
        for k, (A, b, V, mu0) in enumerate(zip(As, bs, Vs, mu0s)):
            # Initial condition
            mus_k_init = mu0 * np.ones((self.lags, D))

            # Subsequent means are determined by the AR process
            mus_k_ar = np.dot(input[self.lags:, :M], V.T)
            for l in range(self.lags):
                Al = A[:, l*D:(l + 1)*D]
                mus_k_ar = mus_k_ar + np.dot(data[self.lags-l-1:-l-1], Al.T)
            mus_k_ar = mus_k_ar + b

            # Append concatenated mean
            mus.append(np.vstack((mus_k_init, mus_k_ar)))

        return np.array(mus)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool)
        mus = np.swapaxes(self._compute_mus(data, input, mask, tag), 0, 1)
        return (expectations[:, :, None] * mus).sum(1)


class AutoRegressiveObservations(_AutoRegressiveObservationsBase):
    """
    AutoRegressive observation model with Gaussian noise.

        (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)

    where S_k is a positive definite covariance matrix.

    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, K, D, M=0, lags=1,
                 l2_penalty_A=1e-8,
                 l2_penalty_b=1e-8,
                 l2_penalty_V=1e-8,
                 nu0=1e-4, Psi0=1e-4):
        super(AutoRegressiveObservations, self).\
            __init__(K, D, M, lags=lags)

        # Initialize the dynamics and the noise covariances
        self._As = .80 * np.array([
                np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))])
            for _ in range(K)])

        self._sqrt_Sigmas_init = np.tile(np.eye(D)[None, ...], (K, 1, 1))
        self._sqrt_Sigmas = npr.randn(K, D, D)

        # Set natural parameters of Gaussian prior on (A, V, b) weight matrix
        J0_diag = np.concatenate((l2_penalty_A * np.ones(D * lags),
                                  l2_penalty_V * np.ones(M),
                                  l2_penalty_b * np.ones(1)))
        self.J0 = np.tile(np.diag(J0_diag)[None, :, :], (K, 1, 1))

        h0 = np.concatenate((l2_penalty_A * np.eye(D),
                             np.zeros((D * (lags - 1), D)),
                             np.zeros((M + 1, D))))
        self.h0 = np.tile(h0[None, :, :], (K, 1, 1))

        # Set natural parameters of inverse Wishart prior on Sigma
        self.nu0 = nu0
        self.Psi0 = Psi0 * np.eye(D) if np.isscalar(Psi0) else Psi0

        self.l2_penalty_A = l2_penalty_A
        self.l2_penalty_b = l2_penalty_b
        self.l2_penalty_V = l2_penalty_V

    @property
    def A(self):
        return self.As[0]

    @A.setter
    def A(self, value):
        assert value.shape == self.As[0].shape
        self.As[0] = value

    @property
    def b(self):
        return self.bs[0]

    @b.setter
    def b(self, value):
        assert value.shape == self.bs[0].shape
        self.bs[0] = value

    @property
    def Sigmas_init(self):
        return np.matmul(self._sqrt_Sigmas_init, np.swapaxes(self._sqrt_Sigmas_init, -1, -2))

    @Sigmas_init.setter
    def Sigmas_init(self, value):
        assert value.shape == (self.K, self.D, self.D)
        self._sqrt_Sigmas_init = np.linalg.cholesky(value + 1e-8 * np.eye(self.D))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (self.K, self.D, self.D)
        self._sqrt_Sigmas = np.linalg.cholesky(value + 1e-8 * np.eye(self.D))

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self._sqrt_Sigmas,)

    @params.setter
    def params(self, value):
        self._sqrt_Sigmas = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    def log_likelihoods(self, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        L = self.lags
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        # stats.multivariate_studentst_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), (D,D), and (,)
        # arrays as inputs
        ll_init = np.column_stack([stats.multivariate_normal_logpdf(data[:L], mu[:L], Sigma)
                               for mu, Sigma in zip(mus, self.Sigmas_init)])

        ll_ar = np.column_stack([stats.multivariate_normal_logpdf(data[L:], mu[L:], Sigma)
                               for mu, Sigma in zip(mus, self.Sigmas)])
        return np.row_stack((ll_init, ll_ar))

    def _get_sufficient_statistics(self, expectations, datas, inputs):
        K, D, M, lags = self.K, self.D, self.M, self.lags
        D_in = D * lags + M + 1

        # Initialize the outputs
        ExuxuTs = np.zeros((K, D_in, D_in))
        ExuyTs = np.zeros((K, D_in, D))
        EyyTs = np.zeros((K, D, D))
        Ens = np.zeros(K)

        # Iterate over data arrays and discrete states
        for (Ez, _, _), data, input in zip(expectations, datas, inputs):
            u = input[lags:]
            y = data[lags:]
            for k in range(K):
                w = Ez[lags:, k]

                # ExuxuTs[k]
                for l1 in range(lags):
                    x1 = data[lags-l1-1:-l1-1]
                    # Cross terms between lagged data and other lags
                    for l2 in range(l1, lags):
                        x2 = data[lags - l2 - 1:-l2 - 1]
                        ExuxuTs[k, l1*D:(l1+1)*D, l2*D:(l2+1)*D] += np.einsum('t,ti,tj->ij', w, x1, x2)

                    # Cross terms between lagged data and inputs and bias
                    ExuxuTs[k, l1*D:(l1+1)*D, D*lags:D*lags+M] += np.einsum('t,ti,tj->ij', w, x1, u)
                    ExuxuTs[k, l1*D:(l1+1)*D, -1] += np.einsum('t,ti->i', w, x1)

                ExuxuTs[k, D*lags:D*lags+M, D*lags:D*lags+M] += np.einsum('t,ti,tj->ij', w, u, u)
                ExuxuTs[k, D*lags:D*lags+M, -1] += np.einsum('t,ti->i', w, u)
                ExuxuTs[k, -1, -1] += np.sum(w)

                # ExuyTs[k]
                for l1 in range(lags):
                    x1 = data[lags - l1 - 1:-l1 - 1]
                    ExuyTs[k, l1*D:(l1+1)*D, :] += np.einsum('t,ti,tj->ij', w, x1, y)
                ExuyTs[k, D*lags:D*lags+M, :] += np.einsum('t,ti,tj->ij', w, u, y)
                ExuyTs[k, -1, :] += np.einsum('t,ti->i', w, y)

                # EyyTs[k] and Ens[k]
                EyyTs[k] += np.einsum('t,ti,tj->ij',w,y,y)
                Ens[k] += np.sum(w)

        # Symmetrize the expectations
        for k in range(K):
            for l1 in range(lags):
                for l2 in range(l1, lags):
                    ExuxuTs[k, l2*D:(l2+1)*D, l1*D:(l1+1)* D] = ExuxuTs[k, l1*D:(l1+1)*D, l2*D:(l2+1)*D].T
                ExuxuTs[k, D*lags:D*lags+M, l1*D:(l1+1)*D] = ExuxuTs[k, l1*D:(l1+1)*D, D*lags:D*lags+M].T
                ExuxuTs[k, -1, l1*D:(l1+1)*D] = ExuxuTs[k, l1*D:(l1+1)*D, -1].T
            ExuxuTs[k, -1, D*lags:D*lags+M] = ExuxuTs[k, D*lags:D*lags+M, -1].T

        return ExuxuTs, ExuyTs, EyyTs, Ens

    def _extend_given_sufficient_statistics(self, expectations, continuous_expectations, inputs):
        # Extend continuous_expectations with given inputs and discrete weights
        assert self.lags == 1, "_extend_given_sufficient_statistics assumes lags == 1."
        K, D, M, lags = self.K, self.D, self.M, self.lags
        D_in = D * lags + M + 1

        # Initialize the outputs
        ExuxuTs = np.zeros((K, D_in, D_in))
        ExuyTs = np.zeros((K, D_in, D))
        EyyTs = np.zeros((K, D, D))
        Ens = np.zeros(K)

        for (Ez, _, _), (_, Ex, smoothed_sigmas, Exxn), u in \
                zip(expectations, continuous_expectations, inputs):
            ExxT = smoothed_sigmas + np.einsum('ti,tj->tij', Ex, Ex)
            u = u[lags:]

            for k in range(K):
                w = Ez[lags:, k]

                ExuxuTs[k, :D, :D] += np.einsum('t,tij->ij', w, ExxT[:-1])
                ExuxuTs[k, :D, D:D + M] += np.einsum('t,ti,tj->ij', w, Ex[:-1], u)
                ExuxuTs[k, :D, -1] += np.einsum('t,ti->i', w, Ex[:-1])
                ExuxuTs[k, D:D + M, D:D + M] += np.einsum('t,ti,tj->ij', w, u, u)
                ExuxuTs[k, D:D + M, -1] += np.einsum('t,ti->i', w, u)
                ExuxuTs[k, -1, -1] += np.sum(w)

                ExuyTs[k, :D, :] += np.einsum('t,tij->ij', w, Exxn)
                ExuyTs[k, D:D + M, :] += np.einsum('t,ti,tj->ij', w, u, Ex[1:])
                ExuyTs[k, -1, :] += np.einsum('t,ti->i', w, Ex[1:])

                EyyTs[k] += np.einsum('t,tij->ij', w, ExxT[1:])
                Ens[k] += np.sum(w)

        # Symmetrize the expectations
        for k in range(K):
            ExuxuTs[k, D:D + M, :D] = ExuxuTs[k, :D, D:D + M].T
            ExuxuTs[k, -1, :D] = ExuxuTs[k, :D, -1].T
            ExuxuTs[k, -1, D:D + M] = ExuxuTs[k, D:D + M, -1].T

        return ExuxuTs, ExuyTs, EyyTs, Ens

    def m_step(self, expectations, datas, inputs, masks, tags,
               continuous_expectations=None, **kwargs):
        """Compute M-step for Gaussian Auto Regressive Observations.

        If `continuous_expectations` is not None, this function will
        compute an exact M-step using the expected sufficient statistics for the
        continuous states. In this case, we ignore the prior provided by (J0, h0),
        because the calculation is exact. `continuous_expectations` should be a tuple of
        (Ex, Ey, ExxT, ExyT, EyyT).

        If `continuous_expectations` is None, we use `datas` and `expectations,
        and (optionally) the prior given by (J0, h0). In this case, we estimate the sufficient
        statistics using `datas,` which is typically a single sample of the continuous
        states from the posterior distribution.
        """
        K, D, M, lags = self.K, self.D, self.M, self.lags
        # Collect sufficient statistics
        if continuous_expectations is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(expectations, datas, inputs)
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = \
                self._extend_given_sufficient_statistics(expectations, continuous_expectations, inputs)

        # Solve the linear regressions
        As = np.zeros((K, D, D * lags))
        Vs = np.zeros((K, D, M))
        bs = np.zeros((K, D))
        Sigmas = np.zeros((K, D, D))
        for k in range(K):
            Wk = np.linalg.solve(ExuxuTs[k] + self.J0[k], ExuyTs[k] + self.h0[k]).T
            As[k] = Wk[:, :D * lags]
            Vs[k] = Wk[:, D * lags:-1]
            bs[k] = Wk[:, -1]

            # Solve for the MAP estimate of the covariance
            EWxyT =  Wk @ ExuyTs[k]
            sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs[k] @ Wk.T
            nu = self.nu0 + Ens[k]
            Sigmas[k] = (sqerr + self.Psi0) / (nu + D + 1)

        # If any states are unused, set their parameters to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                As[k] = As[i] + 0.01 * npr.randn(*As[i].shape)
                Vs[k] = Vs[i] + 0.01 * npr.randn(*Vs[i].shape)
                bs[k] = bs[i] + 0.01 * npr.randn(*bs[i].shape)
                Sigmas[k] = Sigmas[i]

        # Update parameters via their setter
        self.As = As
        self.Vs = Vs
        self.bs = bs
        self.Sigmas = Sigmas

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, Vs = self.D, self.As, self.bs, self.Vs

        if xhist.shape[0] < self.lags:
            # Sample from the initial distribution
            S = np.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + np.dot(S, npr.randn(D))
        else:
            # Sample from the autoregressive distribution
            mu = Vs[z].dot(input[:self.M]) + bs[z]
            for l in range(self.lags):
                Al = As[z][:,l*D:(l+1)*D]
                mu += Al.dot(xhist[-l-1])

            S = np.linalg.cholesky(self.Sigmas[z]) if with_noise else 0
            return mu + np.dot(S, npr.randn(D))

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute negative Hessian of autoregressive obsevations with missing data."
        assert self.lags == 1, "Does not compute negative Hessian of autoregressive observations with lags > 1"

        # initial distribution contributes a Gaussian term to first diagonal block
        J_ini = np.sum(Ez[0, :, None, None] * np.linalg.inv(self.Sigmas_init), axis=0)

        # first part is transition dynamics - goes to all terms except final one
        # E_q(z) x_{t} A_{z_t+1}.T Sigma_{z_t+1}^{-1} A_{z_t+1} x_{t}
        inv_Sigmas = np.linalg.inv(self.Sigmas)
        dynamics_terms = np.array([A.T@inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)]) # A^T Qinv A terms
        J_dyn_11 = np.sum(Ez[1:,:,None,None] * dynamics_terms[None,:], axis=1)

        # second part of diagonal blocks are inverse covariance matrices - goes to all but first time bin
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} x_{t+1}
        J_dyn_22 = np.sum(Ez[1:,:,None,None] * inv_Sigmas[None,:], axis=1)

        # lower diagonal blocks are (T-1,D,D):
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} A_{z_t+1} x_t
        off_diag_terms = np.array([inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)])
        J_dyn_21 = -1 * np.sum(Ez[1:,:,None,None] * off_diag_terms[None,:], axis=1)

        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22


class AutoRegressiveObservationsNoInput(AutoRegressiveObservations):
    """
    AutoRegressive observation model without the inputs.
    """
    def __init__(self, K, D, M=0, lags=1,
                 l2_penalty_A=1e-8,
                 l2_penalty_b=1e-8):

        super(AutoRegressiveObservationsNoInput, self).\
            __init__(K, D, M=0, lags=lags,
                     l2_penalty_A=l2_penalty_A,
                     l2_penalty_b=l2_penalty_b)


class AutoRegressiveDiagonalNoiseObservations(AutoRegressiveObservations):
    """
    AutoRegressive observation model with diagonal Gaussian noise.

        (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)

    where

        S_k = diag([sigma_{k,1}, ..., sigma_{k, D}])

    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, K, D, M=0, lags=1,
                 l2_penalty_A=1e-8,
                 l2_penalty_b=1e-8,
                 l2_penalty_V=1e-8):

        super(AutoRegressiveDiagonalNoiseObservations, self).\
            __init__(K, D, M, lags=lags,
                     l2_penalty_A=l2_penalty_A,
                     l2_penalty_b=l2_penalty_b,
                     l2_penalty_V=l2_penalty_V)

        # Initialize the dynamics and the noise covariances
        self._As = .80 * np.array([
                np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))])
            for _ in range(K)])

        # Get rid of the square root parameterization and replace with log diagonal
        del self._sqrt_Sigmas_init
        del self._sqrt_Sigmas
        self._log_sigmasq_init = np.zeros((K, D))
        self._log_sigmasq = np.zeros((K, D))

    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @sigmasq_init.setter
    def sigmasq_init(self, value):
        assert value.shape == (self.K, self.D)
        assert np.all(value > 0)
        self._log_sigmasq_init = np.log(value)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert value.shape == (self.K, self.D)
        assert np.all(value > 0)
        self._log_sigmasq = np.log(value)

    @property
    def Sigmas_init(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq_init])

    @property
    def Sigmas(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq])

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (self.K, self.D, self.D)
        sigmasq = np.array([np.diag(S) for S in value])
        assert np.all(sigmasq > 0)
        self._log_sigmasq = np.log(sigmasq)

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self._log_sigmasq,)

    @params.setter
    def params(self, value):
        self._log_sigmasq = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."

        L = self.lags
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        # stats.multivariate_studentst_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), (D,D), and (,)
        # arrays as inputs
        ll_init = np.column_stack([stats.diagonal_gaussian_logpdf(data[:L], mu[:L], sigmasq)
                               for mu, sigmasq in zip(mus, self.sigmasq_init)])

        ll_ar = np.column_stack([stats.diagonal_gaussian_logpdf(data[L:], mu[L:], sigmasq)
                               for mu, sigmasq in zip(mus, self.sigmasq)])


        # Compute the likelihood of the initial data and remainder separately
        return np.row_stack((ll_init, ll_ar))


class IndependentAutoRegressiveObservations(_AutoRegressiveObservationsBase):
    def __init__(self, K, D, M=0, lags=1):
        super(IndependentAutoRegressiveObservations, self).__init__(K, D, M, lags=lags)

        self._As = np.concatenate((.95 * np.ones((K, D, 1)), np.zeros((K, D, lags-1))), axis=2)
        self._log_sigmasq_init = np.zeros((K, D))
        self._log_sigmasq = np.zeros((K, D))

    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def As(self):
        return np.array([
                np.column_stack([np.diag(Ak[:,l]) for l in range(self.lags)])
            for Ak in self._As
        ])

    @As.setter
    def As(self, value):
        # TODO: extract the diagonal components
        raise NotImplementedError

    @property
    def params(self):
        return self._As, self.bs, self.Vs, self._log_sigmasq

    @params.setter
    def params(self, value):
        self._As, self.bs, self.Vs, self._log_sigmasq = value

    def permute(self, perm):
        self.mu_init = self.mu_init[perm]
        self._As = self._As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    def _compute_mus(self, data, input, mask, tag):
        """
        Re-implement compute_mus for this class since we can do it much
        more efficiently than in the general AR case.
        """
        T, D = data.shape
        As, bs, Vs = self.As, self.bs, self.Vs

        # Instantaneous inputs, lagged data, and bias
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :self.M, None])[:, :, :, 0]
        for l in range(self.lags):
            mus += As[:, :, l] * data[self.lags-l-1:-l-1, None, :]
        mus += bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.diagonal_gaussian_logpdf(data[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.diagonal_gaussian_logpdf(data[L:, None, :], mus[L:], self.sigmasq)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        D, M = self.D, self.M

        for d in range(self.D):
            # Collect data for this dimension
            xs, ys, weights = [], [], []
            for (Ez, _, _), data, input, mask in zip(expectations, datas, inputs, masks):
                # Only use data if it is complete
                if np.all(mask[:, d]):
                    xs.append(
                        np.hstack([data[self.lags-l-1:-l-1, d:d+1] for l in range(self.lags)]
                                  + [input[self.lags:, :M], np.ones((data.shape[0]-self.lags, 1))]))
                    ys.append(data[self.lags:, d])
                    weights.append(Ez[self.lags:])

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            weights = np.concatenate(weights)

            # If there was no data for this dimension then skip it
            if len(xs) == 0:
                self.As[:, d, :] = 0
                self.Vs[:, d, :] = 0
                self.bs[:, d] = 0
                continue

            # Otherwise, fit a weighted linear regression for each discrete state
            for k in range(self.K):
                # Check for zero weights (singular matrix)
                if np.sum(weights[:, k]) < self.lags + M + 1:
                    self.As[k, d] = 1.0
                    self.Vs[k, d] = 0
                    self.bs[k, d] = 0
                    self._log_sigmasq[k, d] = 0
                    continue

                # Solve for the most likely A,V,b (no prior)
                Jk = np.sum(weights[:, k][:, None, None] * xs[:,:,None] * xs[:, None,:], axis=0)
                hk = np.sum(weights[:, k][:, None] * xs * ys[:, None], axis=0)
                muk = np.linalg.solve(Jk, hk)

                self.As[k, d] = muk[:self.lags]
                self.Vs[k, d] = muk[self.lags:self.lags+M]
                self.bs[k, d] = muk[-1]

                # Update the variances
                yhats = xs.dot(np.concatenate((self.As[k, d], self.Vs[k, d], [self.bs[k, d]])))
                sqerr = (ys - yhats)**2
                sigma = np.average(sqerr, weights=weights[:, k], axis=0) + 1e-16
                self._log_sigmasq[k, d] = np.log(sigma)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, self.sigmasq

        # Sample the initial condition
        if xhist.shape[0] < self.lags:
            sigma_init = self.sigmasq_init[z] if with_noise else 0
            return self.mu_init[z] + np.sqrt(sigma_init) * npr.randn(D)

        # Otherwise sample the AR model
        muz = bs[z].copy()
        for lag in range(self.lags):
            muz += As[z, :, lag] * xhist[-lag - 1]

        sigma = sigmas[z] if with_noise else 0
        return muz + np.sqrt(sigma) * npr.randn(D)


# Robust autoregressive models with diagonal Student's t noise
class _RobustAutoRegressiveObservationsMixin(object):
    """
    Mixin for AR models where the noise is distributed according to a
    multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    We use this equivalence to perform the M step (update of Sigma and tau)
    via an inner expectation maximization algorithm.

    This mixin mus be used in conjunction with either AutoRegressiveObservations or
    AutoRegressiveDiagonalNoiseObservations, which provides the parameterization for
    Sigma.  The mixin does not capitalize on structure in Sigma, so it will pay
    a small complexity penalty when used in conjunction with the diagonal noise model.
    """
    def __init__(self, K, D, M=0, lags=1,
                 l2_penalty_A=1e-8,
                 l2_penalty_b=1e-8,
                 l2_penalty_V=1e-8):

        super(_RobustAutoRegressiveObservationsMixin, self).\
            __init__(K, D, M=M, lags=lags,
                     l2_penalty_A=l2_penalty_A,
                     l2_penalty_b=l2_penalty_b,
                     l2_penalty_V=l2_penalty_V)
        self._log_nus = np.log(4) * np.ones(K)

        J_diag = np.concatenate((l2_penalty_A * np.ones(D * lags),
                                 l2_penalty_V * np.ones(M),
                                 l2_penalty_b * np.ones(1)))
        self.J0 = np.tile(np.diag(J_diag)[None, :, :], (K, 1, 1))
        self.h0 = np.zeros((K, D * lags + M + 1, D))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return super(_RobustAutoRegressiveObservationsMixin, self).params + (self._log_nus,)

    @params.setter
    def params(self, value):
        self._log_nus = value[-1]
        super(_RobustAutoRegressiveObservationsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_RobustAutoRegressiveObservationsMixin, self).permute(perm)
        self._log_nus = self._log_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        # Compute the likelihood of the initial data and remainder separately
        # stats.multivariate_studentst_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), (D,D), and (,)
        # arrays as inputs
        ll_init = np.column_stack([stats.multivariate_normal_logpdf(data[:L], mu[:L], Sigma)
                               for mu, Sigma in zip(mus, self.Sigmas_init)])

        ll_ar = np.column_stack([stats.multivariate_studentst_logpdf(data[L:], mu[L:], Sigma, nu)
                               for mu, Sigma, nu in zip(mus, self.Sigmas, self.nus)])

        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, datas, inputs, masks, tags, num_em_iters=1):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t
        for complete details.
        """
        self._m_step_ar(expectations, datas, inputs, masks, tags, num_em_iters)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_ar(self, expectations, datas, inputs, masks, tags, num_em_iters):
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[lags-l-1:-l-1] for l in range(lags)]
                          + [input[lags:, :self.M], np.ones((data.shape[0]-lags, 1))]))
            ys.append(data[lags:])
            Ezs.append(Ez[lags:])

        for itr in range(num_em_iters):
            # E Step: compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]

                # nu: (K,)  mus: (T, K, D)  sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
                alpha = self.nus / 2 + D/2
                sqrt_Sigmas = np.linalg.cholesky(self.Sigmas)
                beta = self.nus / 2 + 1/2 * stats.batch_mahalanobis(sqrt_Sigmas, y[:, None, :] - mus)
                taus.append(alpha / beta)

            # M step: Fit the weighted linear regressions for each K and D
            # This is exactly the same as the M-step for the AutoRegressiveObservations,
            # but it has an extra scaling factor of tau applied to the weight.
            J = self.J0.copy()
            h = self.h0.copy()
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                weight = Ez * tau
                # Einsum is concise but slow!
                # J += np.einsum('tk, ti, tj -> kij', weight, x, x)
                # h += np.einsum('tk, ti, td -> kid', weight, x, y)
                # Do weighted products for each of the k states
                for k in range(K):
                    weighted_x = x * weight[:, k:k+1]
                    J[k] += np.dot(weighted_x.T, x)
                    h[k] += np.dot(weighted_x.T, y)

            mus = np.linalg.solve(J, h)
            self.As = np.swapaxes(mus[:, :D*lags, :], 1, 2)
            self.Vs = np.swapaxes(mus[:, D*lags:D*lags+M, :], 1, 2)
            self.bs = mus[:, -1, :]

            # Update the covariance
            sqerr = np.zeros((K, D, D))
            weight = np.zeros(K)
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                yhat = np.matmul(x[None, :, :], mus)
                resid = y[None, :, :] - yhat
                sqerr += np.einsum('tk,kti,ktj->kij', Ez * tau, resid, resid)
                weight += np.sum(Ez, axis=0)

            self.Sigmas = sqerr / weight[:, None, None] + 1e-8 * np.eye(D)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        Update the degrees of freedom parameter of the multivariate t distribution
        using a generalized Newton update. See notes in the ssm repo.
        """
        K, D, L = self.K, self.D, self.lags
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for (Ez, _, _,), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # nu: (K,)  mus: (T, K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            mus = np.swapaxes(self._compute_mus(data, input, mask, tag), 0, 1)

            alpha = self.nus/2 + D/2
            sqrt_Sigma = np.linalg.cholesky(self.Sigmas)
            # TODO: Performance could be improved by iterating over K outside batch_mahalanobis
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(sqrt_Sigma, data[L:, None, :] - mus[L:])

            E_taus += np.sum(Ez[L:, :] * alpha / beta, axis=0)
            E_logtaus += np.sum(Ez[L:, :] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, Vs, Sigmas, nus = self.D, self.As, self.bs, self.Vs, self.Sigmas, self.nus
        if xhist.shape[0] < self.lags:
            S = np.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + np.dot(S, npr.randn(D))
        else:
            mu = Vs[z].dot(input[:self.M]) + bs[z]
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            S = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
            return mu + np.dot(S, npr.randn(D))


class RobustAutoRegressiveObservations(_RobustAutoRegressiveObservationsMixin, AutoRegressiveObservations):
    """
    AR model where the noise is distributed according to a multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    Here, Sigma is a general covariance matrix.
    """
    pass


class RobustAutoRegressiveObservationsNoInput(RobustAutoRegressiveObservations):
    """
    RobusAutoRegressiveObservations model without the inputs.
    """
    def __init__(self, K, D, M=0, lags=1,
             l2_penalty_A=1e-8,
             l2_penalty_b=1e-8,
             l2_penalty_V=1e-8):

        super(RobustAutoRegressiveObservationsNoInput, self).\
            __init__(K, D, M=0, lags=lags,
                     l2_penalty_A=l2_penalty_A,
                     l2_penalty_b=l2_penalty_b,
                     l2_penalty_V=l2_penalty_V)



class RobustAutoRegressiveDiagonalNoiseObservations(
    _RobustAutoRegressiveObservationsMixin, AutoRegressiveDiagonalNoiseObservations):
    """
    AR model where the noise is distributed according to a multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    Here, Sigma is a diagonal covariance matrix.
    """
    pass

# Robust autoregressive models with diagonal Student's t noise
class AltRobustAutoRegressiveDiagonalNoiseObservations(AutoRegressiveDiagonalNoiseObservations):
    """
    An alternative formulation of the robust AR model where the noise is
    distributed according to a independent scalar t distribution,

    For each output dimension d,

        epsilon_d ~ t(0, sigma_d^2, nu_d)

    which is equivalent to,

        tau_d ~ Gamma(nu_d/2, nu_d/2)
        epsilon_d | tau_d ~ N(0, sigma_d^2 / tau_d)

    """
    def __init__(self, K, D, M=0, lags=1):
        super(AltRobustAutoRegressiveDiagonalNoiseObservations, self).__init__(K, D, M=M, lags=lags)
        self._log_nus = np.log(4) * np.ones((K, D))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        super(AltRobustAutoRegressiveDiagonalNoiseObservations, self).permute(perm)
        self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        mus = np.swapaxes(self._compute_mus(data, input, mask, tag), 0, 1)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.diagonal_gaussian_logpdf(data[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.independent_studentst_logpdf(data[L:, None, :], mus[L:], self.sigmasq, self.nus)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, datas, inputs, masks, tags,
               num_em_iters=1, optimizer="adam", num_iters=10, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t
        for complete details.
        """
        self._m_step_ar(expectations, datas, inputs, masks, tags, num_em_iters)
        self._m_step_nu(expectations, datas, inputs, masks, tags, optimizer, num_iters, **kwargs)

    def _m_step_ar(self, expectations, datas, inputs, masks, tags, num_em_iters):
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[lags-l-1:-l-1] for l in range(lags)]
                          + [input[lags:, :self.M], np.ones((data.shape[0]-lags, 1))]))
            ys.append(data[lags:])
            Ezs.append(Ez[lags:])

        for itr in range(num_em_iters):
            # E Step: compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                # mus = self._compute_mus(data, input, mask, tag)
                # sigmas = self._compute_sigmas(data, input, mask, tag)
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]

                # nu: (K,D)  mus: (T, K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
                alpha = self.nus / 2 + 1/2
                beta = self.nus / 2 + 1/2 * (y[:, None, :] - mus)**2 / self.sigmasq
                taus.append(alpha / beta)

            # M step: Fit the weighted linear regressions for each K and D
            J = np.tile(np.eye(D * lags + M + 1)[None, None, :, :], (K, D, 1, 1))
            h = np.zeros((K, D,  D*lags + M + 1,))
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                robust_ar_statistics(Ez, tau, x, y, J, h)

            mus = np.linalg.solve(J, h)
            self.As = mus[:, :, :D*lags]
            self.Vs = mus[:, :, D*lags:D*lags+M]
            self.bs = mus[:, :, -1]

            # Fit the variance
            sqerr = 0
            weight = 0
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                yhat = np.matmul(x[None, :, :], np.swapaxes(mus, -1, -2))
                sqerr += np.einsum('tk, tkd, ktd -> kd', Ez, tau, (y - yhat)**2)
                weight += np.sum(Ez, axis=0)
            self._log_sigmasq = np.log(sqerr / weight[:, None] + 1e-16)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags, optimizer, num_iters, **kwargs):
        K, D, L = self.K, self.D, self.lags
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for (Ez, _, _,), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # nu: (K,D)  mus: (T, K, D)  sigmas: (K, D)  y: (T, D)  -> w: (T, K, D)
            mus = np.swapaxes(self._compute_mus(data, input, mask, tag), 0, 1)

            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (data[L:, None, :] - mus[L:])**2 / self.sigmasq

            E_taus += np.sum(Ez[L:, :, None] * alpha / beta, axis=0)
            E_logtaus += np.sum(Ez[L:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmasq, nus = self.D, self.As, self.bs, self.sigmasq, self.nus
        if xhist.shape[0] < self.lags:
            sigma_init = self.sigmasq_init[z] if with_noise else 0
            return self.mu_init[z] + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            var = sigmasq[z] / tau if with_noise else 0
            return mu + np.sqrt(var) * npr.randn(D)


class VonMisesObservations(Observations):
    def __init__(self, K, D, M=0):
        super(VonMisesObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K, D)))

    @property
    def params(self):
        return self.mus, self.log_kappas

    @params.setter
    def params(self, value):
        self.mus, self.log_kappas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        mus, kappas = self.mus, np.exp(self.log_kappas)

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.vonmises_logpdf(data[:, None, :], mus, kappas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        return npr.vonmises(self.mus[z], kappas[z], D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])  # T x D
        assert x.shape[0] == weights.shape[0]

        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T x 2 x D

        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D
        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D

        mus_k = np.divide(r_k, r_norm[:, None])  # K x 2 x D
        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D

        mask = (r_norm.sum(1) == 0)
        mus_k[mask] = 0
        r_bar[mask] = 0

        # Approximation
        kappa0 = r_bar * (self.D + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K,D

        kappa0[kappa0 == 0] += 1e-6

        for k in range(self.K):
            self.mus[k] = np.arctan2(*mus_k[k])  #
            self.log_kappas[k] = np.log(kappa0[k])  # K, D

    def smooth(self, expectations, data, input, tag):
        mus = self.mus
        return expectations.dot(mus)
