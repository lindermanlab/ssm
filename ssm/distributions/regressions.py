import jax.numpy as np
import jax.random
import jax.scipy.special as spsp
import jax.scipy.stats as spst
from jax.ops import index_update, index_add
from functools import partial

from numba import jit

from ssm.distributions.base import Distribution, \
    ExponentialFamilyDistribution, CompoundDistribution, ConjugatePrior
from ssm.distributions.distributions import Bernoulli, Categorical, Poisson
from ssm.util import generalized_outer


class LinearRegression(ExponentialFamilyDistribution):
    """A linear regression model. The optimal weights are computed
    via the expected sufficient statistics of the data.
    """
    def __init__(self, weights, covariance_matrix,
                 bias=None, fit_bias=True,
                 prior=None, **kwargs):
        self.weights = weights
        self.out_dim, self.in_dim = weights.shape

        # Extract the bias
        if bias is not None:
            assert bias.shape == (self.out_dim,)
            self.fit_bias = True
            self.bias = bias
        elif fit_bias:
            self.fit_bias = True
            self.bias = np.zeros(self.out_dim)
        else:
            self.fit_bias = False

        assert covariance_matrix.shape == (self.out_dim, self.out_dim)
        self.covariance_matrix = covariance_matrix
        super(LinearRegression, self).__init__(prior=prior)

    @classmethod
    def from_example(cls, data, covariates, bias=None, fit_bias=True, **kwargs):
        out_dim = data.shape[-1]
        in_dim = covariates.shape[-1]
        return cls(weights=np.zeros((out_dim, in_dim)),
                   covariance_matrix=np.eye(out_dim),
                   bias=bias, fit_bias=fit_bias,
                   **kwargs)

    @property
    def dimension(self):
        return self.combined_weights.shape

    @property
    def bias(self):
        assert self.fit_bias, "`intercept` only exists with `fit_intercept=True`."
        return self._bias

    @bias.setter
    def bias(self, value):
        assert self.fit_bias, "`intercept` only exists with `fit_intercept=True`"
        self._bias = value

    @property
    def combined_weights(self):
        return np.column_stack((self.weights, self.bias)) \
            if self.fit_bias else self.weights

    @combined_weights.setter
    def combined_weights(self, value):
        if self.fit_bias:
            self.weights, self.bias = value[:, :-1], value[:, -1]
        else:
            self.weights = value

    @property
    def covariance_matrix(self):
        return self._covariance_tril @ self._covariance_tril.T

    @covariance_matrix.setter
    def covariance_matrix(self, value):
        self._covariance_tril = np.linalg.cholesky(value)

    @property
    def unconstrained_params(self):
        return self.combined_weights, self._covariance_tril

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        if self.fit_bias:
            self.weights, self._bias, self._covariance_tril = value
        else:
            self.weights, self._covariance_tril = value

    def combined_covariates(self, data, covariates):
        all_covariates = [covariates]
        if self.fit_bias:
            all_covariates.append(np.ones((data.shape[0], 1)))
        return all_covariates

    def predict(self, data, covariates):
        prediction = covariates @ self.weights.T
        if self.fit_bias:
            prediction += self.bias
        return prediction

    def sample(self, rng, sample_shape=(), covariates=None):
        assert covariates is not None, "Regression needs covariates!"
        prediction = self.predict(None, covariates)
        data = jax.random.multivariate_normal(rng, prediction, self.covariance_matrix)
        return data

    def log_prior(self):
        return self.prior.log_prob((self.combined_weights,
                                    self.covariance_matrix))

    def log_prob(self, data, covariates, **kwargs):
        from jax import jit
        from jax.scipy.stats import multivariate_normal
        predictions = self.predict(data, covariates)
        logpdf = jit(multivariate_normal.logpdf)
        lps = logpdf(data, predictions, self.covariance_matrix)
        return np.array(lps)

    def sufficient_statistics(self, data, covariates, **kwargs):
        all_covariates = self.combined_covariates(data, covariates)
        return generalized_outer(all_covariates, all_covariates), \
               generalized_outer(data, all_covariates), \
               generalized_outer(data, data)


class _AutoRegressionMixin(object):
    """
    We need to override a few functions to change a `LinearRegression`
    or `MultivariateStudentsTRegression` object into an AutoRegression.
    """
    @property
    def autoregression_weights(self):
        num_lags, dim = self.num_lags, self.out_dim
        # Equivalent to list comprehension...
        # As = np.reshape(self.weights[:, :dim * num_lags],
        #                 (dim, num_lags, dim))
        # return np.swapaxes(As, 1, 0)
        return [self.weights[:, (i * dim):((i + 1) * dim)]
                for i in range(num_lags)]

    @property
    def covariate_weights(self):
        num_lags, dim = self.num_lags, self.out_dim
        return self.weights[:, dim * num_lags:]

    @property
    def covariate_dim(self):
        return self.in_dim - self.num_lags * self.out_dim

    def combined_covariates(self, data, covariates=None, **kwargs):
        num_lags, out_dim, = self.num_lags, self.out_dim
        all_covariates = []
        for lag in range(1, num_lags+1):
            # TODO: Try to compute the avoid memory allocation
            all_covariates.append(
                np.row_stack([np.zeros((lag, out_dim)), data[:-lag]]))
        if covariates is not None:
            all_covariates.append(covariates)
        if self.fit_bias:
            all_covariates.append(np.ones((data.shape[0], 1)))
        return all_covariates

    def predict(self, data, covariates=None, **kwargs):
        prediction = np.zeros_like(data)
        for i, weights in enumerate(self.autoregression_weights):
            lag = i + 1
            prediction = index_add(prediction, slice(lag, None),
                                   data[:-lag] @ weights.T)
        if covariates is not None:
            prediction += covariates @ self.covariate_weights.T
        if self.fit_bias:
            prediction += self.bias
        return prediction

    def predict_next(self, preceding_data, covariates=None, **kwargs):
        num_lags, out_dim, in_dim = self.num_lags, self.out_dim, self.in_dim
        preceding_data = preceding_data.reshape((-1, out_dim))
        # Predict the next observation give the most recent `num_lags`
        # of preceding data
        prediction = np.zeros(out_dim)
        for i, weights in enumerate(self.autoregression_weights):
            lag = i + 1
            if len(preceding_data) >= lag:
                prediction += np.dot(weights, preceding_data[-lag])
            else:
                break
        if covariates is not None:
            prediction += np.dot(self.covariate_weights, covariates)
        if self.fit_bias:
            prediction += self.bias

        return prediction

    def log_prob(self, data, covariates=None, **kwargs):
        # Same as for regular regressions but now `covariates` can be None.
        lps = super(_AutoRegressionMixin, self).\
            log_prob(data, covariates, **kwargs)

        # Zero out initial data since we don't have all the covariates
        lps = index_update(lps, slice(0, self.num_lags), 0)
        return lps

    def sufficient_statistics(self, data, covariates=None, **kwargs):
        # Same as for regular regressions but now `covariates` can be None.
        stats = super(_AutoRegressionMixin, self).\
            sufficient_statistics(data, covariates, **kwargs)

        # Zero out initial stats since we don't have all the covariates
        return [index_update(ss, slice(0, self.num_lags), 0)
                 for ss in stats]


class LinearAutoRegression(_AutoRegressionMixin, LinearRegression):
    def __init__(self, weights, covariance_matrix,
                 bias=None, fit_bias=True, num_lags=1, prior=None,
                 **kwargs):
        assert num_lags > 0
        self.num_lags = num_lags
        assert weights.shape[1] >= num_lags * weights.shape[0]
        super(LinearAutoRegression, self).__init__(
            weights=weights, covariance_matrix=covariance_matrix,
            bias=bias, fit_bias=fit_bias, prior=prior)

    @classmethod
    def from_example(cls, data, covariates=None,
                     bias=None, fit_bias=True, num_lags=1,
                     **kwargs):
        out_dim = data.shape[-1]
        in_dim = num_lags * out_dim
        if covariates is not None:
            in_dim += covariates.shape[-1]

        return cls(weights=np.zeros((out_dim, in_dim)),
                   covariance_matrix=np.eye(out_dim),
                   bias=bias, fit_bias=fit_bias, num_lags=num_lags,
                   **kwargs)

    def sample(self, rng, sample_shape=(), preceding_data=None, covariates=None):
        num_lags, out_dim, in_dim = self.num_lags, self.out_dim, self.in_dim
        prediction = self.predict_next(preceding_data, covariates=covariates)
        return jax.random.multivariate_normal(rng, prediction, self.covariance_matrix)


class MultivariateStudentsTLinearRegression(LinearRegression,
                                            CompoundDistribution):
    """A linear regression model with multivariate Student's t errors.
    This class is to `MultivariateNormalLinearRegression` as
    `MultivariateStudentsT` is to `MultivariateNormal`.
    """
    def __init__(self, weights, covariance_matrix, dof,
                 bias=None, fit_bias=True,
                 prior=None, **kwargs):
        self.dof = dof
        super(MultivariateStudentsTLinearRegression, self).__init__(
            weights, covariance_matrix, bias=bias, fit_bias=fit_bias,
            prior=prior)

    @classmethod
    def from_example(cls, data, covariates, bias=None, fit_bias=True, **kwargs):
        out_dim = data.shape[-1]
        in_dim = covariates.shape[-1]
        return cls(weights=np.zeros((out_dim, in_dim)),
                   covariance_matrix=np.eye(out_dim),
                   bias=bias, fit_bias=fit_bias,
                   dof=out_dim+2,
                   **kwargs)

    @property
    def dof(self):
        return np.exp(self._log_dof)

    @dof.setter
    def dof(self, value):
        self._log_dof = np.log(value)

    @property
    def unconstrained_nonconj_params(self):
        return self._log_dof

    @unconstrained_nonconj_params.setter
    def unconstrained_nonconj_params(self, value):
        self._log_dof = value

    def sample(self, rng, sample_shape=(), covariates=None):
        assert covariates is not None, "Regression needs covariates!"
        prediction = self.predict(None, covariates)
        key1, key2 = jax.random.split(rng, 2)
        scale = jax.random.gamma(key1, self.dof / 2.0, 2.0 / self.dof)
        data = jax.random.multivariate_normal(
            key2, prediction, self.covariance_matrix / scale)
        return data

    def log_prob(self, data, covariates):
        predictions = self.predict(data, covariates)
        chol, dof, dim = self._covariance_tril, self.dof, self.out_dim
        assert data.ndim == 2 and data.shape[1] == dim

        # Quadratic term
        tmp = np.linalg.solve(chol, (data - predictions).T).T
        lp = - 0.5 * (dof + dim) * np.log1p(np.sum(tmp**2, axis=1) / dof)

        # Normalizer
        lp += spsp.gammaln(0.5 * (dof + dim)) - spsp.gammaln(0.5 * dof)
        lp += - 0.5 * dim * np.log(np.pi) - 0.5 * dim * np.log(dof)
        # L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]
        lp += -np.sum(np.log(np.diag(chol)))
        return lp

    def conditional_expectations(self, data, covariates, **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.`
        """
        # The auxiliary precision \tau is conditionally gamma distributed.
        alpha = 0.5 * (self.dof + self.out_dim)
        prediction = self.predict(data, covariates)
        tmp = np.linalg.solve(self._covariance_tril, (data - prediction).T).T
        beta = 0.5 * (self.dof + np.sum(tmp * tmp, axis=1))

        # Compute gamma expectations
        E_tau = alpha / beta
        E_log_tau = spsp.digamma(alpha) - np.log(beta)
        return E_tau, E_log_tau

    def expected_sufficient_statistics(self, expectations, data, covariates=None, **kwargs):
        """Given the precision, the data is conditionally Gaussian.
        """
        all_covariates = self.combined_covariates(data, covariates)
        E_tau, _ = expectations

        xxT = generalized_outer(all_covariates, all_covariates)
        yxT = generalized_outer(data, all_covariates)
        yyT = generalized_outer(data, data)
        return np.einsum('n,nij->nij', E_tau, xxT), \
               np.einsum('n,nij->nij', E_tau, yxT), \
               np.einsum('n,nij->nij', E_tau, yyT)

    def expected_log_prob(self, expectations, data, **kwargs):
        """Compute the expected log probability.  This function will be
        optimized with respect to the remaining, non-conjugate parameters
        of the distribution.
        """
        E_tau, E_log_tau = expectations
        hdof = 0.5 * self.dof
        return hdof * np.log(hdof) \
            - spsp.gammaln(hdof) + (hdof - 1) * E_log_tau \
            - hdof * E_tau

    def fit(self, dataset, num_iters=20, tol=1e-3):
        # Directly call the CompoundDistribution.fit method as it is
        # not first in line given the inheritance structure.
        CompoundDistribution.fit(self, dataset, num_iters=num_iters, tol=tol)


class MultivariateStudentsTAutoRegression(_AutoRegressionMixin,
                                          MultivariateStudentsTLinearRegression):
    def __init__(self, weights, covariance_matrix, dof,
                 bias=None, fit_bias=True,
                 num_lags=1,
                 prior=None):
        assert num_lags > 0
        self.num_lags = num_lags
        assert weights.shape[1] >= num_lags * weights.shape[0]
        super(MultivariateStudentsTAutoRegression, self).__init__(
            weights=weights, covariance_matrix=covariance_matrix, dof=dof,
            bias=bias, fit_bias=fit_bias, prior=prior)

    @classmethod
    def from_example(cls, data, covariates=None,
                     bias=None, fit_bias=True,
                     num_lags=1,
                     **kwargs):
        out_dim = data.shape[-1]
        in_dim = num_lags * out_dim
        if covariates is not None:
            in_dim += covariates.shape[-1]

        return cls(weights=np.zeros((out_dim, in_dim)),
                   covariance_matrix=np.eye(out_dim),
                   dof=out_dim + 2,
                   bias=bias, fit_bias=fit_bias,
                   num_lags=num_lags,
                   **kwargs)

    def sample(self, rng, sample_shape=(), preceding_data=None, covariates=None):
        num_lags, out_dim, in_dim = self.num_lags, self.out_dim, self.in_dim
        prediction = self.predict_next(preceding_data, covariates=covariates)
        key1, key2 = jax.random.split(rng, 2)
        scale = jax.random.gamma(key1, self.dof / 2, 2 / self.dof)
        return jax.random.multivariate_normal(key2,
            prediction, self.covariance_matrix / scale)

    def conditional_expectations(self, data, covariates=None, **kwargs):
        return super(MultivariateStudentsTAutoRegression, self).\
            conditional_expectations(data, covariates, **kwargs)


class GeneralizedLinearModel(Distribution):
    """A simple GLM class with support for various observation models
    and link functions. Here we just fit the weights with BFGS, though
    IRLS would be a nice alternative for canonical link functions.
    """
    def __init__(self,
                 weights,
                 observations="bernoulli",
                 link="logistic",
                 prior=None):

        super(GeneralizedLinearModel, self).__init__(prior=prior)

        # TODO: Support batches of linear regressions
        assert weights.ndim == 2, "Weights must be a 2D array"
        self.out_dim, self.in_dim = weights.shape
        self.weights = weights

        # Store the keyword argument. It's necessary for the fitting.
        self.observations = observations
        self.observation_class = dict(
            bernoulli=Bernoulli,
            categorical=Categorical,
            poisson=Poisson
        )[observations]

        # Store the keyword argument. It's necessary for the fitting.
        self.link = link
        if callable(link):
            # user supplied a link function
            self.link_function = link
        else:
            self.link_function = dict(
                logistic=spsp.expit,
                sigmoid=spsp.expit,
                expit=spsp.expit,
                softmax=partial(spsp.softmax, axis=-1),
                softplus=lambda x: np.log1p(np.exp(x)),
                exp=np.exp,
            )[link]

    @property
    def dimension(self):
        return self.weights.shape

    @property
    def unconstrained_params(self):
        return self.weights

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self.weights = value

    def predict(self, covariates):
        return self.link_function(np.matmul(covariates, self.weights.T))

    def sample(self, key,
               sample_shape=(),
               covariates=None):
        raise NotImplementedError
        # key1, key2 = jax.random.split(key, 2)
        # sample_covariates = (covariates is None)
        # if sample_covariates:
        #     shape = sample_shape + (self.in_dim,)
        #     covariates = jax.random.normal(key1, shape=shape)
        # prediction = self.predict(covariates)
        # dist = self.observation_class(prediction)
        # targets = dist.sample(key2)
        # return targets, covariates if sample_covariates else targets

    def log_prob(self, data, covariates):
        prediction = self.predict(covariates)
        dist = self.observation_class(prediction)
        lps = dist.log_prob(data)
        # Sum out the event dimension if present
        return lps if lps.ndim == 1 else np.sum(lps, axis=-1)
        # return np.sum(lps, axis=tuple(range(1, lps.ndim)))
