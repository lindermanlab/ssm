import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as spsp
import autograd.scipy.stats as spst
from functools import partial

from ssm.distributions.base import Distribution, \
    ExponentialFamilyDistribution, CompoundDistribution, ConjugatePrior


class Bernoulli(ExponentialFamilyDistribution):
    """A Bernoulli distribution that implements the
    ExponentialFamilyDistribution interface.
    """
    def __init__(self, probs, prior=None):
        self.probs = probs
        super(Bernoulli, self).__init__(prior=prior)

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(probs=0.5 * np.ones(dim), **kwargs)

    @property
    def dimension(self):
        return 1

    @property
    def unconstrained_params(self):
        return self._logits

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._logits = value

    @property
    def probs(self):
        return spsp.expit(self._logits)

    @probs.setter
    def probs(self, value):
        with np.errstate(divide='ignore'):
            self._logits = spsp.logit(value)

    def log_prior(self):
        return self.prior.log_prob(self.probs)

    def sufficient_statistics(self, data, **kwargs):
        return (data, 1-data)


class Beta(ConjugatePrior):
    """A conjugate prior distribution for the Bernoulli, Binomial,
    Geometric, and NegativeBinomial/GammaPoisson (with fixed shape)
    distributions.

    ..math::
        \pi \sim \mathrm{Be}(\beta_1, \beta_0)
        p(\pi) \propto exp\{(\beta_1 - 1) \log \pi + (\beta_0 - 1) \log (1-\pi)\}

    where :math:`\beta_1` and :math:`\beta_0` are non-negative
    pseudo-observations of 'heads' and 'tails', respectively.

    In the Bernoulli likelihood, for example,
    .. math::
        p(x \mid \pi) = \exp\{x \log \pi + (1-x) \log (1-\pi)}

    with sufficient statistics
    .. :math:
        t(x)_1 = x
        t(x)_2 = (1-x)

    The prior lends pseudo-observations,
    .. math:
        s_1 = \beta_1
        s_0 = \beta_0

    We default to a uniform prior with :math:`\beta_1 = 1` and
    :math:`\beta_0 = 1` so that the MAP estimate coincides with the
    MLE.
    """
    def __init__(self, shape1=1.0, shape0=1.0, dimension=None):
        self.shape1 = shape1
        self.shape0 = shape0
        super(Beta, self).__init__()

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(shape1=np.ones(dim),
                   shape0=np.ones(dim),
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self._log_shape1, self._log_shape0

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._log_shape1, self._log_shape0 = value

    @property
    def shape1(self):
        return np.exp(self._log_shape1)

    @shape1.setter
    def shape1(self, value):
        with np.errstate(divide='ignore'):
            self._log_shape1 = np.log(value)

    @property
    def shape0(self):
        return np.exp(self._log_shape0)

    @shape0.setter
    def shape0(self, value):
        with np.errstate(divide='ignore'):
            self._log_shape0 = np.log(value)

    def log_prob(self, data, **kwargs):
        return spst.beta.logpdf(data, self.shape1, self.shape0)

    @property
    def mode(self):
        """Return the mode of the prior if it is well defined;
        otherwise return the mean.
        """
        p1 = (self.shape1 - 1) / (self.shape1 + self.shape0 - 2)
        p2 = (self.shape1) / (self.shape1 + self.shape0)
        valid = (self.shape1 > 0) * (self.shape0 > 0)
        return dict(probs=p1 * valid + p2 * (~valid))

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Convert the statistics and counts back into NIW parameters.
        """
        shape1, shape0 = stats
        return cls(shape1, shape0)

    @property
    def pseudo_obs(self):
        """Return the pseudo observations under this prior.
        These should match up with the sufficient statistics of
        the conjugate distribution.
        """
        return self.shape1, self.shape0

    @property
    def pseudo_counts(self):
        """Return the pseudo observations under this prior."""
        return 0


class Binomial(ExponentialFamilyDistribution):
    """A binomial distribution that implements the
    ExponentialFamilyDistribution interface.
    """
    def __init__(self, probs, total_count, prior=None):
        self.total_count = total_count
        self.probs = probs
        super(Binomial, self).__init__(prior=prior)

    @property
    def dimension(self):
        return 1

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(probs=0.5 * np.ones(dim),
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self._logits

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._logits = value

    @property
    def probs(self):
        return spsp.expit(self._logits)

    @probs.setter
    def probs(self, value):
        with np.errstate(divide='ignore'):
            self._logits = spsp.logit(value)

    def log_prior(self):
        return self.prior.log_prob(self.probs)

    def sufficient_statistics(self, data, **kwargs):
        return (data, self.total_count - data)


class Categorical(ExponentialFamilyDistribution):
    """A categorical distribution that implements the
    ExponentialFamilyDistribution interface.
    """
    def __init__(self, probs, prior=None):
        self.probs = probs
        super(Categorical, self).__init__(prior=prior)

    @property
    def dimension(self):
        return self.probs.shape[-1]

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(probs=np.ones(dim) / dim, **kwargs)

    @property
    def unconstrained_params(self):
        return self._logits

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._logits = value

    @property
    def probs(self):
        return np.exp(
            self._logits - spsp.logsumexp(self._logits, axis=-1, keepdims=True))

    @probs.setter
    def probs(self, value):
        with np.errstate(divide='ignore'):
            self._logits = np.log(value)

    @property
    def num_categories(self):
        return self._logits.shape[-1]

    def log_prior(self):
        return self.prior.log_prob(self.probs)

    def log_prob(self, data, **kwargs):
        return np.log(self.probs)[data]

    def sufficient_statistics(self, data, **kwargs):
        from ssm.util import one_hot
        return (one_hot(data, self.num_categories),)


class Dirichlet(ConjugatePrior):
    """A conjugate prior distribution for the Categorical and Multinomial
    Distributions.

    ..math::
        \pi \sim \mathrm{Dir}(\alpha_0)

    where :math:`\alpha_0` is a non-negative concentration vector of
    length :math:`D`.

    The natural parameters of the categorical distribution
    are,
    .. math::
        \eta = \log \pi

    and they correspond to the sufficient statistics,
    .. math::
        t(x) = [I[x=1], ..., I[x=D]],

    where :math:`I[x=d]` is the indicator function of the categorical
    distribution.

    The prior lends pseudo-observations,
    .. math:
        s = \alpha_0

    (Note: we don't need pseudo_counts for this case.)

    We default to a uniform prior with :math:`\alpha_0 = 1`.
    """
    def __init__(self, concentration=1, dimension=1):
        self.concentration = concentration * np.ones(dimension)
        super(Dirichlet, self).__init__()

    @property
    def dimension(self):
        return self.concentration.shape[-1]

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(concentration=1.0, dimension=dim, **kwargs)

    @property
    def unconstrained_params(self):
        return self._logits_mean, self._log_concentration

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._logits_mean, self._log_concentration = value

    @property
    def concentration(self):
        return np.exp(self._log_concentration)

    @concentration.setter
    def concentration(self, value):
        self._log_concentration = np.log(value)

    def log_prob(self, data, **kwargs):
        return spst.dirichlet.logpdf(data, self.concentration) \
            if np.all(self.concentration > 0) else 0

    @property
    def mode(self):
        """Return the mode of the prior, which is only well defined if all
        entries of :math:`\alpha_0` are at least 1.
        """
        alpha = self.concentration
        if np.all(alpha >= 1):
            return dict(probs=(alpha - 1) / np.sum(alpha - 1, axis=-1, keepdims=True))
        else:
            return dict(probs=alpha / alpha.sum())

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Convert the statistics and counts back into NIW parameters.
        """
        concentration, = stats
        return cls(concentration, dimension=concentration.shape[0])

    @property
    def pseudo_obs(self):
        return (self.concentration,)

    @property
    def pseudo_counts(self):
        # This isn't actually used for the Dirichlet
        return -1


class Gamma(ConjugatePrior):
    """A conjugate prior distribution for the Poisson, Exponential,
    Gamma (fixed shape), Normal (fixed mean), and other distributions.

    ..math::
        \lambda \sim \mathrm{Ga}(\alpha_0, \beta_0)
        p(\lambda) \propto exp\{(\alpha_0-1) \log \lambda - \beta_0 \lambda\}

    where :math:`\alpha_0` and :math:`\beta_0` are non-negative
    shape and rate parameters, respectively.

    In the Poisson distribution, for example,
    .. math::
        p(x \mid \lambda) = 1/x! \exp\{x \log \lambda -\lambda\}

    with sufficient statistic :math:`t(x) = x`.

    The prior lends pseudo-observations and pseudo-counts,
    .. math:
        s = (\alpha_0 - 1)
        n = \beta_0

    We default to an improper prior with :math:`\alpha_0 = 1` and
    :math:`\beta_0 = 0` so that the MAP estimate coincides with the
    MLE.
    """
    def __init__(self, shape=1.0, rate=0.0, dimension=1):
        self.shape = shape
        self.rate = rate
        super(Gamma, self).__init__()

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(shape=np.ones(dim),
                   rate=np.ones(dim),
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self._log_shape, self._log_rate

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._log_shape, self._log_rate = value

    @property
    def shape(self):
        return np.exp(self._log_shape)

    @shape.setter
    def shape(self, value):
        self._log_shape = np.log(value)

    @property
    def rate(self):
        return np.exp(self._log_rate)

    @rate.setter
    def rate(self, value):
        self._log_rate = np.log(value)

    def log_prob(self, data, **kwargs):
        return spst.gamma.logpdf(data, self.shape, 1 / self.rate) \
            if self.shape > 0 else 0

    @property
    def mode(self):
        """Return the mode of the prior if it is well defined (i.e. if
        :math:`\alpha_0 > 1`); otherwise return the mean.
        """
        rate = (self.alpha0 - 1) / self.beta0 * (self.alpha0 > 1) + \
               self.alpha0 / self.beta0 * (self.alpha0 < 1)
        return dict(rate=rate)

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Convert the statistics and counts back into NIW parameters.
        """
        shape, = stats
        rate = counts
        return cls(shape, rate)

    @property
    def pseudo_obs(self):
        return (self.shape,)

    @property
    def pseudo_counts(self):
        return self.rate


class MatrixNormalInverseWishart(ConjugatePrior):
    """A conjugate prior distribution for a linear regression model,

    ..math::
        y | x ~ N(Ax, Sigma)

    where `A \in \mathbb{R}^{n \times p}` are the regression weights
    and `\Sigma \in \mathbb{S}^{n \times n}` is the noise covariance.
    Expanding the linear regression model,

    ..math::

        \log p(y | x) =
            -1/2 \log |\Sigma|
            -1/2 Tr((y - Ax)^\top \Sigma^{-1} (y - Ax))
          = -1/2 \log |\Sigma|
            -1/2 Tr(x x^\top A^\top \Sigma^{-1} A)
               + Tr(x y^\top \Sigma^{-1} A)
            -1/2 Tr(y y^\top \Sigma^{-1})

    Its natural parameters are
    .. math::
        \eta_1 = -1/2 A^\top \Sigma^{-1} A
        \eta_2 = \Sigma^{-1} A
        \eta_3 = -1/2 \Sigma^{-1}

    and they correspond to the sufficient statistics,
    .. math::
        t(x)_1 = x x^\top,
        t(x)_2 = y x^\top,
        t(x)_3 = y y^\top,

    The matrix-normal inverse-Wishart (MNIW) is a conjugate prior,

    ..math::
        A | \Sigma \sim \mathrm{N}(\vec(A) | \vec(M_0), \Sigma \kron V_0)
            \Sigma \sim \mathrm{IW}(\Sigma | \Psi_0, \nu_0)

    The prior parameters are:

        `M_0`: the prior mean of `A`
        `V_0`: the prior covariance of the columns of `A`
        `Psi_0`: the prior scale matrix for the noise covariance `\Sigma`
        `\nu_0`: the prior degrees of freedom for the noise covariance

    In the special case where the covariates are always one, `x = 1`, and
    hence the matrices `A` and `M_0` are really just column vectors `a` and
    `\mu_0`, the MNIW reduces to a NIW prior,

    ..math::
        a \sim \mathrm{NIW}{\mu_0, 1/V_0, \nu_0, \Psi_0}

    (`\kappa_0` is a precision in the NIW prior, whereas `V_0` is a covariance.)

    The MNIW pdf is proportional to,

    ..math::
        \log p(A , \Sigma) =
            -p/2 \log |\Sigma|
            -1/2 Tr(V_0^{-1} A^\top \Sigma^{-1} A)
               + Tr( V_0^{-1} M_0^\top \Sigma^{-1} A)
            -1/2 Tr(M_0 V_0^{-1} M_0^\top \Sigma^{-1})
            -(\nu_0 + n + 1)/2 \log|\Sigma|
            -1/2 Tr(\Psi_0 \Sigma^{-1})
            + c.

    Collecting terms, the prior contributes the following pseudo-counts
    and pseudo-observations,

    ..math::
        n_1 = \nu_0 + n + p + 1
        s_1 = V_0^{-1}
        s_2 = M_0 V_0^{-1}
        s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top

    We default to an improper prior, with `n_1 = 0` and
     `s_i = 0` for `i=1..3`.
    """
    def __init__(self, M0=0.0, V0=1e16, nu0=0.0, Psi0=1e-4, dimension=(1, 1)):
        out_dim, in_dim = dimension
        self.M0 = M0 * np.ones(dimension)
        self.V0 = np.dot(V0, np.eye(in_dim))
        self.nu0 = nu0 + out_dim + 1
        self.Psi0 = np.dot(Psi0, np.eye(out_dim))

    # TODO: Reparameterize parameters so that they can be optimized with autograd.
    @property
    def unconstrained_params(self):
        return ()

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        pass

    @property
    def in_dim(self):
        return self.M0.shape[-1]

    @property
    def out_dim(self):
        return self.M0.shape[-2]

    def log_prior(self):
        return 0

    def log_prob(self, data, **kwargs):
        """Compute the prior log probability of LinearRegression weights
        and covariance matrix under this MNIW prior.  The IW pdf is provided
        in scipy.stats.  The matrix normal pdf is,

        ..math::
            \log p(A | M, \Sigma, V) =
                -1/2 Tr \left[ V^{-1} (A - M)^\top \Sigma^{-1} (A - M) \right]
                -np/2 \log (2\pi) -n/2 \log |V| -p/2 \log |\Sigma|

              = -1/2 Tr(B B^T) -np/2 \log (2\pi) -n/2 \log |V| -p/2 \log |\Sigma|

        where

        ..math::
            B = U^{-1/2} (A - M) (V^T)^{-1/2}
        """
        weights, covariance_matrix = data

        # Evaluate the matrix normal log pdf
        lp = 0

        # \log p(A | M_0, \Sigma, V_0)
        if np.all(np.isfinite(self.V0)):
            Vsqrt = np.linalg.cholesky(self.V0)
            Ssqrt = np.linalg.cholesky(covariance_matrix)
            B = np.linalg.solve(Ssqrt, np.linalg.solve(
                Vsqrt, (weights - self.M0).T).T)
            lp += -0.5 * np.sum(B**2)
            lp += -self.out_dim * np.sum(np.log(np.diag(Vsqrt)))
            lp += -0.5 * self.in_dim * self.out_dim * np.log(2 * np.pi)
            lp += -self.in_dim * np.sum(np.log(np.diag(Ssqrt)))

        # For comparison, compute the big multivariate normal log pdf explicitly
        # Note: we have to do the kron in the reverse order of what is given
        # on Wikipedia since ravel() is done in row-major ('C') order.
        # lp_test = scipy.stats.multivariate_normal.logpdf(
        #     np.ravel(weights), np.ravel(self.M0),
        #     np.kron(covariance_matrix, self.V0))
        # assert np.allclose(lp, lp_test)

        # \log p(\Sigma | \Psi0, \nu0)
        if self.nu0 >= self.out_dim and \
            np.all(np.linalg.eigvalsh(self.Psi0) > 0):
            # TODO: Use JAX versions of the logpdf's
            import scipy.stats
            lp += scipy.stats.invwishart.logpdf(
                covariance_matrix, self.nu0, self.Psi0)
        return lp

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Convert the statistics and counts back into NIW parameters.
        Recall,

        ..math::
            n_1 = \nu_0 + n + p + 1
            s_1 = V_0^{-1}
            s_2 = M_0 V_0^{-1}
            s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top
        """
        s_1, s_2, s_3 = stats
        out_dim, in_dim = s_2.shape[-2:]

        nu0 = counts - out_dim - in_dim - 1
        if np.allclose(s_1, 0):
            V0 = 1e16 * np.eye(in_dim)
            M0 = np.zeros_like(s_2)
            Psi0 = np.eye(out_dim)
        else:
            # TODO: Use Cholesky factorization for these two steps
            V0 = np.linalg.inv(s_1 + 1e-16 * np.eye(in_dim))
            M0 = s_2 @ V0
            Psi0 = s_3 - M0 @ s_1 @ M0.T

        # assert np.all(np.isfinite(M0))
        # assert np.all(np.isfinite(V0))
        # assert np.all(np.isfinite(nu0))
        # assert np.all(np.isfinite(Psi0))
        return cls(M0, V0, nu0, Psi0, dimension=(out_dim, in_dim))

    @property
    def pseudo_obs(self):
        V0iM0T = np.linalg.solve(self.V0, self.M0.T)
        return (np.linalg.inv(self.V0),
                V0iM0T.T,
                self.Psi0 + self.M0 @ V0iM0T)

    @property
    def pseudo_counts(self):
        return self.nu0 + self.out_dim + self.in_dim + 1

    @property
    def mode(self):
        """Solve for the mode. Recall,
        .. math::
            p(A, \Sigma) \propto
                \mathrm{N}(\vec(A) | \vec(M_0), \Sigma \kron V_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)

        The optimal mean is :math:`A^* = M_0`. Substituting this in,
        .. math::
            p(A^*, \Sigma) \propto IW(\Sigma | \nu_0 + p, \Psi_0)

        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + p + n + 1)
        """
        return dict(combined_weights=self.M0,
                    covariance_matrix=self.Psi0 / (self.nu0 + self.in_dim + self.out_dim + 1))


class MultivariateNormal(ExponentialFamilyDistribution):
    """A multivariate normal distribution that implements the
    ExponentialFamilyDistribution interface.
    """
    def __init__(self, mean, covariance_matrix, prior=None):
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        super(MultivariateNormal, self).__init__(prior=prior)

    @property
    def dimension(self):
        return self.mean.shape[0]

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(mean=np.zeros(dim),
                   covariance_matrix=np.eye(dim),
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self.mean, self._covariance_tril

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self.mean, self._covariance_tril = value

    @property
    def covariance_matrix(self):
        return self._covariance_tril @ self._covariance_tril.T

    @covariance_matrix.setter
    def covariance_matrix(self, value):
        assert np.all(np.linalg.eigvalsh(value) > 0)
        self._covariance_tril = np.linalg.cholesky(value)

    @property
    def dim(self):
        return self.mean.shape[0]

    def log_prior(self):
        return self.prior.log_prob((self.mean, self.covariance_matrix))

    def sample(self, size=(), **kwargs):
        """Return a sample from the distribution.
        """
        return npr.multivariate_normal(
            self.mean, self.covariance_matrix, size=size)

    def log_prob(self, data, **kwargs):
        return spst.multivariate_normal.logpdf(
            data, self.mean, self.covariance_matrix)

    def sufficient_statistics(self, data, **kwargs):
        n = data.shape[0]
        return np.ones(n), data, np.einsum('ni,nj->nij', data, data)


class MultivariateStudentsT(CompoundDistribution):
    """A multivariate Student's T distribution with the CompoundDistribution
    interface for fitting with EM.
    """
    def __init__(self, mean, covariance_matrix, dof, prior=None):
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.dof = dof
        super(MultivariateStudentsT, self).__init__(prior=prior)

    @property
    def dimension(self):
        return self.mean.shape[0]

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(mean=np.zeros(dim),
                   covariance_matrix=np.eye(dim),
                   dof=dim+2,
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self.mean, self._log_variance, self._log_dof

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self.mean, self._log_variance, self._log_dof = value

    @property
    def unconstrained_nonconj_params(self):
        return self._log_dof

    @unconstrained_nonconj_params.setter
    def unconstrained_nonconj_params(self, value):
        self._log_dof = value

    @property
    def covariance_matrix(self):
        return self._covariance_tril @ self._covariance_tril.T

    @covariance_matrix.setter
    def covariance_matrix(self, value):
        self._covariance_tril = np.linalg.cholesky(value)

    @property
    def dof(self):
        return np.exp(self._log_dof)

    @dof.setter
    def dof(self, value):
        self._log_dof = np.log(value)

    def log_prior(self):
        return self.prior.log_prob((self.mean, self.covariance_matrix))

    def log_prob(self, data, **kwargs):
        mean, chol, dof, dim = \
            self.mean, self._covariance_tril, self.dof, self.dimension
        assert data.ndim == 2 and data.shape[1] == dim

        # Quadratic term
        tmp = np.linalg.solve(chol, (data - mean).T).T
        lp = - 0.5 * (dof + dim) * np.log1p(np.sum(tmp**2, axis=1) / dof)

        # Normalizer
        lp += spsp.gammaln(0.5 * (dof + dim)) - spsp.gammaln(0.5 * dof)
        lp += - 0.5 * dim * np.log(np.pi) - 0.5 * dim * np.log(dof)
        # L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]
        lp += -np.sum(np.log(np.diag(chol)))
        return lp

    def conditional_expectations(self, data, **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.`
        """
        # The auxiliary precision \tau is conditionally gamma distributed.
        alpha = 0.5 * (self.dof + self.dimension)
        tmp = np.linalg.solve(self._covariance_tril, (data - self.mean).T).T
        beta = 0.5 * (self.dof + np.sum(tmp**2, axis=1))

        # Compute gamma expectations
        E_tau = alpha / beta
        E_log_tau = spsp.digamma(alpha) - np.log(beta)
        return E_tau, E_log_tau

    def expected_sufficient_statistics(self, expectations, data, **kwargs):
        """Given the precision, the data is conditionally Gaussian.
        """
        E_tau, E_log_tau = expectations
        return (E_tau,
                np.einsum('n,ni->ni', E_tau, data),
                np.einsum('n,ni,nj->nij', E_tau, data, data))

    def expected_log_prob(self, expectations, data, regularization=1e-3, **kwargs):
        """Compute the expected log probability.  This function will be
        optimized with respect to the remaining, non-conjugate parameters
        of the distribution.
        """
        E_tau, E_log_tau = expectations
        hdof = 0.5 * self.dof
        lp = -np.sum(np.log(np.diag(self._covariance_tril)))
        lp += 0.5 * E_log_tau
        # The quadratic term is in expectation zero at the optimal params,
        # and it doesn't depend on the dof, so we drop it.
        # tmp = np.linalg.solve(self._covariance_tril, (data - self.mean).T).T
        # lp -= 0.5 * E_tau * np.sum(tmp**2, axis=1)
        lp += hdof * np.log(hdof)
        lp -= spsp.gammaln(hdof)
        lp += (hdof - 1) * E_log_tau
        lp -= hdof * E_tau
        lp -= regularization * hdof
        return lp

class Normal(ExponentialFamilyDistribution):
    """Scalar normal (i.e. Gaussian) distribution.
    """
    def __init__(self, mean, variance=1.0, prior=None):
        self.mean = mean
        self.variance = variance
        super(Normal, self).__init__(prior=prior)

    @property
    def dimension(self):
        return 1

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(mean=np.zeros(dim),
                   variance=np.ones(dim),
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self.mean, self._log_variance

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self.mean, self._log_variance = value

    @property
    def variance(self):
        return np.exp(self._log_variance)

    @variance.setter
    def variance(self, value):
        self._log_variance = np.log(value)

    def log_prob(self, data, **kwargs):
        return spst.norm.logpdf(data, self.mean, np.sqrt(self.variance))

    def sufficient_statistics(self, data, **kwargs):
        return np.ones_like(data), data, data**2


class NormalInverseGamma(ConjugatePrior):
    """A conjugate prior distribution for the normal distribution.
    ..math::
    \mu | \sigma^2 \sim \mathrm{N}(\mu | \mu_0, \sigma^2 / \kappa_0)
          \sigma^2 \sim \mathrm{IGa}(\sigma^2 | \alpha_0, \beta_0)

    and the log pdf is proportional to,

    ..math::
    \log p(\mu, \sigma^2) =
        -(\alpha_0 + 1 + 0.5) \log \sigma^2
        + \kappa_0 * (-0.5 \mu^2 / \sigma^2)
        + \kappa_0 \mu_0 \mu / \sigma^2
        + (2 * \beta_0 + \kappa_0 \mu_0^2) * (-0.5 / \sigma^2)

    The natural parameters of the normal distribution are,
    .. math::
    \eta_1 = -0.5 \mu^2 / \sigma^2
    \eta_2 = \mu / \sigma^2,
    \eta_3 = -0.5 / \sigma^2,

    and they correspond to the sufficient statistics of the normal likelihood,
    .. math::
        t(x)_1 = 1
        t(x)_2 = x,
        t(x)_3 = x^2

    This looks a bit unconventional in that the first sufficient statistic
    is actually not a function of the data.  This parameterization makes
    sense when we write the normal distribution as a linear regression,

    ..math::
        y \mid x=1 \sim N(y \mid \mu x, \sigma^2)

    where :math:`x=1` is a fixed covariate.  In that case, the first sufficient
    statistic becomes :math:`t(x) = x^2 = 1`.  See the
    `MultivariateNormalLinearRegression` class for more detail.

    The NIG prior provides the following statistics and pseudo-counts:
    .. math:
        s_1 = \kappa_0
        s_2 = \kappa_0 \mu_0
        s_3 = 2 * \beta_0 + \kappa_0 \mu_0^2
        n = \alpha_0 + 1.5

    We default to an improper uniform prior with zero pseudo counts"""

    def __init__(self, mu0=0, kappa0=0, alpha0=-1.5, beta0=0):
        # Store the standard parameters
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    @property
    def dimension(self):
        return 1

    # TODO: Reparameterize parameters so that they can be optimized with autograd.
    @property
    def unconstrained_params(self):
        return ()

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        pass

    def log_prior(self):
        return 0

    def log_prob(self, data, **kwargs):
        """Compute the prior log probability of a MultivariateNormal
        distribution's parameters under this NIW prior.

        Note that the NIW prior is only properly specified in certain
        parameter regimes (otherwise the density does not normalize).
        Only compute the log prior if there is a density.
        """
        mean, variance = data

        lp = 0
        if self.kappa0 > 0:
            lp += np.sum(spst.norm.logpdf(
                mean, self.mu0, np.sqrt(variance / self.kappa0)))

        if self.alpha0 > 0:
            # TODO: Use autograd versions of the logpdf's
            import scipy.stats
            lp += scipy.stats.invgamma.logpdf(
                variance, self.alpha0, scale=self.beta0)
        return lp

    @property
    def mode(self):
        """Solve for the mode. Recall,

        ..math::
            p(\mu, \sigma^2) \propto
                \mathrm{N}(\mu | \mu_0, \sigma^2 / \kappa_0) \times
                \mathrm{IGa}(\Sigma | \alpha_0, \beta_0)

        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        ..math::
            p(\mu^*, \sigma^2) \propto IGa(\Sigma | \alpha_0 + 0.5, \beta_0)

        and the mode of this inverse Wishart distribution is at
        ..math::
            (\sigma^2)* = \beta_0 / (\alpha_0 + 0.5)

        if ..math:`\alpha_0 > 0.5`.  Otherwise, return the mean,
        ..math::
            (\sigma^2)* = \beta_0 / (\alpha + 1.5)
        """
        if self.alpha0 > 0.5:
            return dict(mean=self.mu0,
                        variance=self.beta0 / (self.alpha0 + 0.5))
        else:
            # Return the mean instead of the mode
            return dict(mean=self.mu0,
                        variance=self.beta0 / (self.alpha0 + 1.5))

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Convert the statistics and counts back into NIG parameters.
        .. math:
            s_1 = \kappa_0
            s_2 = \kappa_0 \mu_0
            s_3 = 2 * \beta_0 + \kappa_0 \mu_0^2
            n = \alpha_0 + 1.5
        """
        s_1, s_2, s_3 = stats
        kappa0 = s_1
        mu0 = s_2 / kappa0 if kappa0 > 0 else np.zeros_like(s_1)
        beta0 = 0.5 * (s_3 - kappa0 * mu0**2)
        alpha0 = counts - 1.5
        return cls(mu0, kappa0, alpha0, beta0)

    @property
    def pseudo_obs(self):
        return (self.kappa0,
                self.kappa0 * self.mu0,
                2 * self.beta0 + self.kappa0 * self.mu0**2)

    @property
    def pseudo_counts(self):
        return self.alpha0 + 1.5


class NormalInverseWishart(ConjugatePrior):
    """A conjugate prior distribution for the MultivariateNormal.
    ..math::
    \mu | \Sigma \sim \mathrm{N}(\mu | \mu_0, \Sigma / \kappa_0)
          \Sigma \sim \mathrm{IW}(\Sigma | \Psi_0, \nu_0)

    The natural parameters of the multivariate normal distribution
    are,
    .. math::
    \eta_1 = -1/2 \mu^\top \Sigma^{-1} \mu
    \eta_2 = \Sigma^{-1} \mu,
    \eta_3 = -1/2 \Sigma^{-1},

    and they correspond to the sufficient statistics,
    .. math::
        t(x)_1 = 1
        t(x)_2 = x,
        t(x)_3 = x x^\top,

    This looks a bit unconventional in that the first sufficient statistic
    is actually not a function of the data.  This parameterization makes
    sense when we write the normal distribution as a linear regression,

    ..math::
        y \mid x=1 \sim N(y \mid \mu x, \Sigma)

    where :math:`x=1` is a fixed covariate.  In that case, the first sufficient
    statistic becomes :math:`t(x) = x x^\top = 1`.  See the
    `MultivariateNormalLinearRegression` class for more detail.

    The NIW prior gives the following sufficient statistics and counts:
    .. math:
        s_1 = \kappa_0
        s_2 = \kappa_0 \mu_0
        s_3 = \Psi_0 + \kappa_0 \mu_0 \mu_0^\top
        n = \nu_0 + d + 2

    where :math:`d` is the dimension of the data.

    We default to an improper uniform prior.  Note that this differs from
    the uninformative prior presented in Gelman et al, "Bayesian Data Analysis"
     (page 73), which corresponds to a pseudo-count of :math:`n = d + 1` for
     the covariance matrix."""

    def __init__(self, mu0=0, kappa0=0, nu0=None, Psi0=1e-4, dimension=1):
        self.mu0 = mu0 * np.ones(dimension)
        self.kappa0 = kappa0
        self.nu0 = -(dimension + 2) if nu0 is None else nu0
        self.Psi0 = np.dot(Psi0, np.eye(dimension))

    @property
    def dimension(self):
        return self.mu0.shape[0]

    # TODO: Reparameterize parameters so that they can be optimized with autograd.
    @property
    def unconstrained_params(self):
        return ()

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        pass

    def log_prior(self):
        return 0

    def log_prob(self, data, **kwargs):
        """Compute the prior log probability of a MultivariateNormal
        distribution's parameters under this NIW prior.

        Note that the NIW prior is only properly specified in certain
        parameter regimes (otherwise the density does not normalize).
        Only compute the log prior if there is a density.
        """
        mean, covariance_matrix = data
        assert mean.shape[0] == self.dimension

        lp = 0
        if self.kappa0 > 0:
            lp += np.sum(spst.multivariate_normal.logpdf(
                mean, self.mu0, covariance_matrix / self.kappa0))

        if self.nu0 >= self.dimension:
            # TODO: Use autograd versions of the logpdf's
            import scipy.stats
            lp += scipy.stats.invwishart.logpdf(
                covariance_matrix, self.nu0, self.Psi0)
        return lp

    @property
    def mode(self):
        """Solve for the mode. Recall,

        .. math::
            p(\mu, \Sigma) \propto
                \mathrm{N}(\mu | \mu_0, \Sigma / \kappa_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)

        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        .. math::
            p(\mu^*, \Sigma) \propto IW(\Sigma | \nu_0 + 1, \Psi_0)

        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + d + 2)

        """
        return dict(mean=self.mu0,
                    covariance_matrix=self.Psi0 / (self.nu0 + self.dimension + 2))

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Convert the statistics and counts back into NIW parameters.
        """
        s_1, s_2, s_3 = stats
        dim = np.size(s_2)

        kappa0 = s_1
        mu0 = s_2 / kappa0 if kappa0 > 0 else np.zeros_like(s_2)
        Psi0 = s_3 - kappa0 * np.outer(mu0, mu0)
        nu0 = counts - dim - 2
        return cls(mu0, kappa0, nu0, Psi0, dimension=dim)

    @property
    def pseudo_obs(self):
        return (self.kappa0,
                self.kappa0 * self.mu0,
                self.Psi0 + self.kappa0 * np.outer(self.mu0, self.mu0))

    @property
    def pseudo_counts(self):
        return self.nu0 + self.dimension + 2


class Poisson(ExponentialFamilyDistribution):
    """A Poisson distribution that implements the
    ExponentialFamilyDistribution interface.
    """
    def __init__(self, rate, prior=None):
        self.rate = rate
        super(Poisson, self).__init__(prior=prior)

    @property
    def dimension(self):
        return 1

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(rate=np.ones(dim), **kwargs)

    @property
    def unconstrained_params(self):
        return self._log_rate

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self._log_rate = value

    @property
    def rate(self):
        return np.exp(self._log_rate)

    @rate.setter
    def rate(self, value):
        self._log_rate = np.log(value)

    def sufficient_statistics(self, data, **kwargs):
        return (data,)


class StudentsT(CompoundDistribution):
    """A Student's T distribution with the CompoundDistribution interface
    for fitting with EM.
    """
    def __init__(self, mean, variance, dof, prior=None):
        self.mean = mean
        self.variance = variance
        self.dof = dof
        super(StudentsT, self).__init__(prior=prior)

    @property
    def dimension(self):
        return 1

    @classmethod
    def from_example(cls, data, **kwargs):
        dim = data.shape[-1]
        return cls(mean=np.zeros(dim),
                   variance=np.ones(dim),
                   dof=3,
                   **kwargs)

    @property
    def unconstrained_params(self):
        return self.mean, self._log_variance, self._log_dof

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        self.mean, self._log_variance, self._log_dof = value

    @property
    def unconstrained_nonconj_params(self):
        return self._log_dof

    @unconstrained_nonconj_params.setter
    def unconstrained_nonconj_params(self, value):
        self._log_dof = value

    @property
    def variance(self):
        return np.exp(self._log_variance)

    @variance.setter
    def variance(self, value):
        self._log_variance = np.log(value)

    @property
    def dof(self):
        return np.exp(self._log_dof)

    @dof.setter
    def dof(self, value):
        self._log_dof = np.log(value)

    def log_prior(self):
        return self.prior.log_prob((self.mean, self.variance))

    def log_prob(self, data, **kwargs):
        raise NotImplementedError
        # return spst.normal.logpdf(
        #     data, self.mean, self.variance)

    def conditional_expectations(self, data, **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.  In the Student's t, for example,
        the auxiliary variables are the per-datapoint precision, :math:`\tau`,
        and the necessary expectations are :math:`\mathbb{E}[\tau]` and
        :math:`\mathbb{E}[\log \tau]`
        """
        # The auxiliary precision \tau is conditionally gamma distributed.
        alpha = 0.5 * (self.dof + 1)
        beta = 0.5 * (self.dof + (data - self.mean)**2 / self.variance)

        # Compute gamma expectations
        E_tau += alpha / beta
        E_log_tau += spsp.digamma(alpha) - np.log(beta)
        return E_tau, E_log_tau

    def expected_sufficient_statistics(self, expectations, data, **kwargs):
        """Given the precision, the data is conditionally Gaussian.
        """
        E_tau, E_log_tau = expectations
        return E_tau, E_tau * data, E_tau * data**2

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
