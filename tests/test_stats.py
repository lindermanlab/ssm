from itertools import product

from nose.tools import nottest

import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.stats import norm, t, bernoulli, poisson, vonmises
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp

from ssm.stats import multivariate_normal_logpdf, \
    expected_multivariate_normal_logpdf, diagonal_gaussian_logpdf, \
    independent_studentst_logpdf, bernoulli_logpdf, categorical_logpdf, \
    poisson_logpdf, vonmises_logpdf


def test_multivariate_normal_logpdf_simple(D=10):
    # Test single datapoint log pdf
    x = npr.randn(D)
    mu = npr.randn(D)
    L = npr.randn(D, D)
    Sigma = np.dot(L, L.T)

    ll1 = multivariate_normal_logpdf(x, mu, Sigma)
    ll2 = mvn.logpdf(x, mu, Sigma)
    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_simple_masked(D=10):
    # Test single datapoint log pdf with mask
    x = npr.randn(D)
    mask = npr.rand(D) < 0.5
    mask[0] = True
    mu = npr.randn(D)
    L = npr.randn(D, D)
    Sigma = np.dot(L, L.T)

    ll1 = multivariate_normal_logpdf(x, mu, Sigma, mask=mask)
    ll2 = mvn.logpdf(x[mask], mu[mask], Sigma[np.ix_(mask, mask)])
    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_shared_params(D=10):
    # Test broadcasting over datapoints with shared parameters
    leading_ndim = npr.randint(1, 4)
    shp = npr.randint(1, 10, size=leading_ndim)
    x = npr.randn(*shp, D)
    mu = npr.randn(D)
    L = npr.randn(D, D)
    Sigma = np.dot(L, L.T)

    ll1 = multivariate_normal_logpdf(x, mu, Sigma)
    ll2 = np.reshape(mvn.logpdf(x, mu, Sigma), shp)
    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_unique_params(D=10):
    # Test broadcasting over datapoints and corresponding parameters
    leading_ndim = npr.randint(1, 4)
    shp = npr.randint(1, 10, size=leading_ndim)
    x = npr.randn(*shp, D)
    mu = npr.randn(*shp, D)
    L = npr.randn(*shp, D, D) + 1e-8 * np.eye(D)
    Sigma = np.matmul(L, np.swapaxes(L, -1, -2))

    ll1 = multivariate_normal_logpdf(x, mu, Sigma)
    ll2 = np.empty(shp)
    for inds in product(*[np.arange(s) for s in shp]):
        ll2[inds] = mvn.logpdf(x[inds], mu[inds], Sigma[inds])
    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_batches_and_states(D=10):
    # Test broadcasting over B batches, N datapoints, and K parameters
    B = 3
    N = 100
    K = 5
    x = npr.randn(B, N, D)
    mu = npr.randn(K, D)
    L = npr.randn(K, D, D)
    Sigma = np.matmul(L, np.swapaxes(L, -1, -2))

    ll1 = multivariate_normal_logpdf(x[:, :, None, :], mu, Sigma)
    assert ll1.shape == (B, N, K)

    ll2 = np.empty((B, N, K))
    for b in range(B):
        for n in range(N):
            for k in range(K):
                ll2[b, n, k] = mvn.logpdf(x[b, n], mu[k], Sigma[k])
    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_batches_and_states_shared_cov(D=10):
    # Test broadcasting over B batches, N datapoints, and K means, 1 covariance
    B = 3
    N = 100
    K = 5
    x = npr.randn(B, N, D)
    mu = npr.randn(K, D)
    L = npr.randn(D, D)
    Sigma = np.dot(L, L.T)

    ll1 = multivariate_normal_logpdf(x[:, :, None, :], mu, Sigma)
    assert ll1.shape == (B, N, K)

    ll2 = np.empty((B, N, K))
    for b in range(B):
        for n in range(N):
            for k in range(K):
                ll2[b, n, k] = mvn.logpdf(x[b, n], mu[k], Sigma)
    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_batches_and_states_masked(D=10):
    # Test broadcasting over B batches, N datapoints, and K parameters with masks
    B = 3
    N = 100
    K = 5
    x = npr.randn(B, N, D)
    mask = npr.rand(B, N, D) < .5
    mu = npr.randn(K, D)
    L = npr.randn(K, D, D)
    Sigma = np.matmul(L, np.swapaxes(L, -1, -2))

    ll1 = multivariate_normal_logpdf(x[:, :, None, :], mu, Sigma, mask=mask[:, :, None, :])
    assert ll1.shape == (B, N, K)

    ll2 = np.empty((B, N, K))
    for b in range(B):
        for n in range(N):
            m = mask[b, n]
            if m.sum() == 0:
                ll2[b, n] = 0
            else:
                for k in range(K):
                    ll2[b, n, k] = mvn.logpdf(x[b, n][m], mu[k][m], Sigma[k][np.ix_(m, m)])

    assert np.allclose(ll1, ll2)


def test_multivariate_normal_logpdf_batches_and_states_shared_cov_masked(D=10):
    # Test broadcasting over B batches, N datapoints, and K means, 1 covariance, with masks
    B = 3
    N = 100
    K = 5
    x = npr.randn(B, N, D)
    mask = npr.rand(B, N, D) < .5
    mu = npr.randn(K, D)
    L = npr.randn(D, D)
    Sigma = np.dot(L, L.T)

    ll1 = multivariate_normal_logpdf(x[:, :, None, :], mu, Sigma, mask=mask[:, :, None, :])
    assert ll1.shape == (B, N, K)

    ll2 = np.empty((B, N, K))
    for b in range(B):
        for n in range(N):
            m = mask[b, n]
            if m.sum() == 0:
                ll2[b, n] = 0
            else:
                for k in range(K):
                    ll2[b, n, k] = mvn.logpdf(x[b, n][m], mu[k][m], Sigma[np.ix_(m, m)])

    assert np.allclose(ll1, ll2)


def test_expected_multivariate_normal_logpdf_simple(D=10):
    # Test single datapoint log pdf
    x = npr.randn(D)
    mu = npr.randn(D)
    L = npr.randn(D, D)
    Sigma = np.dot(L, L.T)

    # Check that when the covariance is zero we get the regular mvn pdf
    xxT = np.outer(x, x)
    mumuT = np.outer(mu, mu)
    ll1 = expected_multivariate_normal_logpdf(x, xxT, mu, mumuT, Sigma)
    ll2 = multivariate_normal_logpdf(x, mu, Sigma)
    assert np.allclose(ll1, ll2)


def test_expected_multivariate_normal_logpdf_bound(D=10):
    """
    Make sure the expected likelihood at the mode is less than
    the likelihood at the mode for nonzero Sigma.
    """
    x = npr.randn(D)
    sqrt_x_cov = npr.randn(D, D)
    x_cov = np.dot(sqrt_x_cov, sqrt_x_cov.T)

    mu = x.copy()
    sqrt_Sigma = npr.randn(D, D)
    Sigma = np.dot(sqrt_Sigma, sqrt_Sigma.T)

    # Check that when the covariance is zero we get the regular mvn pdf
    xxT = x_cov + np.outer(x, x)
    mumuT = np.outer(mu, mu)
    ell = expected_multivariate_normal_logpdf(x, xxT, mu, mumuT, Sigma)
    ll = multivariate_normal_logpdf(x, mu, Sigma)
    assert ell < ll


def test_diagonal_gaussian_logpdf(T=100, K=4, D=10):
    # Test single datapoint log pdf
    x = npr.randn(T, D)
    mu = npr.randn(K, D)
    sigmasqs = np.exp(npr.randn(K, D))

    ll1 = diagonal_gaussian_logpdf(x[:, None, :], mu, sigmasqs)
    ll2 = np.sum(norm.logpdf(x[:, None, :], mu[None, :, :], np.sqrt(sigmasqs[None, :, :])), axis=-1)
    assert np.allclose(ll1, ll2)


def test_independent_studentst_logpdf(T=100, K=4, D=10):
    # Test single datapoint log pdf
    x = npr.randn(T, D)
    mu = npr.randn(K, D)
    sigmasqs = np.exp(npr.randn(K, D))
    nus = np.exp(npr.randn(K, D))

    ll1 = independent_studentst_logpdf(x[:, None, :], mu, sigmasqs, nus)
    ll2 = np.sum(t.logpdf(x[:, None, :], nus[None, :, :], loc=mu[None, :, :], scale=np.sqrt(sigmasqs[None, :, :])), axis=-1)
    assert np.allclose(ll1, ll2)


def test_bernoulli_logpdf(T=100, K=4, D=10):
    # Test single datapoint log pdf
    x = npr.rand(T, D) < 0.5
    logit_ps = npr.randn(K, D)
    ps = 1 / (1 + np.exp(-logit_ps))
    ll1 = bernoulli_logpdf(x[:, None, :], logit_ps)
    ll2 = np.sum(bernoulli.logpmf(x[:, None, :], ps[None, :, :]), axis=-1)
    assert np.allclose(ll1, ll2)


def test_categorical_logpdf(T=100, K=4, D=10, C=8):
    # Test single datapoint log pdf
    x = npr.randint(0, C, size=(T, D))
    logits = npr.randn(K, D, C)
    logits -= logsumexp(logits, axis=-1, keepdims=True)
    ps = np.exp(logits)
    log_ps = np.log(ps)

    ll1 = categorical_logpdf(x[:, None, :], logits)
    ll2 = np.zeros((T, K))
    for n in range(T):
        for k in range(K):
            for d in range(D):
                ll2[n, k] += log_ps[k, d, x[n, d]]
    assert np.allclose(ll1, ll2)


def test_poisson_logpdf(T=100, K=4, D=10):
    # Test single datapoint log pdf
    x = npr.poisson(1, size=(T, D))
    lambdas = np.exp(npr.randn(K, D))
    ll1 = poisson_logpdf(x[:, None, :], lambdas)
    ll2 = np.sum(poisson.logpmf(x[:, None, :], lambdas[None, :, :]), axis=-1)
    assert np.allclose(ll1, ll2)


@nottest
def test_vonmises_logpdf(T=100, K=4, D=10):
    """
    NOTE: Skipping this test until the scipy special functions
    make it into the release version of autograd.
    """
    # Test single datapoint log pdf
    x = npr.vonmises(0, 1, size=(T, D))
    mus = npr.randn(K, D)
    kappas = np.exp(npr.randn(K, D))
    ll1 = vonmises_logpdf(x[:, None, :], mus, kappas)
    ll2 = np.sum(vonmises.logpdf(x[:, None, :], kappas[None, :, :], loc=mus[None, :, :]), axis=-1)
    assert np.allclose(ll1, ll2)

