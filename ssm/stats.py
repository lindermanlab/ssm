import autograd.numpy as np
from autograd.scipy.special import gammaln, logsumexp
from autograd.scipy.linalg import solve_triangular

from ssm.util import one_hot


def flatten_to_dim(X, d):
    """
    Flatten an array of dimension k + d into an array of dimension 1 + d.

    Example:
        X = npr.rand(10, 5, 2, 2)
        flatten_to_dim(X, 4).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 3).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 2).shape # (50, 2, 2)
        flatten_to_dim(X, 1).shape # (100, 2)

    Parameters
    ----------
    X : array_like
        The array to be flattened.  Must be at least d dimensional

    d : int (> 0)
        The number of dimensions to retain.  All leading dimensions are flattened.

    Returns
    -------
    flat_X : array_like
        The input X flattened into an array dimension d (if X.ndim == d)
        or d+1 (if X.ndim > d)
    """
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def batch_mahalanobis(L, x):
    """
    Compute the squared Mahalanobis distance.
    :math:`x^T M^{-1} x` for a factored :math:`M = LL^T`.

    Copied from PyTorch torch.distributions.multivariate_normal.

    Parameters
    ----------
    L : array_like (..., D, D)
        Cholesky factorization(s) of covariance matrix

    x : array_like (..., D)
        Points at which to evaluate the quadratic term

    Returns
    -------
    y : array_like (...,)
        squared Mahalanobis distance :math:`x^T (LL^T)^{-1} x`

        x^T (LL^T)^{-1} x = x^T L^{-T} L^{-1} x
    """
    # The most common shapes are x: (T, D) and L : (D, D)
    # Special case that one
    if x.ndim == 2 and L.ndim == 2:
        xs = solve_triangular(L, x.T, lower=True)
        return np.sum(xs**2, axis=0)

    # Flatten the Cholesky into a (-1, D, D) array
    flat_L = flatten_to_dim(L, 2)
    # Invert each of the K arrays and reshape like L
    L_inv = np.reshape(np.array([np.linalg.inv(Li.T) for Li in flat_L]), L.shape)
    # dot with L_inv^T; square and sum.
    xs = np.einsum('...i,...ij->...j', x, L_inv)
    return np.sum(xs**2, axis=-1)

def _multivariate_normal_logpdf(data, mus, Sigmas, Ls=None):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)

    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # Quadratic term
    lp = -0.5 * batch_mahalanobis(Ls, data - mus)                    # (...,)
    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp


def multivariate_normal_logpdf(data, mus, Sigmas, mask=None):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D

    # If there's no mask, we can just use the standard log pdf code
    if mask is None:
        return _multivariate_normal_logpdf(data, mus, Sigmas)

    # Otherwise we need to separate the data into sets with the same mask,
    # since each one will entail a different covariance matrix.
    #
    # First, determine the output shape. Allow mus and Sigmas to
    # have different shapes; e.g. many Gaussians with the same
    # covariance but different means.
    shp1 = np.broadcast(data, mus).shape[:-1]
    shp2 = np.broadcast(data[..., None], Sigmas).shape[:-2]
    assert len(shp1) == len(shp2)
    shp = tuple(max(s1, s2) for s1, s2 in zip(shp1, shp2))

    # Broadcast the data into the full shape
    full_data = np.broadcast_to(data, shp + (D,))

    # Get the full mask
    assert mask.dtype == bool
    assert mask.shape == data.shape
    full_mask = np.broadcast_to(mask, shp + (D,))

    # Flatten the mask and get the unique values
    flat_data = flatten_to_dim(full_data, 1)
    flat_mask = flatten_to_dim(full_mask, 1)
    unique_masks, mask_index = np.unique(flat_mask, return_inverse=True, axis=0)

    # Initialize the output
    lls = np.nan * np.ones(flat_data.shape[0])

    # Compute the log probability for each mask
    for i, this_mask in enumerate(unique_masks):
        this_inds = np.where(mask_index == i)[0]
        this_D = np.sum(this_mask)
        if this_D == 0:
            lls[this_inds] = 0
            continue

        this_data = flat_data[np.ix_(this_inds, this_mask)]
        this_mus = mus[..., this_mask]
        this_Sigmas = Sigmas[np.ix_(*[np.ones(sz, dtype=bool) for sz in Sigmas.shape[:-2]], this_mask, this_mask)]

        # Precompute the Cholesky decomposition
        this_Ls = np.linalg.cholesky(this_Sigmas)

        # Broadcast mus and Sigmas to full shape and extract the necessary indices
        this_mus = flatten_to_dim(np.broadcast_to(this_mus, shp + (this_D,)), 1)[this_inds]
        this_Ls = flatten_to_dim(np.broadcast_to(this_Ls, shp + (this_D, this_D)), 2)[this_inds]

        # Evaluate the log likelihood
        lls[this_inds] = _multivariate_normal_logpdf(this_data, this_mus, this_Sigmas, Ls=this_Ls)

    # Reshape the output
    assert np.all(np.isfinite(lls))
    return np.reshape(lls, shp)


def expected_multivariate_normal_logpdf(E_xs, E_xxTs, E_mus, E_mumuTs, Sigmas, Ls=None):
    """
    Compute the expected log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.
    Parameters
    ----------
    E_xs : array_like (..., D)
        The expected value of the points at which to evaluate the log density
    E_xxTs : array_like (..., D, D)
        The second moment of the points at which to evaluate the log density
    E_mus : array_like (..., D)
        The expected mean(s) of the Gaussian distribution(s)
    E_mumuTs : array_like (..., D, D)
        The second moment of the mean
    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)
    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas
    Returns
    -------
    lps : array_like (...,)
        Expected log probabilities under the multivariate Gaussian distribution(s).
    TODO
    ----
    - Allow for uncertainty in the covariance as well.
    """
    # Check inputs
    D = E_xs.shape[-1]
    assert E_xxTs.shape[-2] == E_xxTs.shape[-1] == D
    assert E_mus.shape[-1] == D
    assert E_mumuTs.shape[-2] == E_mumuTs.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # TODO: Figure out how to perform this computation without explicit inverse
    Sigma_invs = np.linalg.inv(Sigmas)

    # Compute  E[(x-mu)^T Sigma^{-1}(x-mu)]
    #        = Tr(Sigma^{-1} E[(x-mu)(x-mu)^T])
    #        = Tr(Sigma^{-1} E[xx^T - x mu^T - mu x^T + mu mu^T])
    #        = Tr(Sigma^{-1} (E[xx^T - E[x]E[mu]^T - E[mu]E[x]^T + E[mu mu^T]]))
    #        = Tr(Sigma^{-1} A)
    #        = Tr((LL^T)^{-1} A)
    #        = Tr(L^{-1} A L^{-T} )
    #        = sum_{ij} [Sigma^{-1}]_{ij} * A_{ij}
    # where
    #
    # A = E[xx^T - E[x]E[mu]^T - E[mu]E[x]^T + E[mu mu^T]]
    #
    # However, since Sigma^{-1} is symmetric, we get the same
    # answer with
    #
    # A = E[xx^T - 2 * E[x]E[mu]^T + E[mu mu^T]]
    #
    E_xmuT = E_xs[..., :, None] * E_mus[..., None, :]
    # E_muxT = np.swapaxes(E_xmuT, -1, -2)
    # As = E_xxTs - E_xmuT - E_muxT + E_mumuTs
    As = E_xxTs - 2 * E_xmuT + E_mumuTs
    lp = -0.5 * np.sum(Sigma_invs * As, axis=(-2, -1))

    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp


def diagonal_gaussian_logpdf(data, mus, sigmasqs, mask=None):
    """
    Compute the log probability density of a Gaussian distribution with
    a diagonal covariance.  This will broadcast as long as data, mus,
    sigmas have the same (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    sigmasqs : array_like (..., D)
        The diagonal variances(s) of the Gaussian distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the diagonal Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert sigmasqs.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    normalizer = -0.5 * np.log(2 * np.pi * sigmasqs)
    return np.sum((normalizer - 0.5 * (data - mus)**2 / sigmasqs) * mask, axis=-1)


def multivariate_studentst_logpdf(data, mus, Sigmas, nus, Ls=None):
    """
    Compute the log probability density of a multivariate Student's t distribution.
    This will broadcast as long as data, mus, Sigmas, nus have the same (or at
    least be broadcast compatible along the) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the t distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the t distribution(s)

    nus : array_like (...,)
        The degrees of freedom of the t distribution(s)

    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # Quadratic term
    q = batch_mahalanobis(Ls, data - mus) / nus                      # (...,)
    lp = - 0.5 * (nus + D) * np.log1p(q)                             # (...,)

    # Normalizer
    lp = lp + gammaln(0.5 * (nus + D)) - gammaln(0.5 * nus)          # (...,)
    lp = lp - 0.5 * D * np.log(np.pi) - 0.5 * D * np.log(nus)        # (...,)
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - half_log_det

    return lp


def expected_multivariate_studentst_logpdf(E_xs, E_xxTs, E_mus, E_mumuTs, Sigmas, nus, Ls=None):
    """
    Compute the expected log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.
    Parameters
    ----------
    E_xs : array_like (..., D)
        The expected value of the points at which to evaluate the log density
    E_xxTs : array_like (..., D, D)
        The second moment of the points at which to evaluate the log density
    E_mus : array_like (..., D)
        The expected mean(s) of the Gaussian distribution(s)
    E_mumuTs : array_like (..., D, D)
        The second moment of the mean
    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)
    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas
    Returns
    -------
    lps : array_like (...,)
        Expected log probabilities under the multivariate Gaussian distribution(s).
    TODO
    ----
    - Allow for uncertainty in the covariance Sigmas and dof nus as well.
    """
    # Check inputs
    D = E_xs.shape[-1]
    assert E_xxTs.shape[-2] == E_xxTs.shape[-1] == D
    assert E_mus.shape[-1] == D
    assert E_mumuTs.shape[-2] == E_mumuTs.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # TODO: Figure out how to perform this computation without explicit inverse
    Sigma_invs = np.linalg.inv(Sigmas)

    # Compute  E[(x-mu)^T Sigma^{-1}(x-mu)]
    #        = Tr(Sigma^{-1} E[(x-mu)(x-mu)^T])
    #        = Tr(Sigma^{-1} E[xx^T - 2 x mu^T + mu mu^T])
    #        = Tr(Sigma^{-1} (E[xx^T - 2 E[x]E[mu]^T + E[mu mu^T]]))
    #        = Tr(Sigma^{-1} A)
    #        = Tr((LL^T)^{-1} A)
    #        = Tr(L^{-1} A L^{-T} )
    #        = sum_{ij} [Sigma^{-1}]_{ij} * A_{ij}
    # where
    #
    # A = E[xx^T - 2 E[x]E[mu]^T + E[mu mu^T]]
    #
    As = E_xxTs - 2 * E_xs[..., :, None] * E_mus[..., None, :] + E_mumuTs   # (..., D, D)
    q = np.sum(Sigma_invs * As, axis=(-2, -1)) / nus                        # (...,)
    lp = - 0.5 * (nus + D) * np.log1p(q)                                    # (...,)

    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]            # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)                     # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det                    # (...,)

    return lp


def independent_studentst_logpdf(data, mus, sigmasqs, nus, mask=None):
    """
    Compute the log probability density of a set of independent Student's t 
    random variables. This will broadcast as long as data, mus, nus, and
    sigmas have the same (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Student's t distribution(s)

    sigmasqs : array_like (..., D)
        The diagonal variances(s) of the Student's t distribution(s)

    nus : array_like (..., D)
        The degrees of freedom of the Student's t distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Student's t distribution(s).
    """
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert sigmasqs.shape[-1] == D
    assert nus.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    normalizer = gammaln(0.5 * (nus + 1)) - gammaln(0.5 * nus)
    normalizer = normalizer - 0.5 * (np.log(np.pi) + np.log(nus) + np.log(sigmasqs))
    ll = normalizer - 0.5 * (nus + 1) * np.log(1.0 + (data - mus)**2 / (sigmasqs * nus))
    return np.sum(ll * mask, axis=-1)


def bernoulli_logpdf(data, logit_ps, mask=None):
    """
    Compute the log probability density of a Bernoulli distribution.
    This will broadcast as long as data and logit_ps have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    logit_ps : array_like (..., D)
        The logit(s) log p / (1 - p) of the Bernoulli distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Bernoulli distribution(s).
    """
    D = data.shape[-1]
    assert (data.dtype == int or data.dtype == bool)
    assert data.min() >= 0 and data.max() <= 1
    assert logit_ps.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    # Evaluate log probability
    # log Pr(x | p) = x * log(p) + (1-x) * log(1-p)
    #               = x * log(p / (1-p)) + log(1-p)
    #               = x * log(p / (1-p)) - log(1/(1-p))
    #               = x * log(p / (1-p)) - log(1 + p/(1-p)).
    #
    # Let u = log (p / (1-p)) = logit(p), then
    #
    # log Pr(x | p) = x * u - log(1 + e^u)
    #               = x * u - log(e^0 + e^u)
    #               = x * u - log(e^m * (e^-m + e^(u-m))
    #               = x * u - m - log(exp(-m) + exp(u-m)).
    #
    # This holds for any m. we choose m = max(0, u) to avoid overflow.
    m = np.maximum(0, logit_ps)
    lls = data * logit_ps - m - np.log(np.exp(-m) + np.exp(logit_ps - m))
    return np.sum(lls * mask, axis=-1)


def poisson_logpdf(data, lambdas, mask=None):
    """
    Compute the log probability density of a Poisson distribution.
    This will broadcast as long as data and lambdas have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    lambdas : array_like (..., D)
        The rates of the Poisson distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Poisson distribution(s).
    """
    D = data.shape[-1]
    assert data.dtype in (int, np.int8, np.int16, np.int32, np.int64)
    assert lambdas.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    # Compute log pdf
    lls = -gammaln(data + 1) - lambdas + data * np.log(lambdas)
    return np.sum(lls * mask, axis=-1)


def categorical_logpdf(data, logits, mask=None):
    """
    Compute the log probability density of a categorical distribution.
    This will broadcast as long as data and logits have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D) int (0 <= data < C)
        The points at which to evaluate the log density

    lambdas : array_like (..., D, C)
        The logits of the categorical distribution(s) with C classes

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the categorical distribution(s).
    """
    D = data.shape[-1]
    C = logits.shape[-1]
    assert data.dtype in (int, np.int8, np.int16, np.int32, np.int64)
    assert np.all((data >= 0) & (data < C))
    assert logits.shape[-2] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    logits = logits - logsumexp(logits, axis=-1, keepdims=True)      # (..., D, C)
    x = one_hot(data, C)                                             # (..., D, C)
    lls = np.sum(x * logits, axis=-1)                                # (..., D)
    return np.sum(lls * mask, axis=-1)                               # (...,)


def vonmises_logpdf(data, mus, kappas, mask=None):
    """
    Compute the log probability density of a von Mises distribution.
    This will broadcast as long as data, mus, and kappas have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The means of the von Mises distribution(s)

    kappas : array_like (..., D)
        The concentration of the von Mises distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the von Mises distribution(s).
    """
    try:
        from autograd.scipy.special import i0
    except:
        raise Exception("von Mises relies on the function autograd.scipy.special.i0. "
                        "This is present in the latest Github code, but not on pypi. "
                        "Please use the Github version of autograd instead.")

    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert kappas.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    ll = kappas * np.cos(data - mus) - np.log(2 * np.pi) - np.log(i0(kappas))
    return np.sum(ll * mask, axis=-1)


def exponential_logpdf(data, lambdas, mask=None):
    """
    Compute the log probability density of an exponential distribution.
    This will broadcast as long as data and lambdas have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    lambdas : array_like (..., D)
        The rates of the Poisson distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Poisson distribution(s).
    """
    D = data.shape[-1]
    assert lambdas.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    # Compute log pdf
    lls = np.log(lambdas) - lambdas * data
    return np.sum(lls * mask, axis=-1)
