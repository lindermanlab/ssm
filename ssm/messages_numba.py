import numba
import numpy as np
import numpy.random as npr
import scipy.special as scsp

@numba.jit(nopython=True, cache=True)
def logsumexp(x):
    N = x.shape[0]

    # find the max
    m = -np.inf
    for i in range(N):
        m = max(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += np.exp(x[i] - m)

    return m + np.log(out)

@numba.jit(nopython=True, cache=True)
def forward_pass(pi0,
                 Ps,
                 log_likes,
                 alphas):

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    alphas[0] = np.log(pi0) + log_likes[0]
    for t in range(T-1):
        m = np.max(alphas[t])
        alphas[t+1] = np.log(np.dot(np.exp(alphas[t] - m), Ps[t * hetero])) + m + log_likes[t+1]
    return logsumexp(alphas[T-1])


@numba.jit(nopython=True, cache=True)
def backward_pass(log_Ps,
                  log_likes,
                  betas):

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert log_Ps.shape[0] == T-1 or log_Ps.shape[0] == 1
    assert log_Ps.shape[1] == K
    assert log_Ps.shape[2] == K
    assert betas.shape[0] == T
    assert betas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (log_Ps.shape[0] == T-1)
    tmp = np.zeros(K)

    # Initialize the last output
    betas[T-1] = 0

    for t in range(T-2,-1,-1):
        # betal[t] = logsumexp(Al + betal[t+1] + aBl[t+1],axis=1)
        for k in range(K):
            for j in range(K):
                tmp[j] = log_Ps[t * hetero, k, j] + betas[t+1, j] + log_likes[t+1, j]
            betas[t, k] = logsumexp(tmp)


@numba.jit(nopython=True, cache=True)
def _condition_on(m, S, C, D, R, u, y, mcond, Scond):
    # Condition a Gaussian potential on a new linear Gaussian observation
    #
    # The unnormalized potential is
    #
    #   p(x) \propto N(x | m, S) * N(y | Cx + Du, R)
    #
    #        \propto \exp{-1/2 (x-m)^T S^{-1} (x-m) - 1/2 (y - Du - Cx)^T R^{-1} (y - Du - Cx)}
    #        \propto \exp{-1/2 x^T [S^{-1} + C^T R^{-1} C] x
    #                        + x^T [S^{-1} m + C^T R^{-1} (y - Du)]}
    #
    #  => p(x) = N(m', S')  where
    #
    #     S' = [S^{-1} + C^T R^{-1} C]^{-1}
    #     m' = S' [S^{-1} m + C^T R^{-1} (y - Du)]
    #
    #  Now use the matrix inversion lemma
    #
    #     S' = S - K C S
    #
    #  where K = S C^T (R + C S C^T)^{-1}
    K = np.linalg.solve(R + C @ S @ C.T, C @ S).T
    Scond[:] = S - K @ C @ S
    # Scond[:] = S - np.linalg.solve(R + C @ S @ C.T, C @ S).T @ C @ S
    mcond[:] = Scond @ (np.linalg.solve(S, m) + C.T @ np.linalg.solve(R, y - D @ u))
    # return mcond, Scond


@numba.jit(nopython=True, cache=True)
def _condition_on_diagonal(m, S, C, D, R_diag, u, y):
    # Same as above but where R is assumed to be a diagonal covariance matrix
    raise NotImplementedError


@numba.jit(nopython=True, cache=True)
def _predict(m, S, A, B, Q, u, mpred, Spred):
    # Predict next mean and covariance under a linear Gaussian model
    #
    #   p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | Ax_t + Bu, Q)
    #              = N(x_{t+1} | Am + Bu, A S A^T + Q)
    mpred[:] = A @ m + B @ u
    Spred[:] = A @ S @ A.T + Q

@numba.jit(nopython=True, cache=True)
def _sample_gaussian(m, S, z):
    # Sample a multivariate Gaussian with mean m, covariance S,
    # using a standard normal vector z. Put the output in out.
    L = np.linalg.cholesky(S)
    return m + L @ z

@numba.jit(nopython=True, cache=True)
def kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    """
    Standard Kalman filter for time-varying linear dynamical system with inputs.

    Notation:

    T:  number of time steps
    D:  continuous latent state dimension
    U:  input dimension
    N:  observed data dimension

    mu0: (D,)       initial state mean
    S0:  (D, D)     initial state covariance
    As:  (T, D, D)  dynamics matrices
    Bs:  (T, D, U)  input to latent state matrices
    Qs:  (T, D, D)  dynamics covariance matrices
    Cs:  (T, N, D)  emission matrices
    Ds:  (T, N, U)  input to emissions matrices
    Rs:  (T, N, N)  emission covariance matrices
    us:  (T, U)     inputs
    ys:  (T, N)     observations
    """
    T, N = ys.shape
    D = mu0.shape[0]

    predicted_mus = np.zeros((T, D))        # preds E[x_t | y_{1:t-1}]
    predicted_Sigmas = np.zeros((T, D, D))  # preds Cov[x_t | y_{1:t-1}]
    filtered_mus = np.zeros((T, D))         # means E[x_t | y_{1:t}]
    filtered_Sigmas = np.zeros((T, D, D))   # means Cov[x_t | y_{1:t}]

    # Initialize
    predicted_mus[0] = mu0
    predicted_Sigmas[0] = S0
    K = np.zeros((D, N))

    # Run the Kalman filter
    for t in range(T):
        # filtered_mus[t], filtered_Sigmas[t] = \
        _condition_on(predicted_mus[t], predicted_Sigmas[t],
            Cs[t], Ds[t], Rs[t], us[t], ys[t],
            filtered_mus[t], filtered_Sigmas[t])

        if t == T-1:
            break

        # predicted_mus[t+1], predicted_Sigmas[t+1] = \
        _predict(filtered_mus[t], filtered_Sigmas[t],
            As[t], Bs[t], Qs[t], us[t],
            predicted_mus[t+1], predicted_Sigmas[t+1])

    return filtered_mus, filtered_Sigmas


@numba.jit(nopython=True, cache=True)
def kalman_sample(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    """
    Sample from a linear Gaussian model.  Run the KF to get
    the filtered probability distributions, then sample
    backward in time.

    Notation:

    T:  number of time steps
    D:  continuous latent state dimension
    U:  input dimension
    N:  observed data dimension

    mu0: (D,)       initial state mean
    S0:  (D, D)     initial state covariance
    As:  (T, D, D)  dynamics matrices
    Bs:  (T, D, U)  input to latent state matrices
    Qs:  (T, D, D)  dynamics covariance matrices
    Cs:  (T, N, D)  emission matrices
    Ds:  (T, N, U)  input to emissions matrices
    Rs:  (T, N, N)  emission covariance matrices
    us:  (T, U)     inputs
    ys:  (T, N)     observations
    """
    T, N = ys.shape
    D = mu0.shape[0]

    # Run the Kalman Filter
    filtered_mus, filtered_Sigmas = \
        kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

    # Initialize outputs, noise, and temporary variables
    xs = np.zeros((T, D))
    noise = np.random.randn(T, D)
    mu_cond = np.zeros(D)
    Sigma_cond = np.zeros((D, D))

    # Sample backward in time
    xs[-1] = _sample_gaussian(filtered_mus[-1], filtered_Sigmas[-1], noise[-1])
    for t in range(T-2, -1, -1):
        _condition_on(filtered_mus[t], filtered_Sigmas[t],
            As[t], Bs[t], Qs[t], us[t], xs[t+1],
            mu_cond, Sigma_cond)

        xs[t] = _sample_gaussian(mu_cond, Sigma_cond, noise[t])

    return xs

@numba.jit(nopython=True, cache=True)
def kalman_smoother(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    """
    Compute the conditional mean and variance of the latent
    states given observed data ys and inputs us.  Run the KF to get
    the filtered probability distributions, then run the Rauch-Tung-Striebel
    smoother backward in time.

    Notation:

    T:  number of time steps
    D:  continuous latent state dimension
    U:  input dimension
    N:  observed data dimension

    Parameters:

    mu0: (D,)       initial state mean
    S0:  (D, D)     initial state covariance
    As:  (T, D, D)  dynamics matrices
    Bs:  (T, D, U)  input to latent state matrices
    Qs:  (T, D, D)  dynamics covariance matrices
    Cs:  (T, N, D)  emission matrices
    Ds:  (T, N, U)  input to emissions matrices
    Rs:  (T, N, N)  emission covariance matrices
    us:  (T, U)     inputs
    ys:  (T, N)     observations

    Return:

    smoothed_mus:         (T, D)          # posterior marginal mean
    smoothed_Sigmas:      (T, D, D)       # posterior marginal covariance
    smoothed_CrossSigmas: (T-1, D, D)     # posterior marginal cross-covariance Cov(x_t, x_{t+1})
    """
    T, N = ys.shape
    D = mu0.shape[0]

    # Run the Kalman Filter
    filtered_mus, filtered_Sigmas = \
        kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

    # Initialize outputs, noise, and temporary variables
    smoothed_mus = np.zeros((T, D))
    smoothed_Sigmas = np.zeros((T, D, D))
    smoothed_CrossSigmas = np.zeros((T-1, D, D))
    Gt = np.zeros((D, D))

    # The last time step is known from the Kalman filter
    smoothed_mus[-1] = filtered_mus[-1]
    smoothed_Sigmas[-1] = filtered_Sigmas[-1]

    # Run the smoother backward in time
    for t in range(T-2, -1, -1):
        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        Gt = np.linalg.solve(Qs[t] + As[t] @ filtered_Sigmas[t] @ As[t].T,
                             As[t] @ filtered_Sigmas[t]).T

        smoothed_mus[t] = filtered_mus[t] + Gt @ (smoothed_mus[t+1] - As[t] @ filtered_mus[t] - Bs[t] @ us[t])
        smoothed_Sigmas[t] = filtered_Sigmas[t] + \
                             Gt @ (smoothed_Sigmas[t+1] - As[t] @ filtered_Sigmas[t] @ As[t].T - Qs[t]) @ Gt.T
        smoothed_CrossSigmas[t] = Gt @ smoothed_Sigmas[t+1]

    return smoothed_mus, smoothed_Sigmas, smoothed_CrossSigmas

## Test
def test_hmm():

    def forward_pass_np(log_pi0, log_Ps, log_likes):
        T, K = log_likes.shape
        alphas = []
        alphas.append(log_likes[0] + log_pi0)
        for t in range(T-1):
            anext = scsp.logsumexp(alphas[t] + log_Ps[t].T, axis=1)
            anext += log_likes[t+1]
            alphas.append(anext)
        return np.array(alphas)

    def make_parameters(T, K):
        log_pi0 = -np.log(K) * np.ones(K)
        As = npr.rand(T-1, K, K)
        As /= As.sum(axis=2, keepdims=True)
        log_Ps = np.log(As)
        ll = npr.randn(T, K)
        return log_pi0, log_Ps, ll


    T = 1000
    K = 3

    # Test forward pass
    log_pi0, log_Ps, ll = make_parameters(T, K)
    a1 = forward_pass_np(log_pi0, log_Ps, ll)
    a2 = np.zeros((T, K))
    forward_pass(-np.log(K) * np.ones(K), log_Ps, ll, a2)
    assert np.allclose(a1, a2)

    # Test backward pass
    from pyhsmm.internals.hmm_messages_interface import messages_backwards_log

    # Make parameters
    log_pi0 = -np.log(K) * np.ones(K)
    As = npr.rand(K, K)
    As /= As.sum(axis=-1, keepdims=True)
    log_Ps = np.log(np.repeat(As[None, :, :], T-1, axis=0))
    ll = npr.randn(T, K)

    # Use pyhsmm to compute
    true_betas = np.zeros((T, K))
    messages_backwards_log(As, ll, true_betas)

    # Use ssm to compute
    test_betas = np.zeros((T, K))
    backward_pass(log_Ps, ll, test_betas)

    assert np.allclose(true_betas, test_betas)


def test_lds():
    T = 1000
    D = 2
    N = 10
    m0 = np.zeros(D)
    S0 = np.eye(D)
    As = np.tile(0.99 * np.eye(D)[None, :, :], (T, 1, 1))
    Bs = np.zeros((T, D, 1))
    Qs = np.tile(0.01 * np.eye(D)[None, :, :], (T, 1, 1))
    Cs = np.tile(npr.randn(1, N, D), (T, 1, 1))
    Ds = np.zeros((T, N, 1))
    Rs = np.tile(0.01 * np.eye(N)[None, :, :], (T, 1, 1))
    us = np.ones((T, 1))
    ys = np.sin(2 * np.pi * np.arange(T) / 50)[:, None] * npr.randn(1, 10) + 0.1 * npr.randn(T, N)

    # Filter
    from pylds.lds_messages_interface import kalman_filter as kf
    ll, filtered_mus1, filtered_Sigmas1 = kf(m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)
    filtered_mus2, filtered_Sigmas2 = kalman_filter(m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)
    assert np.allclose(filtered_mus1, filtered_mus2)
    assert np.allclose(filtered_Sigmas1, filtered_Sigmas2)

    # Sample
    xs = kalman_sample(m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

    # Smooth
    from pylds.lds_messages_interface import E_step as ks
    ll, smoothed_mus1, smoothed_Sigmas1, ExnxT1 = ks(m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)
    smoothed_mus2, smoothed_Sigmas2, smoothed_CrossSigmas = kalman_smoother(m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)
    ExxnT2 = smoothed_CrossSigmas + smoothed_mus2[:-1][:, :, None] * smoothed_mus2[1:, None, :]
    ExnxT2 = np.swapaxes(ExxnT2, 1, 2)
    assert np.allclose(smoothed_mus1, smoothed_mus2)
    assert np.allclose(smoothed_Sigmas1, smoothed_Sigmas2)
    assert np.allclose(ExnxT1, ExnxT2)


if __name__ == "__main__":
    test_hmm()
    test_lds()
