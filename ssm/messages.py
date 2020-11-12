import numba
import numpy as np
import numpy.random as npr
import scipy.special as scsp
from functools import partial

from autograd.tracer import getval
from autograd.extend import primitive, defvjp
from ssm.util import LOG_EPS, DIV_EPS

to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)

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
def dlse(a, out):
    K = a.shape[0]
    lse = logsumexp(a)
    for k in range(K):
        out[k] = np.exp(a[k] - lse)


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
def hmm_filter(pi0, Ps, ll):
    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)

    # Check if using heterogenous transition matrices
    hetero = (Ps.shape[0] == T-1)

    # Predict forward with the transition matrix
    pz_tt = np.empty((T-1, K))
    pz_tp1t = np.empty((T-1, K))
    for t in range(T-1):
        m = np.max(alphas[t])
        pz_tt[t] = np.exp(alphas[t] - m)
        pz_tt[t] /= np.sum(pz_tt[t])
        pz_tp1t[t] = pz_tt[t].dot(Ps[hetero*t])

    # Include the initial state distribution
    # Numba's version of vstack requires all arrays passed to vstack
    # to have the same number of dimensions.
    pi0 = np.expand_dims(pi0, axis=0)
    pz_tp1t = np.vstack((pi0, pz_tp1t))

    # Numba implementation of np.sum does not allow axis keyword arg,
    # and does not support np.allclose, so we loop over the time range
    # to verify that each sums to 1.
    for t in range(T):
        assert np.abs(np.sum(pz_tp1t[t]) - 1.0) < 1e-8

    return pz_tp1t


@numba.jit(nopython=True, cache=True)
def backward_pass(Ps,
                  log_likes,
                  betas):

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert betas.shape[0] == T
    assert betas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    tmp = np.zeros(K)

    # Initialize the last output
    betas[T-1] = 0
    for t in range(T-2,-1,-1):
        tmp = log_likes[t+1] + betas[t+1]
        m = np.max(tmp)
        betas[t] = np.log(np.dot(Ps[t * hetero], np.exp(tmp - m))) + m



@numba.jit(nopython=True, cache=True)
def _compute_stationary_expected_joints(alphas, betas, lls, log_P, E_zzp1):
    """
    Helper function to compute summary statistics, summing over time.
    NOTE: Can rewrite this in nicer form with Numba.
    """
    T = alphas.shape[0]
    K = alphas.shape[1]
    assert betas.shape[0] == T and betas.shape[1] == K
    assert lls.shape[0] == T and lls.shape[1] == K
    assert log_P.shape[0] == K and log_P.shape[1] == K
    assert E_zzp1.shape[0] == K and E_zzp1.shape[1] == K

    tmp = np.zeros((K, K))

    # Compute the sum over time axis of the expected joints
    for t in range(T-1):
        maxv = -np.inf
        for i in range(K):
            for j in range(K):
                # Compute expectations in this batch
                tmp[i, j] = alphas[t,i] + betas[t+1,j] + lls[t+1,j] + log_P[i, j]
                if tmp[i, j] > maxv:
                    maxv = tmp[i, j]

        # safe exponentiate
        tmpsum = 0.0
        for i in range(K):
            for j in range(K):
                tmp[i, j] = np.exp(tmp[i, j] - maxv)
                tmpsum += tmp[i, j]

        # Add to expected joints
        for i in range(K):
            for j in range(K):
                E_zzp1[i, j] += tmp[i, j] / (tmpsum + DIV_EPS)


def hmm_expected_states(pi0, Ps, ll):
    T, K = ll.shape

    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)
    normalizer = logsumexp(alphas[-1])

    betas = np.zeros((T, K))
    backward_pass(Ps, ll, betas)

    # Compute E[z_t] for t = 1, ..., T
    expected_states = alphas + betas
    expected_states -= scsp.logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)

    # Compute the log transition matrices.
    # Suppress log(0) warnings as they are expected.
    with np.errstate(divide="ignore"):
        log_Ps = np.log(Ps)


    # Compute E[z_t, z_{t+1}] for t = 1, ..., T-1
    # Note that this is an array of size T*K*K, which can be quite large.
    # To be a bit more frugal with memory, first check if the given log_Ps
    # are TxKxK.  If so, instantiate the full expected joints as well, since
    # we will need them for the M-step.  However, if log_Ps is 1xKxK then we
    # know that the transition matrix is stationary, and all we need for the
    # M-step is the sum of the expected joints.
    stationary = (Ps.shape[0] == 1)
    if not stationary:
        expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
        expected_joints -= expected_joints.max((1,2))[:,None, None]
        expected_joints = np.exp(expected_joints)
        expected_joints /= expected_joints.sum((1,2))[:,None,None]

    else:
        # Compute the sum over time axis of the expected joints
        expected_joints = np.zeros((K, K))
        _compute_stationary_expected_joints(alphas, betas, ll, log_Ps[0], expected_joints)
        expected_joints = expected_joints[None, :, :]

    return expected_states, expected_joints, normalizer


@numba.jit(nopython=True, cache=True)
def backward_sample(Ps, log_likes, alphas, us, zs):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K
    assert us.shape[0] == T
    assert zs.shape[0] == T

    lpzp1 = np.zeros(K)
    lpz = np.zeros(K)

    # Trick for handling time-varying transition matrices
    hetero = (Ps.shape[0] == T-1)

    for t in range(T-1,-1,-1):
        # compute normalized log p(z[t] = k | z[t+1])
        lpz = lpzp1 + alphas[t]
        Z = logsumexp(lpz)

        # sample
        acc = 0
        zs[t] = K-1
        for k in range(K):
            acc += np.exp(lpz[k] - Z)
            if us[t] < acc:
                zs[t] = k
                break

        # set the transition potential
        if t > 0:
            lpzp1 = np.log(Ps[(t-1) * hetero, :, int(zs[t])] + LOG_EPS)


@numba.jit(nopython=True, cache=True)
def _hmm_sample(pi0, Ps, ll):
    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)

    # Sample backward
    us = npr.rand(T)
    zs = -1 * np.ones(T)
    backward_sample(Ps, ll, alphas, us, zs)
    return zs


def hmm_sample(pi0, Ps, ll):
    return _hmm_sample(pi0, Ps, ll).astype(int)


@numba.jit(nopython=True, cache=True)
def _viterbi(pi0, Ps, ll):
    """
    This is modified from pyhsmm.internals.hmm_states
    by Matthew Johnson.
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or
    # time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)
    if not hetero:
        assert Ps.shape[0] == 1

    # Pass max-sum messages backward
    scores = np.zeros((T, K))
    args = np.zeros((T, K))
    for t in range(T-2,-1,-1):
        vals = np.log(Ps[t * hetero] + LOG_EPS) + scores[t+1] + ll[t+1]
        for k in range(K):
            args[t+1, k] = np.argmax(vals[k])
            scores[t, k] = np.max(vals[k])

    # Now maximize forwards
    z = np.zeros(T)
    z[0] = (scores[0] + np.log(pi0 + LOG_EPS) + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, int(z[t-1])]

    return z


def viterbi(pi0, Ps, ll):
    """
    Find the most likely state sequence
    """
    return _viterbi(pi0, Ps, ll).astype(int)


@numba.jit(nopython=True, cache=True)
def grad_hmm_normalizer(log_Ps,
                        alphas,
                        d_log_pi0,
                        d_log_Ps,
                        d_log_likes):

    T = alphas.shape[0]
    K = alphas.shape[1]
    assert (log_Ps.shape[0] == T-1) or (log_Ps.shape[0] == 1)
    assert d_log_Ps.shape[0] == log_Ps.shape[0]
    assert log_Ps.shape[1] == d_log_Ps.shape[1] == K
    assert log_Ps.shape[2] == d_log_Ps.shape[2] == K
    assert d_log_pi0.shape[0] == K
    assert d_log_likes.shape[0] == T
    assert d_log_likes.shape[1] == K

    # Initialize temp storage for gradients
    tmp1 = np.zeros((K,))
    tmp2 = np.zeros((K, K))

    # Trick for handling time-varying transition matrices
    hetero = (log_Ps.shape[0] == T-1)

    dlse(alphas[T-1], d_log_likes[T-1])
    for t in range(T-1, 0, -1):
        # tmp2 = dLSE_da(alphas[t-1], log_Ps[t-1])
        #      = np.exp(alphas[t-1] + log_Ps[t-1].T - logsumexp(alphas[t-1] + log_Ps[t-1].T, axis=1))
        #      = [dlse(alphas[t-1] + log_Ps[t-1, :, k]) for k in range(K)]
        for k in range(K):
            for j in range(K):
                tmp1[j] = alphas[t-1, j] + log_Ps[(t-1) * hetero, j, k]
            dlse(tmp1, tmp2[k])


        # d_log_Ps[t-1] = vjp_LSE_B(alphas[t-1], log_Ps[t-1], d_log_likes[t])
        #               = d_log_likes[t] * dLSE_da(alphas[t-1], log_Ps[t-1]).T
        #               = d_log_likes[t] * tmp2.T
        #
        # d_log_Ps[t-1, j, k] = d_log_likes[t, k] * tmp2.T[j, k]
        #                     = d_log_likes[t, k] * tmp2[k, j]
        for j in range(K):
            for k in range(K):
                d_log_Ps[(t-1) * hetero, j, k] += d_log_likes[t, k] * tmp2[k, j]

        # d_log_likes[t-1] = d_log_likes[t].dot(dLSE_da(alphas[t-1], log_Ps[t-1]))
        #                  = d_log_likes[t].dot(tmp2)
        for k in range(K):
            d_log_likes[t-1, k] = 0
            for j in range(K):
                d_log_likes[t-1, k] += d_log_likes[t, j] * tmp2[j, k]

    # d_log_pi0 = d_log_likes[0]
    for k in range(K):
        d_log_pi0[k] = d_log_likes[0, k]


@primitive
def hmm_normalizer(pi0, Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))

    # Make sure everything is C contiguous
    pi0 = to_c(pi0)
    Ps = to_c(Ps)
    ll = to_c(ll)

    forward_pass(pi0, Ps, ll, alphas)
    return logsumexp(alphas[-1])


def _make_grad_hmm_normalizer(argnum, ans, pi0, Ps, ll):
    # Make sure everything is C contiguous and unboxed
    pi0 = to_c(pi0)
    Ps = to_c(Ps)
    ll = to_c(ll)

    dlog_pi0 = np.zeros_like(pi0)
    dlog_Ps= np.zeros_like(Ps)
    dll = np.zeros_like(ll)
    T, K = ll.shape

    # Forward pass to get alphas
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)
    grad_hmm_normalizer(np.log(Ps + LOG_EPS), alphas, dlog_pi0, dlog_Ps, dll)

    # Compute necessary gradient
    # Account for the log transformation
    # df/dP = df/dlogP * dlogP/dP = df/dlogP * 1 / P
    if argnum == 0:
        return lambda g: g * dlog_pi0 / (pi0 + DIV_EPS)
    if argnum == 1:
        return lambda g: g * dlog_Ps / (Ps + DIV_EPS)
    if argnum == 2:
        return lambda g: g * dll

defvjp(hmm_normalizer,
       partial(_make_grad_hmm_normalizer, 0),
       partial(_make_grad_hmm_normalizer, 1),
       partial(_make_grad_hmm_normalizer, 2))

##
# Gaussian linear dynamical systems message passing code
##
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
def gaussian_logpdf(y, m, S):
    D = m.shape[0]
    L = np.linalg.cholesky(S)
    x = np.linalg.solve(L, y - m)
    return -0.5 * D * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))) -0.5 * np.sum(x**2)


@numba.jit(nopython=True, cache=True)
def _sample_gaussian(m, S, z):
    # Sample a multivariate Gaussian with mean m, covariance S,
    # using a standard normal vector z. Put the output in out.
    L = np.linalg.cholesky(S)
    return m + L @ z

@numba.jit(nopython=True, cache=True)
def _kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    T, N = ys.shape
    D = mu0.shape[0]

    # Check for stationary dynamics parameters
    hetero = As.shape[0] > 1

    predicted_mus = np.zeros((T, D))        # preds E[x_t | y_{1:t-1}]
    predicted_Sigmas = np.zeros((T, D, D))  # preds Cov[x_t | y_{1:t-1}]
    filtered_mus = np.zeros((T, D))         # means E[x_t | y_{1:t}]
    filtered_Sigmas = np.zeros((T, D, D))   # means Cov[x_t | y_{1:t}]

    # Initialize
    predicted_mus[0] = mu0
    predicted_Sigmas[0] = S0
    K = np.zeros((D, N))
    ll = 0

    # Run the Kalman filter
    for t in range(T):
        At = As[t * hetero]
        Bt = Bs[t * hetero]
        Qt = Qs[t * hetero]
        Ct = Cs[t * hetero]
        Dt = Ds[t * hetero]
        Rt = Rs[t * hetero]

        # Update the log likelihood
        ll += gaussian_logpdf(ys[t],
            Ct @ predicted_mus[t] + Dt @ us[t],
            Ct @ predicted_Sigmas[t] @ Ct.T + Rt
            )

        # Condition on this frame's observations
        _condition_on(predicted_mus[t], predicted_Sigmas[t],
            Ct, Dt, Rt, us[t], ys[t],
            filtered_mus[t], filtered_Sigmas[t])

        if t == T-1:
            break

        # Predict the next frame's latent state
        _predict(filtered_mus[t], filtered_Sigmas[t],
            At, Bt, Qt, us[t],
            predicted_mus[t+1], predicted_Sigmas[t+1])

    return ll, filtered_mus, filtered_Sigmas


@numba.jit(nopython=True, cache=True)
def _kalman_sample(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    T, N = ys.shape
    D = mu0.shape[0]

    # Check for stationary dynamics parameters
    hetero = As.shape[0] > 1

    # Run the Kalman Filter
    ll, filtered_mus, filtered_Sigmas = \
        _kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

    # Initialize outputs, noise, and temporary variables
    xs = np.zeros((T, D))
    noise = np.random.randn(T, D)
    mu_cond = np.zeros(D)
    Sigma_cond = np.zeros((D, D))

    # Sample backward in time
    xs[-1] = _sample_gaussian(filtered_mus[-1], filtered_Sigmas[-1], noise[-1])
    for t in range(T-2, -1, -1):
        At = As[t * hetero]
        Bt = Bs[t * hetero]
        Qt = Qs[t * hetero]

        _condition_on(filtered_mus[t], filtered_Sigmas[t],
            At, Bt, Qt, us[t], xs[t+1],
            mu_cond, Sigma_cond)

        xs[t] = _sample_gaussian(mu_cond, Sigma_cond, noise[t])

    return ll, xs


@numba.jit(nopython=True, cache=True)
def _kalman_smoother(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    T, N = ys.shape
    D = mu0.shape[0]

    # Check for stationary dynamics parameters
    hetero = As.shape[0] > 1

    # Run the Kalman Filter
    ll, filtered_mus, filtered_Sigmas = \
        _kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

    # Initialize outputs, noise, and temporary variables
    smoothed_mus = np.zeros((T, D))
    smoothed_Sigmas = np.zeros((T, D, D))
    # smoothed_CrossSigmas = np.zeros((T-1, D, D))
    ExxnT = np.zeros((T-1, D, D))
    Gt = np.zeros((D, D))

    # The last time step is known from the Kalman filter
    smoothed_mus[-1] = filtered_mus[-1]
    smoothed_Sigmas[-1] = filtered_Sigmas[-1]

    # Run the smoother backward in time
    for t in range(T-2, -1, -1):
        At = As[t * hetero]
        Bt = Bs[t * hetero]
        Qt = Qs[t * hetero]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        Gt = np.linalg.solve(Qt + At @ filtered_Sigmas[t] @ At.T,
                             At @ filtered_Sigmas[t]).T

        smoothed_mus[t] = filtered_mus[t] + Gt @ (smoothed_mus[t+1] - At @ filtered_mus[t] - Bt @ us[t])
        smoothed_Sigmas[t] = filtered_Sigmas[t] + \
                             Gt @ (smoothed_Sigmas[t+1] - At @ filtered_Sigmas[t] @ At.T - Qt) @ Gt.T
        # smoothed_CrossSigmas[t] = Gt @ smoothed_Sigmas[t+1]
        ExxnT[t] = Gt @ smoothed_Sigmas[t+1] + np.outer(smoothed_mus[t], smoothed_mus[t+1])

    return ll, smoothed_mus, smoothed_Sigmas, ExxnT


def kalman_wrapper(f):
    def wrapper(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
        """
        Notation:

        T:  number of time steps
        D:  continuous latent state dimension
        U:  input dimension
        N:  observed data dimension

        mu0: (D,)                   initial state mean
        S0:  (D, D)                 initial state covariance
        As:  (D, D) or (T-1, D, D)  dynamics matrices
        Bs:  (D, U) or (T-1, D, U)  input to latent state matrices
        Qs:  (D, D) or (T-1, D, D)  dynamics covariance matrices
        Cs:  (N, D) or (T, N, D)    emission matrices
        Ds:  (N, U) or (T, N, U)    input to emissions matrices
        Rs:  (N, N) or (T, N, N)    emission covariance matrices
        us:  (T, U)                 inputs
        ys:  (T, N)                 observations
        """
        # Get shapes
        D = mu0.shape[0]
        T, N = ys.shape
        U = us.shape[1]

        assert mu0.shape == (D,)
        assert S0.shape == (D, D)
        assert As.shape == (D, D) or As.shape == (T-1, D, D)
        assert Bs.shape == (D, U) or Bs.shape == (T-1, D, U)
        assert Qs.shape == (D, D) or Qs.shape == (T-1, D, D)
        assert Cs.shape == (N, D) or Cs.shape == (T, N, D)
        assert Ds.shape == (N, U) or Ds.shape == (T, N, U)
        assert Rs.shape == (N, N) or Rs.shape == (T, N, N)
        assert us.shape == (T, U)

        # Add extra time dimension if necessary
        As = As if As.ndim == 3 else np.reshape(As, (1,) + As.shape)
        Bs = Bs if Bs.ndim == 3 else np.reshape(Bs, (1,) + Bs.shape)
        Qs = Qs if Qs.ndim == 3 else np.reshape(Qs, (1,) + Qs.shape)
        Cs = Cs if Cs.ndim == 3 else np.reshape(Cs, (1,) + Cs.shape)
        Ds = Ds if Ds.ndim == 3 else np.reshape(Ds, (1,) + Ds.shape)
        Rs = Rs if Rs.ndim == 3 else np.reshape(Rs, (1,) + Rs.shape)
        return f(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

    return wrapper


@kalman_wrapper
def kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    """
    Standard Kalman filter for time-varying linear dynamical system with inputs.
    """
    return _kalman_filter(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)


@kalman_wrapper
def kalman_sample(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    """
    Sample from a linear Gaussian model.  Run the KF to get
    the filtered probability distributions, then sample
    backward in time.
    """
    return _kalman_sample(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)

@kalman_wrapper
def kalman_smoother(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    """
    Compute the conditional mean and variance of the latent
    states given observed data ys and inputs us.  Run the KF to get
    the filtered probability distributions, then run the Rauch-Tung-Striebel
    smoother backward in time.
    """
    return _kalman_smoother(mu0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys)


##
# Kalman filter/sampler/smoother with diagonal observation noise
##
@numba.jit(nopython=True, cache=True)
def _condition_on_diagonal(m, S, C, D, R_diag, u, y, mcond, Scond):
    # Same as above but where R is assumed to be a diagonal covariance matrix
    # The unnormalized potential is
    #
    #   p(x) \propto N(x | m, S) * N(y | Cx + Du, R)
    #        = N(m', S')  where
    #
    #     S' = [S^{-1} + C^T R^{-1} C]^{-1}
    #     m' = S' [S^{-1} m + C^T R^{-1} (y - Du)]
    #
    Scond[:] = np.linalg.inv(np.linalg.inv(S) + (C.T / R_diag) @ C)
    # Scond[:] = S - np.linalg.solve(R + C @ S @ C.T, C @ S).T @ C @ S
    mcond[:] = Scond @ (np.linalg.solve(S, m) + (C.T / R_diag) @ (y - D @ u))


@numba.jit(nopython=True, cache=True)
def gaussian_logpdf_lrpd(y, m, C, S, r):
    # compute N(y | m, diag(r) + C S C^T)
    #
    # We need to compute J = (diag(r) + C S C^T)^{-1} and its determinant.
    # By the matrix inversion lemma, this is,
    #
    #    J = I/r - (C/r) (S^{-1} + (C/r)^T C)^{-1} (C/r)^T
    #      = I/r - (C/r) L^{-T} L^{-1} (C/r)^T
    #      = I/r - K K^T
    #    L = chol(S^{-1} + (C/r)^T C)
    #    K = (C/r) L^{-T}
    #    KT = L^{-1} (C/r)^T
    #
    # The determinant is
    #    det(J) = det(I/r) det(I + (K * r)^T K)
    #           = 1/prod(r) det(I + (K * r)^T K)
    n = C.shape[0]
    d = C.shape[1]
    CTr = C.T / r
    L = np.linalg.cholesky(np.linalg.inv(S) + CTr @ C)
    KT = np.linalg.solve(L, CTr)
    tmp = np.linalg.cholesky(np.eye(d) - (KT * r) @ KT.T)
    x = KT @ (y - m)
    return -0.5 * n * np.log(2 * np.pi) \
           -0.5 * np.sum(np.log(r)) + np.sum(np.log(np.diag(tmp))) \
           -0.5 * np.sum((y - m)**2 / r) +0.5 * np.sum(x**2)


@numba.jit(nopython=True, cache=True)
def _kalman_filter_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
    T, N = ys.shape
    D = mu0.shape[0]

    # Check for stationary dynamics parameters
    hetero = As.shape[0] > 1

    predicted_mus = np.zeros((T, D))        # preds E[x_t | y_{1:t-1}]
    predicted_Sigmas = np.zeros((T, D, D))  # preds Cov[x_t | y_{1:t-1}]
    filtered_mus = np.zeros((T, D))         # means E[x_t | y_{1:t}]
    filtered_Sigmas = np.zeros((T, D, D))   # means Cov[x_t | y_{1:t}]

    # Initialize
    predicted_mus[0] = mu0
    predicted_Sigmas[0] = S0
    K = np.zeros((D, N))
    ll = 0

    # Run the Kalman filter
    for t in range(T):
        At = As[t * hetero]
        Bt = Bs[t * hetero]
        Qt = Qs[t * hetero]
        Ct = Cs[t * hetero]
        Dt = Ds[t * hetero]
        Rt = R_diags[t * hetero]

        # Update the log likelihood
        ll += gaussian_logpdf_lrpd(ys[t],
            Ct @ predicted_mus[t] + Dt @ us[t],
            Ct, predicted_Sigmas[t], Rt
            )

        # Condition on this frame's observations
        _condition_on_diagonal(predicted_mus[t], predicted_Sigmas[t],
            Ct, Dt, Rt, us[t], ys[t],
            filtered_mus[t], filtered_Sigmas[t])

        if t == T-1:
            break

        # Predict the next frame's latent state
        _predict(filtered_mus[t], filtered_Sigmas[t],
            At, Bt, Qt, us[t],
            predicted_mus[t+1], predicted_Sigmas[t+1])

    return ll, filtered_mus, filtered_Sigmas


@numba.jit(nopython=True, cache=True)
def _kalman_sample_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
    T, N = ys.shape
    D = mu0.shape[0]

    # Check for stationary dynamics parameters
    hetero = As.shape[0] > 1

    # Run the Kalman Filter
    ll, filtered_mus, filtered_Sigmas = \
        _kalman_filter_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys)

    # Initialize outputs, noise, and temporary variables
    xs = np.zeros((T, D))
    noise = np.random.randn(T, D)
    mu_cond = np.zeros(D)
    Sigma_cond = np.zeros((D, D))

    # Sample backward in time
    xs[-1] = _sample_gaussian(filtered_mus[-1], filtered_Sigmas[-1], noise[-1])
    for t in range(T-2, -1, -1):
        At = As[t * hetero]
        Bt = Bs[t * hetero]
        Qt = Qs[t * hetero]

        _condition_on(filtered_mus[t], filtered_Sigmas[t],
            At, Bt, Qt, us[t], xs[t+1],
            mu_cond, Sigma_cond)

        xs[t] = _sample_gaussian(mu_cond, Sigma_cond, noise[t])

    return ll, xs

@numba.jit(nopython=True, cache=True)
def _kalman_smoother_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
    T, N = ys.shape
    D = mu0.shape[0]

    # Check for stationary dynamics parameters
    hetero = As.shape[0] > 1

    # Run the Kalman Filter
    ll, filtered_mus, filtered_Sigmas = \
        _kalman_filter_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys)

    # Initialize outputs, noise, and temporary variables
    smoothed_mus = np.zeros((T, D))
    smoothed_Sigmas = np.zeros((T, D, D))
    # smoothed_CrossSigmas = np.zeros((T-1, D, D))
    ExxnT = np.zeros((T-1, D, D))
    Gt = np.zeros((D, D))

    # The last time step is known from the Kalman filter
    smoothed_mus[-1] = filtered_mus[-1]
    smoothed_Sigmas[-1] = filtered_Sigmas[-1]

    # Run the smoother backward in time
    for t in range(T-2, -1, -1):
        At = As[t * hetero]
        Bt = Bs[t * hetero]
        Qt = Qs[t * hetero]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        Gt = np.linalg.solve(Qt + At @ filtered_Sigmas[t] @ At.T,
                             At @ filtered_Sigmas[t]).T

        smoothed_mus[t] = filtered_mus[t] + Gt @ (smoothed_mus[t+1] - At @ filtered_mus[t] - Bt @ us[t])
        smoothed_Sigmas[t] = filtered_Sigmas[t] + \
                             Gt @ (smoothed_Sigmas[t+1] - At @ filtered_Sigmas[t] @ At.T - Qt) @ Gt.T
        # smoothed_CrossSigmas[t] = Gt @ smoothed_Sigmas[t+1]
        ExxnT[t] = Gt @ smoothed_Sigmas[t+1] + np.outer(smoothed_mus[t], smoothed_mus[t+1])

    return ll, smoothed_mus, smoothed_Sigmas, ExxnT


def kalman_wrapper_diagonal(f):
    def wrapper(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
        """
        Notation:
        T:  number of time steps
        D:  continuous latent state dimension
        U:  input dimension
        N:  observed data dimension
        mu0: (D,)                   initial state mean
        S0:  (D, D)                 initial state covariance
        As:  (D, D) or (T-1, D, D)  dynamics matrices
        Bs:  (D, U) or (T-1, D, U)  input to latent state matrices
        Qs:  (D, D) or (T-1, D, D)  dynamics covariance matrices
        Cs:  (N, D) or (T, N, D)    emission matrices
        Ds:  (N, U) or (T, N, U)    input to emissions matrices
        R_diags:  (N,) or (T, N,)   diagonal of emission covariance matrices
        us:  (T, U)                 inputs
        ys:  (T, N)                 observations
        """
        # Get shapes
        D = mu0.shape[0]
        T, N = ys.shape
        U = us.shape[1]

        assert mu0.shape == (D,)
        assert S0.shape == (D, D)
        assert As.shape == (D, D) or As.shape == (T-1, D, D)
        assert Bs.shape == (D, U) or Bs.shape == (T-1, D, U)
        assert Qs.shape == (D, D) or Qs.shape == (T-1, D, D)
        assert Cs.shape == (N, D) or Cs.shape == (T, N, D)
        assert Ds.shape == (N, U) or Ds.shape == (T, N, U)
        assert R_diags.shape == (N,) or R_diags.shape == (T, N)
        assert us.shape == (T, U)

        # Add extra time dimension if necessary
        As = As if As.ndim == 3 else np.reshape(As, (1,) + As.shape)
        Bs = Bs if Bs.ndim == 3 else np.reshape(Bs, (1,) + Bs.shape)
        Qs = Qs if Qs.ndim == 3 else np.reshape(Qs, (1,) + Qs.shape)
        Cs = Cs if Cs.ndim == 3 else np.reshape(Cs, (1,) + Cs.shape)
        Ds = Ds if Ds.ndim == 3 else np.reshape(Ds, (1,) + Ds.shape)
        R_diags = R_diags if R_diags.ndim == 2 else np.reshape(R_diags, (1,) + R_diags.shape)
        return f(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys)

    return wrapper


@kalman_wrapper_diagonal
def kalman_filter_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
    """
    Standard Kalman filter for time-varying linear dynamical system with inputs.

    This version assumes the emission covariance is diagonal.
    """
    return _kalman_filter_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys)


@kalman_wrapper_diagonal
def kalman_sample_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
    """
    Sample from a linear Gaussian model.  Run the KF to get the filtered
    probability distributions, then sample backward in time.

    This version assumes the emission covariance is diagonal.
    """
    return _kalman_sample_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys)


@kalman_wrapper_diagonal
def kalman_smoother_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys):
    """
    Compute the conditional mean and variance of the latent
    states given observed data ys and inputs us.  Run the KF to get
    the filtered probability distributions, then run the Rauch-Tung-Striebel
    smoother backward in time.

    This version assumes the emission covariance is diagonal.
    """
    return _kalman_smoother_diagonal(mu0, S0, As, Bs, Qs, Cs, Ds, R_diags, us, ys)

##
# Information form filtering and smoothing
##
@numba.jit(nopython=True, cache=True)
def _info_condition_on(J_pred, h_pred, J_obs, h_obs, log_Z_obs, J_cond, h_cond):
    J_cond[:] = J_pred + J_obs
    h_cond[:] = h_pred + h_obs
    return log_Z_obs


@numba.jit(nopython=True, cache=True)
def _info_lognorm(J, h):
    # Update the log normalizer, marginalizing out x_t
    D = h.shape[0]
    log_Z = 0.5 * h @ np.linalg.solve(J, h)
    log_Z += -0.5 * np.linalg.slogdet(J)[1]
    log_Z += 0.5 * D * np.log(2 * np.pi)
    return log_Z


@numba.jit(nopython=True, cache=True)
def _info_predict(J_filt, h_filt, J_11, J_21, J_22, h_1, h_2, log_Z_dyn, J_pred, h_pred):
    tmp_J = J_filt + J_11
    tmp_h = h_filt + h_1
    J_pred[:] = J_22 - np.dot(J_21, np.linalg.solve(tmp_J, J_21.T))
    h_pred[:] = h_2 - np.dot(J_21, np.linalg.solve(tmp_J, tmp_h))

    # Update the log normalizer, marginalizing out x_t
    return log_Z_dyn + _info_lognorm(tmp_J, tmp_h)


@numba.jit(nopython=True, cache=True)
def _sample_info_gaussian(J, h, noise):
     L = np.linalg.cholesky(J)
     # sample = spla.solve_triangular(L, noise, lower=True, trans='T')
     sample = np.linalg.solve(L.T, noise)
     # from scipy.linalg.lapack import dpotrs
     # sample += dpotrs(L, h, lower=True)[0]
     # sample += spla.cho_solve((L, True), h)
     sample += np.linalg.solve(J, h)
     return sample


@numba.jit(nopython=True, cache=True)
def _kalman_info_filter_with_predictions(
    J_ini, h_ini, log_Z_ini,
    J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
    J_obs, h_obs, log_Z_obs):
    """
    Information form Kalman filter for time-varying linear dynamical system with inputs.
    """
    T, D = h_obs.shape

    # Allocate output arrays
    filtered_Js = np.zeros((T, D, D))
    filtered_hs = np.zeros((T, D))
    predicted_Js = np.zeros((T, D, D))
    predicted_hs = np.zeros((T, D))

    # Initialize
    predicted_Js[0] = J_ini
    predicted_hs[0] = h_ini
    log_Z = log_Z_ini

    # Run the Kalman information filter
    for t in range(T-1):
        # Extract blocks of the dynamics potentials
        J_11 = J_dyn_11[t] if J_dyn_11.shape[0] == T-1 else J_dyn_11[0]
        J_21 = J_dyn_21[t] if J_dyn_21.shape[0] == T-1 else J_dyn_21[0]
        J_22 = J_dyn_22[t] if J_dyn_22.shape[0] == T-1 else J_dyn_22[0]
        h_1 = h_dyn_1[t] if h_dyn_1.shape[0] == T-1 else h_dyn_1[0]
        h_2 = h_dyn_2[t] if h_dyn_2.shape[0] == T-1 else h_dyn_2[0]
        log_Z_d = log_Z_dyn[t] if log_Z_dyn.shape[0] == T-1 else log_Z_dyn[0]
        J_o = J_obs[t] if J_obs.shape[0] == T else J_obs[0]
        h_o = h_obs[t] if h_obs.shape[0] == T else h_obs[0]
        log_Z_o = log_Z_obs[t] if log_Z_obs.shape[0] == T else log_Z_obs[0]

        # Condition on the observed data
        log_Z += _info_condition_on(
            predicted_Js[t], predicted_hs[t],
            J_o, h_o, log_Z_o,
            filtered_Js[t], filtered_hs[t])

        # Predict the next frame
        log_Z += _info_predict(
            filtered_Js[t], filtered_hs[t],
            J_11, J_21, J_22, h_1, h_2, log_Z_d,
            predicted_Js[t+1], predicted_hs[t+1])

    # Condition on the last observation
    log_Z += _info_condition_on(
        predicted_Js[-1], predicted_hs[-1],
        J_obs[-1], h_obs[-1], log_Z_obs[-1],
        filtered_Js[-1], filtered_hs[-1])

    # Account for the last observation potential
    log_Z += _info_lognorm(filtered_Js[-1], filtered_hs[-1])

    return log_Z, filtered_Js, filtered_hs, predicted_Js, predicted_hs


@numba.jit(nopython=True, cache=True)
def _kalman_info_filter(
    J_ini, h_ini, log_Z_ini,
    J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
    J_obs, h_obs, log_Z_obs):

    log_Z, filtered_Js, filtered_hs, _, _ = \
        _kalman_info_filter_with_predictions(
            J_ini, h_ini, log_Z_ini,
            J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
            J_obs, h_obs, log_Z_obs)

    return log_Z, filtered_Js, filtered_hs


#@numba.jit(nopython=True, cache=True)
def _kalman_info_sample(J_ini, h_ini, log_Z_ini,
                       J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
                       J_obs, h_obs, log_Z_obs):
    """
    Information form Kalman sampling for time-varying linear dynamical system with inputs.
    """
    T, D = h_obs.shape

    # Run the forward filter
    log_Z, filtered_Js, filtered_hs = \
        _kalman_info_filter(J_ini, h_ini, log_Z_ini,
                           J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
                           J_obs, h_obs, log_Z_obs)

    # Allocate output arrays
    samples = np.zeros((T, D))
    noise = npr.randn(T, D)

    # Initialize with samples of the last state
    samples[-1] = _sample_info_gaussian(filtered_Js[-1], filtered_hs[-1], noise[-1])

    # Run the Kalman information filter
    for t in range(T-2, -1, -1):
        # Extract blocks of the dynamics potentials
        J_11 = J_dyn_11[t] if J_dyn_11.shape[0] == T-1 else J_dyn_11[0]
        J_21 = J_dyn_21[t] if J_dyn_21.shape[0] == T-1 else J_dyn_21[0]
        h_1 = h_dyn_1[t] if h_dyn_1.shape[0] == T-1 else h_dyn_1[0]

        # Condition on the next observation
        J_post = filtered_Js[t] + J_11
        h_post = filtered_hs[t] + h_1 - np.dot(J_21.T, samples[t+1])
        samples[t] = _sample_info_gaussian(J_post, h_post, noise[t])

    return samples


@numba.jit(nopython=True, cache=True)
def _kalman_info_smoother(J_ini, h_ini, log_Z_ini,
                         J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
                         J_obs, h_obs, log_Z_obs,):
    """
    Information form Kalman smoother for time-varying linear dynamical system with inputs.
    """
    T, D = h_obs.shape

    # Run the forward filter
    log_Z, filtered_Js, filtered_hs, predicted_Js, predicted_hs = \
        _kalman_info_filter_with_predictions(
            J_ini, h_ini, log_Z_ini,
            J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
            J_obs, h_obs, log_Z_obs)

    # Allocate output arrays
    smoothed_Js = np.zeros((T, D, D))
    smoothed_hs = np.zeros((T, D))
    smoothed_mus = np.zeros((T, D))
    smoothed_Sigmas = np.zeros((T, D, D))
    ExxnT = np.zeros((T-1, D, D))

    # Initialize
    smoothed_Js[-1] = filtered_Js[-1]
    smoothed_hs[-1] = filtered_hs[-1]
    smoothed_Sigmas[-1] = np.linalg.inv(smoothed_Js[-1])
    smoothed_mus[-1] = np.dot(smoothed_Sigmas[-1], smoothed_hs[-1])

    # Run the Kalman information filter
    for t in range(T-2, -1, -1):
        # Extract blocks of the dynamics potentials
        J_11 = J_dyn_11[t] if J_dyn_11.shape[0] == T-1 else J_dyn_11[0]
        J_21 = J_dyn_21[t] if J_dyn_21.shape[0] == T-1 else J_dyn_21[0]
        J_22 = J_dyn_22[t] if J_dyn_22.shape[0] == T-1 else J_dyn_22[0]
        h_1 = h_dyn_1[t] if h_dyn_1.shape[0] == T-1 else h_dyn_1[0]
        h_2 = h_dyn_2[t] if h_dyn_2.shape[0] == T-1 else h_dyn_2[0]

        # Combine filtered and smoothed estimates
        J_inner = smoothed_Js[t+1] - predicted_Js[t+1] + J_22
        h_inner = smoothed_hs[t+1] - predicted_hs[t+1] + h_2
        smoothed_Js[t] = filtered_Js[t] + J_11 - np.dot(J_21.T, np.linalg.solve(J_inner, J_21))
        smoothed_hs[t] = filtered_hs[t] + h_1 - np.dot(J_21.T, np.linalg.solve(J_inner, h_inner))

        # Convert info form to mean parameters
        smoothed_Sigmas[t] = np.linalg.inv(smoothed_Js[t])
        smoothed_mus[t] = np.dot(smoothed_Sigmas[t], smoothed_hs[t])

        # TODO: Move this math to a separate document
        # Compute the cross-covariance between times (t, t+1)
        # We have
        #
        #   log p(x_t, x_{t+1} | y_{1:T})
        #       = (x_t, x_{t+1})^T J (x_t, x_{t+1}) + ...
        #
        # where J = [J_a, J_b; J_b.T, J_c] is a (2D, 2D) precision matrix with
        #   J_a = filtered_Js[t] + J_11
        #   J_b = J_12
        #
        # Moreover, J^{-1} = [S_a, S_b; S_b.T, S_c] where
        #   S_a = smoothed_Sigmas[t]
        #   S_c = smoothed_Sigmas[t+1]
        #
        # We want to solve for S_b. From the block inversion formula we have
        #   S_b = -J_a^{-1} J_b S_c
        ExxnT[t] = -np.linalg.solve(filtered_Js[t] + J_11, np.dot(J_21.T, smoothed_Sigmas[t+1]))
        ExxnT[t] += np.outer(smoothed_mus[t], smoothed_mus[t+1])

    return log_Z, smoothed_mus, smoothed_Sigmas, ExxnT


def kalman_info_wrapper(f):
    """
    We write each conditional distribution in terms of its natural parameters.

    log p(x_1) = x_1^T J_ini x_1 + x_1^T h_ini - log_Z_ini
    log p(x_t | x_{t-1}) = (x_t, x_{t-1})^T J_dyn (x_t, x_{t-1}) + (x_t, x_{t-1})^T h_dyn - log_Z_dyn
    log p(y_t | x_t) = x_t^T J_obs x_t + x_t^T h_obs - log_Z_obs

    Notation:

    T:  number of time steps
    D:  continuous latent state dimension

    Initial distribution parameters:
    J_ini:     (D, D)       initial state precision
    h_ini:     (D,)         initial state bias
    log_Z_ini: (,)          initial state log normalizer

    If time-varying dynamics:
    J_dyn_11:  (T-1, D, D)  upper left block of dynamics precision
    J_dyn_21:  (T-1, D, D)  lower left block of dynamics precision
    J_dyn_22:  (T-1, D, D)  lower right block of dynamics precision
    h_dyn_1:   (T-1, D)     upper block of dynamics bias
    h_dyn_2:   (T-1, D)     lower block of dynamics bias
    log_Z_dyn: (T-1,)       dynamics log normalizer

    If stationary dynamics:
    J_dyn_11:  (D, D)       upper left block of dynamics precision
    J_dyn_21:  (D, D)       lower left block of dynamics precision
    J_dyn_22:  (D, D)       lower right block of dynamics precision
    h_dyn_1:   (D,)         upper block of dynamics bias
    h_dyn_2:   (D,)         lower block of dynamics bias
    log_Z_dyn: (,)          dynamics log normalizer

    Observation distribution parameters
    J_obs:     (T, D, D)    observation precision
    h_obs:     (T, D)       observation bias
    log_Z_obs: (T,)         observation log normalizer
    """
    def wrapper(J_ini, h_ini, log_Z_ini,
                J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
                J_obs, h_obs, log_Z_obs, **kwargs):

        T, D = h_obs.shape
        # Check shapes
        assert J_ini.shape == (D, D)
        assert h_ini.shape == (D,)
        assert np.isscalar(log_Z_ini)
        assert J_dyn_11.shape == (D, D) or J_dyn_11.shape == (T-1, D, D)
        assert J_dyn_21.shape == (D, D) or J_dyn_21.shape == (T-1, D, D)
        assert J_dyn_22.shape == (D, D) or J_dyn_22.shape == (T-1, D, D)
        assert h_dyn_1.shape == (T-1, D) or h_dyn_1.shape == (D,)
        assert h_dyn_2.shape == (T-1, D) or h_dyn_2.shape == (D,)
        assert np.isscalar(log_Z_dyn) or log_Z_dyn.shape == (T-1,)
        assert J_obs.shape == (T, D, D)
        assert np.isscalar(log_Z_obs) or log_Z_obs.shape == (T,)

        # Add extra time dimension if necessary
        J_dyn_11 = J_dyn_11 if J_dyn_11.ndim == 3 else np.reshape(J_dyn_11, (1,) + J_dyn_11.shape)
        J_dyn_21 = J_dyn_21 if J_dyn_21.ndim == 3 else np.reshape(J_dyn_21, (1,) + J_dyn_21.shape)
        J_dyn_22 = J_dyn_22 if J_dyn_22.ndim == 3 else np.reshape(J_dyn_22, (1,) + J_dyn_22.shape)
        h_dyn_1 = h_dyn_1 if h_dyn_1.ndim == 2 else np.reshape(h_dyn_1, (1,) + h_dyn_1.shape)
        h_dyn_2 = h_dyn_2 if h_dyn_2.ndim == 2 else np.reshape(h_dyn_2, (1,) + h_dyn_2.shape)
        log_Z_dyn = np.array([log_Z_dyn]) if np.isscalar(log_Z_dyn) else log_Z_dyn
        J_obs = J_obs if J_obs.ndim == 3 else np.reshape(J_obs, (1,) + J_obs.shape)
        log_Z_obs = np.array([log_Z_obs]) if np.isscalar(log_Z_obs) else log_Z_obs

        return f(J_ini, h_ini, log_Z_ini,
                 J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
                 J_obs, h_obs, log_Z_obs, **kwargs)


    return wrapper


@kalman_info_wrapper
def kalman_info_filter(
    J_ini, h_ini, log_Z_ini,
    J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
    J_obs, h_obs, log_Z_obs):

    return _kalman_info_filter(
        J_ini, h_ini, log_Z_ini,
        J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
        J_obs, h_obs, log_Z_obs)


@kalman_info_wrapper
def kalman_info_filter_with_predictions(
    J_ini, h_ini, log_Z_ini,
    J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
    J_obs, h_obs, log_Z_obs):

    return _kalman_info_filter_with_predictions(
        J_ini, h_ini, log_Z_ini,
        J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
        J_obs, h_obs, log_Z_obs)


@kalman_info_wrapper
def kalman_info_sample(
    J_ini, h_ini, log_Z_ini,
    J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
    J_obs, h_obs, log_Z_obs):

    return _kalman_info_sample(
        J_ini, h_ini, log_Z_ini,
        J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
        J_obs, h_obs, log_Z_obs)


@kalman_info_wrapper
def kalman_info_smoother(
    J_ini, h_ini, log_Z_ini,
    J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
    J_obs, h_obs, log_Z_obs):

    return _kalman_info_smoother(
        J_ini, h_ini, log_Z_ini,
        J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
        J_obs, h_obs, log_Z_obs)


# # Solve and multiply symmetric block tridiagonal systems
# def solve_symm_block_tridiag(J_diag, J_lower_diag, v):
#     J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
#     x_flat = solveh_banded(J_banded, np.ravel(v), lower=True)
#     return np.reshape(x_flat, v.shape)


# def symm_block_tridiag_matmul(J_diag, J_lower_diag, v):
#     """
#     Compute matrix-vector product with a symmetric block
#     tridiagonal matrix J and vector v.
#     :param J_diag:          block diagonal terms of J
#     :param J_lower_diag:    lower block diagonal terms of J
#     :param v:               vector to multiply
#     :return:                J * v
#     """
#     T, D, _ = J_diag.shape
#     assert J_diag.ndim == 3 and J_diag.shape[2] == D
#     assert J_lower_diag.shape == (T-1, D, D)
#     assert v.shape == (T, D)

#     out = np.matmul(J_diag, v[:, :, None])[:, :, 0]
#     out[:-1] += np.matmul(np.swapaxes(J_lower_diag, -1, -2), v[1:][:, :, None])[:, :, 0]
#     out[1:] += np.matmul(J_lower_diag, v[:-1][:, :, None])[:, :, 0]
#     return out


##
# Test
##
def make_lds_parameters(T, D, N, U):
    m0 = np.zeros(D)
    S0 = np.eye(D)
    As = 0.99 * np.eye(D)
    Bs = np.zeros((D, U))
    Qs = 0.1 * np.eye(D)
    Cs = npr.randn(N, D)
    Ds = np.zeros((N, U))
    Rs = 0.1 * np.eye(N)
    us = np.zeros((T, U))
    ys = np.sin(2 * np.pi * np.arange(T) / 50)[:, None] * npr.randn(1, 10) + 0.1 * npr.randn(T, N)
    return m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys


@kalman_wrapper
def convert_mean_to_info_args(m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys):
    T = ys.shape[0]

    # Conver the initial distribution
    J_ini = np.linalg.inv(S0)
    h_ini = np.dot(J_ini, m0)

    # Conver the dynamics distribution
    Qi = np.linalg.inv(Qs)
    QiA = np.matmul(Qi, As)
    uB = np.matmul(us[:-1, None, :], np.swapaxes(Bs, 1, 2))
    J_dyn_11 = np.matmul(np.swapaxes(As, 1, 2), QiA) * np.ones((T-1, 1, 1))
    J_dyn_21 = -QiA * np.ones((T-1, 1, 1))
    J_dyn_22 = Qi * np.ones((T-1, 1, 1))
    h_dyn_1 = -np.matmul(uB, QiA)[:, 0, :] * np.ones((T-1, 1))
    h_dyn_2 = np.matmul(uB, Qi)[:, 0, :] * np.ones((T-1, 1))

    # Conver the emission distribution
    RiC = np.linalg.solve(Rs, Cs)
    J_obs = np.matmul(np.swapaxes(Cs, 1, 2), RiC) * np.ones((T, 1, 1))
    h_obs = np.matmul(np.swapaxes(ys[:, :, None] - np.matmul(Ds, us[:, :, None]), 1, 2), RiC)[:, 0, :]
    return J_ini, h_ini, 0, J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, np.zeros(T-1), J_obs, h_obs, np.zeros(T)


def test_lds(T=1000, D=1, N=10, U=3):
    args = make_lds_parameters(T, D, N, U)

    # Test the standard Kalman filter
    from pylds.lds_messages_interface import kalman_filter as kalman_filter_ref
    ll1, filtered_mus1, filtered_Sigmas1 = kalman_filter_ref(*args)
    ll2, filtered_mus2, filtered_Sigmas2 = kalman_filter(*args)
    assert np.allclose(ll1, ll2)
    assert np.allclose(filtered_mus1, filtered_mus2)
    assert np.allclose(filtered_Sigmas1, filtered_Sigmas2)

    # Sample
    xs = kalman_sample(*args)

    # Test the standard Kalman smoother
    from pylds.lds_messages_interface import E_step as kalman_smoother_ref
    ll1, smoothed_mus1, smoothed_Sigmas1, ExnxT1 = kalman_smoother_ref(*args)
    ll2, smoothed_mus2, smoothed_Sigmas2, ExxnT2 = kalman_smoother(*args)
    assert np.allclose(ll1, ll2)
    assert np.allclose(smoothed_mus1, smoothed_mus2)
    assert np.allclose(smoothed_Sigmas1, smoothed_Sigmas2)
    assert np.allclose(ExnxT1, np.swapaxes(ExxnT2, 1, 2))

    # Test the info form filter
    info_args = convert_mean_to_info_args(*args)
    from pylds.lds_messages_interface import kalman_info_filter as kalman_info_filter_ref
    log_Z1, filtered_Js1, filtered_hs1 = kalman_info_filter_ref(*info_args)
    log_Z2, filtered_Js2, filtered_hs2 = kalman_info_filter(*info_args)
    assert np.allclose(log_Z1, log_Z2)
    assert np.allclose(filtered_Js1, filtered_Js2)
    assert np.allclose(filtered_hs1, filtered_hs2)

    # Test the info form smoother
    _, smoothed_mus3, smoothed_Sigmas3, ExxnT3 = kalman_info_smoother(*info_args)
    assert np.allclose(smoothed_mus1, smoothed_mus3)
    assert np.allclose(smoothed_Sigmas1, smoothed_Sigmas3)
    assert np.allclose(ExxnT2, ExxnT3)

    # Plot_the samples vs the smoother
    xs = kalman_info_sample(*info_args)

def test_info_sample(T=100, D=3, N=10, U=3):
    args = make_lds_parameters(T, D, N, U)

    # Test the standard Kalman filter
    from pylds.lds_messages_interface import E_step as kalman_smoother_ref
    ll1, smoothed_mus1, smoothed_Sigmas1, ExnxT1 = kalman_smoother_ref(*args)

    # Plot_the samples vs the smoother
    info_args = convert_mean_to_info_args(*args)
    xs = [kalman_info_sample(*info_args) for _ in range(4)]

    import matplotlib.pyplot as plt
    for i in range(D):
        plt.subplot(D, 1, i+1)
        plt.fill_between(np.arange(T),
                         smoothed_mus1[:, i] - 2 * np.sqrt(smoothed_Sigmas1[:, i, i]),
                         smoothed_mus1[:, i] + 2 * np.sqrt(smoothed_Sigmas1[:, i, i]),
                         color='k', alpha=0.25)
        plt.plot(smoothed_mus1[:, i], '-k', lw=2)
        for x in xs:
            plt.plot(x[:, i])
    plt.show()

if __name__ == "__main__":
    test_lds()
    test_info_sample()
