# Define an autograd extension for HMM normalizer
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.linalg import cholesky_banded, solve_banded, solveh_banded
from autograd.extend import primitive, defvjp
from autograd.tracer import getval
from functools import partial

from ssm.cstats import _blocks_to_bands_lower, _blocks_to_bands_upper, \
                       _bands_to_blocks_lower, _bands_to_blocks_upper, \
                       _transpose_banded, vjp_cholesky_banded_lower, \
                       _vjp_solve_banded_A, _vjp_solveh_banded_A

from ssm.messages import forward_pass, grad_hmm_normalizer
from ssm.util import LOG_EPS, DIV_EPS

to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)

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
    log_Ps = np.log(Ps + LOG_EPS) - logsumexp(Ps, axis=1, keepdims=True)
    grad_hmm_normalizer(log_Ps, alphas, dlog_pi0, dlog_Ps, dll)

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

"""
Block tridiagonal system operations:

The following functions work on matrices of the form:

    A = [Ad[0],    Aod[0],   0,      ...                    ]
        [Aod[0].T, Ad[1],    Aod[1], 0,      ...            ]
        [0,        Aod[1].T, Ad[2],  Aod[2], 0,    ...      ]
        [          ...       ...     ...     ...   ...      ]
        [                            ...   Ad[T-1], Aod[T-1]]
        [                                  Aod[T-1].T, Ad[T]]

This is a banded Hermitian matrix, and scipy.linalg has fast
solvers for such systems. The result is itself a banded matrix.

The precision matrix of a linear dynamical system has exactly
this form.
"""
def bands_to_blocks(A_banded, lower=True):
    """
    Convert a banded matrix to a block tridiagonal matrix.

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    assert A_banded.ndim == 2
    A_banded = to_c(A_banded)
    return _bands_to_blocks_lower(A_banded) if lower else _bands_to_blocks_upper(A_banded)


@primitive
def blocks_to_bands(Ad, Aod, lower=True):
    """
    Convert a block tridiagonal matrix to the banded matrix representation
    required for scipy banded solvers.

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    assert Ad.ndim == 3
    assert Ad.shape[2] == Ad.shape[1]
    assert Aod.ndim == 3
    assert Aod.shape[0] == Ad.shape[0]-1
    assert Aod.shape[1] == Ad.shape[1]
    assert Aod.shape[2] == Ad.shape[1]

    # Make sure its a numpy array
    Ad, Aod = to_c(Ad), to_c(Aod)
    return _blocks_to_bands_lower(Ad, Aod) if lower else _blocks_to_bands_upper(Ad, Aod)

# Gradient of blocks_to_bands is simply a reshaping
def _make_grad_blocks_to_bands(argnum, bands, Ad, Aod, lower=True):
    return lambda g: bands_to_blocks(g, lower=lower)[argnum]


defvjp(blocks_to_bands,
       partial(_make_grad_blocks_to_bands, 0),
       partial(_make_grad_blocks_to_bands, 1))


@primitive
def transpose_banded(l_and_u, A_banded):
    A_banded = to_c(A_banded)
    return _transpose_banded(l_and_u[0], l_and_u[1], A_banded)

def grad_transpose_banded(ans, l_and_u, A_banded):
    l, u = l_and_u
    def vjp(g):
        return transpose_banded((u, l), g)
    return vjp

defvjp(transpose_banded, None, grad_transpose_banded)


# Gradient of cholesky_banded
def grad_cholesky_banded(L_banded, A_banded, lower=True):
    assert lower, "Only implemented lower form so far. Need to do some \
                   algebra to work out the gradient of the upper form."

    assert L_banded.shape == A_banded.shape

    L_banded = to_c(L_banded)
    A_banded = to_c(A_banded)

    def vjp(g):
        # Compute the gradient in cython.  Copy g since it will be overwritten.
        L_bar = to_c(g).copy()
        A_bar = np.zeros_like(A_banded)
        vjp_cholesky_banded_lower(L_bar, L_banded, A_banded, A_bar)
        return A_bar

    return vjp

defvjp(cholesky_banded, grad_cholesky_banded)


# Gradient of solve_banded
def vjp_solve_banded_b(C, l_and_u, A_banded, b, **kwargs):
    # \bar{b} = A^{-T} \bar{C}
    l, u = l_and_u
    A_banded = to_c(A_banded)

    def vjp(C_bar):
        return solve_banded((u, l), transpose_banded((l, u), A_banded), C_bar)
    return vjp

def vjp_solve_banded_A(C, l_and_u, A_banded, b, **kwargs):
    # \bar{A} = -A^{-T} \bar{C} C^T  = -\bar{B} C^T
    l, u = l_and_u
    D, N = A_banded.shape
    assert D == l + u + 1

    C = to_c(C)
    A_banded = to_c(A_banded)
    b = to_c(b)

    def vjp(C_bar):
        b_bar = solve_banded((u, l), transpose_banded((l, u), A_banded), C_bar)
        A_bar = np.zeros_like(A_banded)
        K = b.shape[1] if b.ndim == 2 else 1
        _vjp_solve_banded_A(A_bar,
                            b_bar.reshape(-1, K),
                            C_bar.reshape(-1, K),
                            C.reshape(-1, K),
                            u, A_banded)
        return A_bar
    return vjp


defvjp(solve_banded, None, vjp_solve_banded_A, vjp_solve_banded_b)


# Gradient of solveh_banded for symmetric matrices
def vjp_solveh_banded_b(C, A_banded, b, lower=True, **kwargs):
    # \bar{b} = A^{-T} \bar{C}
    def vjp(C_bar):
        return solveh_banded(A_banded, C_bar, lower=lower, **kwargs)
    return vjp

def vjp_solveh_banded_A(C, A_banded, b, lower=True, **kwargs):
    # \bar{A} = -A^{-T} \bar{C} C^T  = -\bar{B} C^T
    C = to_c(C)
    A_banded = to_c(A_banded)
    b = to_c(b)

    def vjp(C_bar):
        b_bar = solveh_banded(A_banded, C_bar, lower=lower, **kwargs)
        A_bar = np.zeros_like(A_banded)
        K = b.shape[1] if b.ndim == 2 else 1
        _vjp_solveh_banded_A(A_bar,
                             b_bar.reshape(-1, K),
                             C_bar.reshape(-1, K),
                             C.reshape(-1, K),
                             lower,
                             A_banded)

        return A_bar
    return vjp

defvjp(solveh_banded, vjp_solveh_banded_A, vjp_solveh_banded_b)


def blocks_to_full(J_diag, J_lower_diag):
    T, D, _ = J_diag.shape
    J = np.zeros((T*D, T*D))
    for t in range(T):
        J[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag[t]
    for t in range(T-1):
        J[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag[t]
        J[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag[t].T
    return J


# Solve and multiply symmetric block tridiagonal systems
def solve_symm_block_tridiag(J_diag, J_lower_diag, v):
    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
    x_flat = solveh_banded(J_banded, np.ravel(v), lower=True)
    return np.reshape(x_flat, v.shape)


def symm_block_tridiag_matmul(J_diag, J_lower_diag, v):
    """
    Compute matrix-vector product with a symmetric block
    tridiagonal matrix J and vector v.
    :param J_diag:          block diagonal terms of J
    :param J_lower_diag:    lower block diagonal terms of J
    :param v:               vector to multiply
    :return:                J * v
    """
    T, D, _ = J_diag.shape
    assert J_diag.ndim == 3 and J_diag.shape[2] == D
    assert J_lower_diag.shape == (T-1, D, D)
    assert v.shape == (T, D)

    out = np.matmul(J_diag, v[:, :, None])[:, :, 0]
    out[:-1] += np.matmul(np.swapaxes(J_lower_diag, -1, -2), v[1:][:, :, None])[:, :, 0]
    out[1:] += np.matmul(J_lower_diag, v[:-1][:, :, None])[:, :, 0]
    return out


# LDS operations
def convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts):
    """
    Parameterize the LDS in terms of pairwise linear Gaussian dynamics
    and per-timestep Gaussian observations.

        p(x_{1:T}; theta)
            = [prod_{t=1}^{T-1} N(x_{t+1} | A_t x_t + b_t, Q_t)]
                * [prod_{t=1}^T N(x_t | m_t, R_t)]

    We can rewrite this as a Gaussian with a block tridiagonal precision
    matrix J.  The blocks of this matrix are:

    J_{t,t} = A_t.T Q_t^{-1} A_t + Q_{t-1}^{-1} + R_t^{-1}

    J_{t,t+1} = -Q_t^{-1} A_t

    The linear term is h_t

    h_t = -A_t.T Q_t^{-1} b_t + Q_{t-1}^{-1} b_{t-1} + R_t^{-1} m_t

    We parameterize the model in terms of

    theta = {A_t, b_t, Q_t^{-1/2}}_{t=1}^{T-1},  {m_t, R_t^{-1/2}}_{t=1}^T
    """
    T, D = ms.shape
    assert As.shape == (T-1, D, D)
    assert bs.shape == (T-1, D)
    assert Qi_sqrts.shape == (T-1, D, D)
    assert Ri_sqrts.shape == (T, D, D)

    # Construnct the inverse covariance matrices
    Qis = np.matmul(Qi_sqrts, np.swapaxes(Qi_sqrts, -1, -2))
    Ris = np.matmul(Ri_sqrts, np.swapaxes(Ri_sqrts, -1, -2))

    # Construct the joint, block-tridiagonal precision matrix
    J_lower_diag = -np.matmul(Qis, As)
    J_diag = np.concatenate([-np.matmul(np.swapaxes(As, -1, -2), J_lower_diag), np.zeros((1, D, D))]) \
           + np.concatenate([np.zeros((1, D, D)), Qis]) \
           + Ris

    # Construct the linear term
    h = np.concatenate([np.matmul(J_lower_diag, bs[:, :, None])[:, :, 0], np.zeros((1, D))]) \
      + np.concatenate([np.zeros((1, D)), np.matmul(Qis, bs[:, :, None])[:, :, 0]]) \
      + np.matmul(Ris, ms[:, :, None])[:, :, 0]

    return J_diag, J_lower_diag, h


def cholesky_lds(As, bs, Qi_sqrts, ms, Ri_sqrts):
    J_diag, J_lower_diag, _ = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)
    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
    return cholesky_banded(J_banded, lower=True)


def solve_lds(As, bs, Qi_sqrts, ms, Ri_sqrts, v):
    J_diag, J_lower_diag, _ = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)
    return solve_symm_block_tridiag(J_diag, J_lower_diag, v)


def lds_log_probability(x, As, bs, Qi_sqrts, ms, Ri_sqrts):
    """
    Compute the log normalizer of a linear dynamical system.
    """
    T, D = x.shape
    assert As.shape == (T-1, D, D)
    assert bs.shape == (T-1, D)
    assert Qi_sqrts.shape == (T-1, D, D)
    assert ms.shape == (T, D)
    assert Ri_sqrts.shape == (T, D, D)

    return block_tridiagonal_log_probability(x,
            *convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts))

def block_tridiagonal_log_probability(x, J_diag, J_lower_diag, h):

    T, D = x.shape
    assert h.shape == (T, D)
    assert J_diag.shape == (T, D, D)
    assert J_lower_diag.shape == (T-1, D, D)

    # Convert blocks to banded form so we can capitalize on Lapack code
    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)

    # -1/2 x^T J x = -1/2 \sum_{t=1}^T x_t.T J_tt x_t
    ll = -1/2 * np.sum(np.matmul(x[:, None, :], np.matmul(J_diag, x[:, :, None])))

    # -\sum_{t=1}^{T-1} x_t.T J_{t,t+1} x_{t+1}
    ll -= np.sum(np.matmul(x[1:, None, :], np.matmul(J_lower_diag, x[:-1, :, None])))

    # h^T x
    ll += np.sum(h * x)

    # -1/2 h^T J^{-1} h = -1/2 h^T (LL^T)^{-1} h
    #                   = -1/2 h^T L^{-T} L^{-1} h
    #                   = -1/2 (L^{-1}h)^T (L^{-1} h)
    # L = cholesky_block_tridiag(J_diag, J_lower_diag, lower=True)
    L = cholesky_banded(J_banded, lower=True)
    Linv_h = solve_banded((2*D-1, 0), L, h.ravel())
    ll -= 1/2 * np.sum(Linv_h * Linv_h)

    # 1/2 log |J| -TD/2 log(2 pi) = log |L| -TD/2 log(2 pi)
    L_diag = L[0]
    ll += np.sum(np.log(L_diag))
    ll -= 1/2 * T * D * np.log(2 * np.pi)

    return ll


def lds_sample(As, bs, Qi_sqrts, ms, Ri_sqrts, z=None):
    """
    Sample a linear dynamical system
    """
    T, D = ms.shape
    assert As.shape == (T-1, D, D)
    assert bs.shape == (T-1, D)
    assert Qi_sqrts.shape == (T-1, D, D)
    assert Ri_sqrts.shape == (T, D, D)

    return block_tridiagonal_sample(
        *convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts), z=z)


def block_tridiagonal_sample(J_diag, J_lower_diag, h, z=None):
    """
    Sample a Gaussian chain graph represented by a block
    tridiagonal precision matrix and a linear potential.
    """
    T, D = h.shape
    assert J_diag.shape == (T, D, D)
    assert J_lower_diag.shape == (T-1, D, D)

    # Convert blocks to banded form so we can capitalize on Lapack code
    J_banded = A_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
    L = cholesky_banded(J_banded, lower=True)
    U = transpose_banded((2*D-1, 0), L)

    # We have (U^T U)^{-1} = U^{-1} U^{-T} = AA^T = Sigma
    # where A = U^{-1}.  Samples are Az = U^{-1}z = x, or equivalently Ux = z.
    z = npr.randn(T*D,) if z is None else np.reshape(z, (T*D,))
    samples = np.reshape(solve_banded((0, 2*D-1), U, z), (T, D))

    # Get the mean mu = J^{-1} h
    mu = np.reshape(solveh_banded(J_banded, np.ravel(h), lower=True), (T, D))

    # Add the mean
    return samples + mu


def lds_mean(As, bs, Qi_sqrts, ms, Ri_sqrts):
    """
    Compute the posterior mean of the linear dynamical system
    """
    T, D = ms.shape
    assert As.shape == (T-1, D, D)
    assert bs.shape == (T-1, D)
    assert Qi_sqrts.shape == (T-1, D, D)
    assert Ri_sqrts.shape == (T, D, D)

    # Convert to block form
    return block_tridiagonal_mean(
        *convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts), lower=True).reshape((T,D))


def block_tridiagonal_mean(J_diag, J_lower_diag, h, lower=True):
    # Convert blocks to banded form so we can capitalize on Lapack code
    return solveh_banded(
        blocks_to_bands(J_diag, J_lower_diag, lower=lower), h.ravel(), lower=lower)
