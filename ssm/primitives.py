# Define an autograd extension for HMM normalizer
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.linalg import cholesky_banded, solve_banded, solveh_banded
from autograd.extend import primitive, defvjp
from autograd.tracer import getval
from functools import partial

from ssm.cstats import _blocks_to_bands_lower, _blocks_to_bands_upper, \
                       _bands_to_blocks_lower, _bands_to_blocks_upper, \
                       transpose_banded, vjp_cholesky_banded_lower

from ssm.messages import forward_pass, backward_pass, backward_sample, grad_hmm_normalizer

to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)

@primitive
def hmm_normalizer(log_pi0, log_Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))

    # Make sure everything is C contiguous
    log_pi0 = to_c(log_pi0)
    log_Ps = to_c(log_Ps)
    ll = to_c(ll)

    forward_pass(log_pi0, log_Ps, ll, alphas)    
    return logsumexp(alphas[-1])
    
def _make_grad_hmm_normalizer(argnum, ans, log_pi0, log_Ps, ll):
    # Make sure everything is C contiguous and unboxed
    log_pi0 = to_c(log_pi0)
    log_Ps = to_c(log_Ps)
    ll = to_c(ll)

    dlog_pi0 = np.zeros_like(log_pi0)
    dlog_Ps= np.zeros_like(log_Ps)
    dll = np.zeros_like(ll)
    T, K = ll.shape
    
    # Forward pass to get alphas
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)
    grad_hmm_normalizer(log_Ps, alphas, dlog_pi0, dlog_Ps, dll)
    
    if argnum == 0:
        return lambda g: g * dlog_pi0
    if argnum == 1:
        return lambda g: g * dlog_Ps
    if argnum == 2:
        return lambda g: g * dll

defvjp(hmm_normalizer, 
       partial(_make_grad_hmm_normalizer, 0),
       partial(_make_grad_hmm_normalizer, 1),
       partial(_make_grad_hmm_normalizer, 2))


def hmm_expected_states(log_pi0, log_Ps, ll):
    T, K = ll.shape

    # Make sure everything is C contiguous
    log_pi0 = to_c(log_pi0)
    log_Ps = to_c(log_Ps)
    ll = to_c(ll)

    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)
    normalizer = logsumexp(alphas[-1])

    betas = np.zeros((T, K))
    backward_pass(log_Ps, ll, betas)    

    expected_states = alphas + betas
    expected_states -= logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)
    
    expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = np.exp(expected_joints)
    expected_joints /= expected_joints.sum((1,2))[:,None,None]
    
    return expected_states, expected_joints, normalizer


def hmm_filter(log_pi0, log_Ps, ll):
    T, K = ll.shape

    # Make sure everything is C contiguous
    log_pi0 = to_c(log_pi0)
    log_Ps = to_c(log_Ps)
    ll = to_c(ll)

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)

    # Predict forward with the transition matrix
    pz_tt = np.exp(alphas - logsumexp(alphas, axis=1, keepdims=True))
    pz_tp1t = np.matmul(pz_tt[:-1,None,:], np.exp(log_Ps))[:,0,:]

    # Include the initial state distribution
    pz_tp1t = np.row_stack((np.exp(log_pi0 - logsumexp(log_pi0)), pz_tp1t))

    assert np.allclose(np.sum(pz_tp1t, axis=1), 1.0)
    return pz_tp1t


def hmm_sample(log_pi0, log_Ps, ll):
    T, K = ll.shape

    # Make sure everything is C contiguous
    log_pi0 = to_c(log_pi0)
    log_Ps = to_c(log_Ps)
    ll = to_c(ll)

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)

    # Sample backward
    us = npr.rand(T)
    zs = -1 * np.ones(T, dtype=int)
    backward_sample(log_Ps, ll, alphas, us, zs)
    return zs


def viterbi(log_pi0, log_Ps, ll):
    """
    Find the most likely state sequence

    This is modified from pyhsmm.internals.hmm_states
    by Matthew Johnson.
    """
    T, K = ll.shape
    
    # Pass max-sum messages backward
    scores = np.zeros_like(ll)
    args = np.zeros_like(ll, dtype=int)
    for t in range(T-2,-1,-1):
        vals = log_Ps[t] + scores[t+1] + ll[t+1]
        args[t+1] = vals.argmax(axis=1)
        scores[t] = vals.max(axis=1)

    # Now maximize forwards
    z = np.zeros(T, dtype=int)
    z[0] = (scores[0] + log_pi0 + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, z[t-1]]

    return z


# Helpers
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

    A_banded = to_c(A_banded)

    def vjp(C_bar):
        b_bar = solve_banded((u, l), transpose_banded((l, u), A_banded), C_bar)

        # Fill in the gradients of the banded matrix
        A_bar = np.zeros_like(A_banded)
        for d in range(D):
            for j in range(N):
                i = d + j - u
                if i >= 0 and i < N:
                    A_bar[d, j] = -np.dot(b_bar[i], C[j])
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

    D, N = A_banded.shape
    
    def vjp(C_bar):
        b_bar = solveh_banded(A_banded, C_bar, lower=lower, **kwargs)

        # Fill in the gradients of the banded matrix
        A_bar = np.zeros_like(A_banded)
        for j in range(N):
            for d in range(D):
                i = d + j if lower else d + j - u 
                if i < 0 or i >= N:
                    continue

                if i == j:
                    A_bar[d, j] = -np.dot(b_bar[i], C[j])

                else:
                    A_bar[d, j] = -(np.dot(b_bar[i], C[j]) + np.dot(b_bar[j], C[i]))

        # A_bar is the gradient of a full matrix.  A_banded is only one
        # half of the matrix (upper or lower).  We need to double the 
        # off diagonal entries since they would be counted twice
        return A_bar
    return vjp

defvjp(solveh_banded, vjp_solveh_banded_A, vjp_solveh_banded_b)

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
def solve_block_tridiag(Ad, Aod, b, lower=True):
    """
    Solve a block tridiagonal system Ax = b for x, where A is 
    block tridiagonal.
    """
    A_banded = blocks_to_bands(Ad, Aod, lower=lower)
    x_flat = solveh_banded(A_banded, np.ravel(b), lower=lower)
    return np.reshape(x_flat, b.shape)


def cholesky_block_tridiag(Ad, Aod, lower=True):
    """
    Compute the Cholesky decomposition of a block tridiagonal matrix.
    """
    A_banded = blocks_to_bands(Ad, Aod, lower=lower)
    return cholesky_banded(A_banded, lower=lower)
    

def logdet_block_tridiag(Ad, Aod, lower=True):
    """
    Compute the log determinant of a block tridiagonal matrix.
    """
    A_banded = blocks_to_bands(Ad, Aod, lower=lower)
    L = cholesky_banded(A_banded, lower=lower)

    # Get diagonal of Cholesky decomposition. Depends on lower or upper.
    diag = L[0] if lower else L[-1]

    # The log determinant of A is 2 * the sum of the diagonal
    return 2 * np.sum(np.log(diag))


def sample_block_tridiag(Ad, Aod, mu=0, lower=True, size=1, z=None):
    T, D, _ = Ad.shape
    A_upper_diag = Aod if not lower else np.swapaxes(Aod, 1, 2).copy("C")
    A_banded = blocks_to_bands(Ad, A_upper_diag, lower=False)
    U = cholesky_banded(A_banded, lower=False)

    # If lower = False, we have (U^T U)^{-1} = U^{-1} U^{-T} = AA^T = Sigma
    # where A = U^{-1}.  Samples are Az = U^{-1}z = x, or equivalently Ux = z.
    z = npr.randn(T*D, size) if z is None else z
    samples = np.reshape(solve_banded((0, 2*D-1), U, z).T, (size, T, D))

    # Add the mean
    samples += mu

    return samples


# LDS operations 
def _convert_lds_to_block_tridiag(As, Qinv_halves):
    T, D, _ = Qinv_halves.shape
    assert Qinv_halves.shape[2] == D
    assert As.shape == (T, D, D)

    # Construnct Q^{-1} matrices
    Qinvs = np.matmul(Qinv_halves, np.swapaxes(Qinv_halves, -1, -2))
    
    # Construct blocks of the block tridiagonal precision matrix
    nQinvAs = -np.matmul(Qinvs, As)
    J_lower_diag = nQinvAs[:-1]
    J_diag = -np.matmul(np.swapaxes(As, -1, -2), nQinvAs) + Qinvs
    return J_diag, J_lower_diag


def cholesky_lds(As, Qinv_halves):
    J_diag, J_lower_diag = _convert_lds_to_block_tridiag(As, Qinv_halves)
    return cholesky_block_tridiag(J_diag, J_lower_diag, lower=True)


def solve_lds(As, Qinv_halves, b):
    J_diag, J_lower_diag = _convert_lds_to_block_tridiag(As, Qinv_halves)
    return solve_block_tridiag(J_diag, J_lower_diag, b, lower=True)


def lds_normalizer(x, As, Qinv_halves, h):
    """
    Compute the log normalizer of a linear dynamical system with 
    natural parameters J and h.  J is a block tridiagonal matrix.

    The log normalizer is log p(x | J, h)
    """
    T, D = x.shape
    assert As.shape == (T, D, D)
    assert Qinv_halves.shape == (T, D, D)
    assert h.shape == (T, D)

    # Convert to block form
    J_diag, J_lower_diag = _convert_lds_to_block_tridiag(As, Qinv_halves)

    
    # -1/2 x^T J x = -1/2 \sum_{t=1}^T x_t.T J_tt x_t
    ll = -1/2 * np.sum(np.matmul(x[:, None, :], np.matmul(J_diag, x[:, :, None])))

    # -\sum_{t=1}^{T-1} x_t.T J_{t,t+1} x_{t+1}
    ll -= np.sum(np.matmul(x[1:, None, :], np.matmul(J_lower_diag, x[:-1, :, None])))

    # h^T x
    ll += np.sum(h * x)

    # -1/2 h^T J^{-1} h = -1/2 h^T (LL^T)^{-1} h
    #                   = -1/2 h^T L^{-T} L^{-1} h
    #                   = -1/2 (L^{-1}h)^T (L^{-1} h)
    L = cholesky_block_tridiag(J_diag, J_lower_diag, lower=True)
    Linv_h = solve_banded((2*D-1, 0), L, h.ravel())
    ll -= 1/2 * np.sum(Linv_h * Linv_h)

    # 1/2 log |J| -TD/2 log(2 pi) = log |L| -TD/2 log(2 pi)
    L_diag = L[0]
    ll += np.sum(np.log(L_diag))
    ll -= 1/2 * T * D * np.log(2 * np.pi)

    return ll
