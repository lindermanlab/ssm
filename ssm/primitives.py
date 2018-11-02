# Define an autograd extension for HMM normalizer
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.linalg import cholesky_banded, solve_banded, solveh_banded
from autograd.extend import primitive, defvjp
from autograd.tracer import getval
from functools import partial

from ssm.cstats import blocks_to_bands
from ssm.messages import forward_pass, backward_pass, backward_sample, grad_hmm_normalizer

@primitive
def hmm_normalizer(log_pi0, log_Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))

    # Make sure everything is C contiguous
    to_c = lambda arr: np.copy(arr, 'C') if not arr.flags['C_CONTIGUOUS'] else arr
    log_pi0 = to_c(log_pi0)
    log_Ps = to_c(log_Ps)
    ll = to_c(ll)

    forward_pass(log_pi0, log_Ps, ll, alphas)    
    return logsumexp(alphas[-1])
    
def _make_grad_hmm_normalizer(argnum, ans, log_pi0, log_Ps, ll):
    # Unbox the inputs if necessary
    log_pi0 = getval(log_pi0)
    log_Ps = getval(log_Ps)
    ll = getval(ll)

    # Make sure everything is C contiguous
    to_c = lambda arr: np.copy(arr, 'C') if not arr.flags['C_CONTIGUOUS'] else arr
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
    to_c = lambda arr: np.copy(arr, 'C') if not arr.flags['C_CONTIGUOUS'] else arr
    log_pi0 = to_c(getval(log_pi0))
    log_Ps = to_c(getval(log_Ps))
    ll = to_c(getval(ll))

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
    to_c = lambda arr: np.copy(arr, 'C') if not arr.flags['C_CONTIGUOUS'] else arr
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
    to_c = lambda arr: np.copy(arr, 'C') if not arr.flags['C_CONTIGUOUS'] else arr
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


""" 
Block tridiagonal system helpers:

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
    x_flat = solveh_banded(A_banded, b.ravel(), lower=lower)
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

def lds_normalizer(x, J_diag, J_offdiag, h, lower=True):
    """
    Compute the log normalizer of a linear dynamical system with 
    natural parameters J and h.  J is a block tridiagonal matrix.

    The log normalizer is log p(x | J, h)
    """
    T, D = x.shape
    assert J_diag.shape == (T, D, D)
    assert J_offdiag.shape == (T-1, D, D)
    assert h.shape == (T, D)

    # -1/2 x^T J x = -1/2 \sum_{t=1}^T x_t.T J_tt x_t
    ll = -1/2 * np.sum(np.matmul(x[:, None, :], np.matmul(J_diag, x[:, :, None])))

    # -\sum_{t=1}^{T-1} x_t.T J_{t,t+1} x_{t+1}
    if lower:
        ll -= np.sum(np.matmul(x[1:, None, :], np.matmul(J_offdiag, x[:-1, :, None])))
    else:
        ll -= np.sum(np.matmul(x[:-1, None, :], np.matmul(J_offdiag, x[1:, :, None])))

    # h^T x
    ll += np.sum(h * x)

    #   -1/2 h^T J^{-1} h = -1/2 h^T (LL^T)^{-1} h
    #                     = -1/2 h^T L^{-T} L^{-1} h
    #                     = -1/2 (L^{-1}h)^T (L^{-1} h)
    L = cholesky_block_tridiag(J_diag, J_offdiag, lower=lower)
    bounds = (2*D-1, 0) if lower else (0, 2*D-1)
    Linv_h = solve_banded(bounds, L, h.ravel())
    ll -= 1/2 * np.sum(Linv_h * Linv_h)

    # 1/2 log |J| -TD/2 log(2 pi) = log |L| -TD/2 log(2 pi)
    L_diag = L[0] if lower else L[-1]
    ll += np.sum(np.log(L_diag))
    ll -= 1/2 * T * D * np.log(2 * np.pi)

    return ll
