# Define an autograd extension for HMM normalizer
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.extend import primitive, defvjp
from autograd.tracer import getval
from functools import partial

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

    