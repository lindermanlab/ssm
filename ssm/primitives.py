# Define an autograd extension for HMM normalizer
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.extend import primitive, defvjp
from functools import partial

from ssm.messages import forward_pass, backward_pass, grad_hmm_normalizer

@primitive
def hmm_normalizer(log_pi0, log_Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)    
    return logsumexp(alphas[-1])
    
def _make_grad_hmm_normalizer(argnum, ans, log_pi0, log_Ps, ll):
    # Unbox the inputs if necessary
    unbox = lambda x: x if isinstance(x, np.ndarray) else x._value
    log_pi0 = unbox(log_pi0)
    log_Ps = unbox(log_Ps)
    ll = unbox(ll)

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
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)    
    betas = np.zeros((T, K))
    backward_pass(log_Ps, ll, betas)    

    expected_states = alphas + betas
    expected_states -= logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)
    
    expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = np.exp(expected_joints)
    
    return expected_states, expected_joints


def hmm_filter(log_pi0, log_Ps, ll):
    T, K = ll.shape
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