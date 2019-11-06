import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad, elementwise_grad
from autograd.test_util import check_grads

from ssm.messages import forward_pass, backward_pass, grad_hmm_normalizer
from ssm.primitives import hmm_normalizer

from tqdm import trange

def forward_pass_np(pi0, Ps, log_likes):
    T, K = log_likes.shape
    alphas = []
    alphas.append(log_likes[0] + np.log(pi0))
    for t in range(T-1):
        anext = logsumexp(alphas[t] + np.log(Ps[t]).T, axis=1)
        anext += log_likes[t+1]
        alphas.append(anext)
    return np.array(alphas)

def hmm_normalizer_np(pi0, Ps, ll):
    alphas = forward_pass_np(pi0, Ps, ll)
    Z = logsumexp(alphas[-1])
    return Z

def make_parameters(T, K):
    pi0 = np.ones(K) / K
    Ps = npr.rand(T-1, K, K)
    Ps /= Ps.sum(axis=2, keepdims=True)
    ll = npr.randn(T, K)
    return pi0, Ps, ll

def test_forward_pass(T=1000, K=3):
    pi0, Ps, ll = make_parameters(T, K)
    a1 = forward_pass_np(pi0, Ps, ll)
    a2 = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, a2)
    assert np.allclose(a1, a2)

def test_grad_hmm_normalizer(T=10, K=3):
    pi0, Ps, ll = make_parameters(T, K)
    dlogpi0, dlogPs, dll = np.zeros_like(pi0), np.zeros_like(Ps), np.zeros_like(ll)

    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)
    grad_hmm_normalizer(np.log(Ps), alphas, dlogpi0, dlogPs, dll)

    assert np.allclose(dlogpi0 / pi0, grad(hmm_normalizer_np, argnum=0)(pi0, Ps, ll))
    assert np.allclose(dlogPs / Ps, grad(hmm_normalizer_np, argnum=1)(pi0, Ps, ll))
    assert np.allclose(dll, grad(hmm_normalizer_np, argnum=2)(pi0, Ps, ll))

def test_hmm_normalizer_primitive(T=1000, K=3):
    # check reverse-mode to second order
    pi0, Ps, ll = make_parameters(T, K)
    check_grads(hmm_normalizer, argnum=1, modes=['rev'], order=1)(pi0, Ps, ll)


def test_backward_pass(T=1000, K=5, D=2):
    from pyhsmm.internals.hmm_messages_interface import messages_backwards_log

    # Make parameters
    As = npr.rand(K, K)
    As /= As.sum(axis=-1, keepdims=True)
    ll = npr.randn(T, K)

    # Use pyhsmm to compute
    true_betas = np.zeros((T, K))
    messages_backwards_log(As, ll, true_betas)

    # Use ssm to compute
    test_betas = np.zeros((T, K))
    backward_pass(As[None, :, :], ll, test_betas)

    assert np.allclose(true_betas, test_betas)
