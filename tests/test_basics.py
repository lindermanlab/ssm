from time import time

import autograd.numpy as np
import autograd.numpy.random as npr

import ssm


def test_sample(T=10, K=4, D=3, M=2):
    """
    Test that we can construct and sample an HMM
    with or withou, prefixes, noise, and noise.
    """
    transition_names = [
        "standard",
        "sticky",
        "inputdriven",
        "recurrent",
        "recurrent_only",
        "rbf_recurrent",
        "nn_recurrent"
    ]

    observation_names = [
        "gaussian",
        "diagonal_gaussian",
        "t",
        "diagonal_t",
        "exponential",
        "bernoulli",
        "categorical",
        "poisson",
        "vonmises",
        "ar",
        "no_input_ar",
        "diagonal_ar",
        "independent_ar",
        "robust_ar",
        "no_input_robust_ar",
        "diagonal_robust_ar"
    ]

    # Sample basic (no prefix, inputs, etc.)
    for transitions in transition_names:
        for observations in observation_names:
            hmm = ssm.HMM(K, D, M=0, transitions=transitions, observations=observations)
            zsmpl, xsmpl = hmm.sample(T)

    # Sample with prefix
    for transitions in transition_names:
        for observations in observation_names:
            hmm = ssm.HMM(K, D, M=0, transitions=transitions, observations=observations)
            zpre, xpre = hmm.sample(3)
            zsmpl, xsmpl = hmm.sample(T, prefix=(zpre, xpre))

    # Sample with inputs
    for transitions in transition_names:
        for observations in observation_names:
            hmm = ssm.HMM(K, D, M=M, transitions=transitions, observations=observations)
            zpre, xpre = hmm.sample(3, input=npr.randn(3, M))
            zsmpl, xsmpl = hmm.sample(T, prefix=(zpre, xpre), input=npr.randn(T, M))

    # Sample without noise
    for transitions in transition_names:
        for observations in observation_names:
            hmm = ssm.HMM(K, D, M=M, transitions=transitions, observations=observations)
            zpre, xpre = hmm.sample(3, input=npr.randn(3, M))
            zsmpl, xsmpl = hmm.sample(T, prefix=(zpre, xpre), input=npr.randn(T, M), with_noise=False)


def test_hmm_likelihood(T=1000, K=5, D=2):
    # Create a true HMM
    A = npr.rand(K, K)
    A /= A.sum(axis=1, keepdims=True)
    A = 0.75 * np.eye(K) + 0.25 * A
    C = npr.randn(K, D)
    sigma = 0.01

    # Sample from the true HMM
    z = np.zeros(T, dtype=int)
    y = np.zeros((T, D))
    for t in range(T):
        if t > 0:
            z[t] = np.random.choice(K, p=A[z[t-1]])
        y[t] = C[z[t]] + np.sqrt(sigma) * npr.randn(D)

    # Compare to pyhsmm answer
    from pyhsmm.models import HMM as OldHMM
    from pybasicbayes.distributions import Gaussian
    oldhmm = OldHMM([Gaussian(mu=C[k], sigma=sigma * np.eye(D)) for k in range(K)],
                  trans_matrix=A,
                  init_state_distn="uniform")
    true_lkhd = oldhmm.log_likelihood(y)

    # Make an HMM with these parameters
    hmm = ssm.HMM(K, D, observations="diagonal_gaussian")
    hmm.transitions.log_Ps = np.log(A)
    hmm.observations.mus = C
    hmm.observations.sigmasq = sigma * np.ones((K, D))
    test_lkhd = hmm.log_probability(y)

    assert np.allclose(true_lkhd, test_lkhd)


def test_big_hmm_likelihood(T=50000, K=50, D=50):
    test_hmm_likelihood(T=T, K=K, D=D)


def test_expectations(T=1000, K=20, D=2):
    # Create a true HMM
    A = npr.rand(K, K)
    A /= A.sum(axis=1, keepdims=True)
    A = 0.75 * np.eye(K) + 0.25 * A
    C = npr.randn(K, D)
    sigma = 0.01

    # Sample from the true HMM
    z = np.zeros(T, dtype=int)
    y = np.zeros((T, D))
    for t in range(T):
        if t > 0:
            z[t] = np.random.choice(K, p=A[z[t-1]])
        y[t] = C[z[t]] + np.sqrt(sigma) * npr.randn(D)

    # Compare to pyhsmm answer
    from pyhsmm.models import HMM as OldHMM
    from pyhsmm.basic.distributions import Gaussian
    oldhmm = OldHMM([Gaussian(mu=C[k], sigma=sigma * np.eye(D)) for k in range(K)],
                  trans_matrix=A,
                  init_state_distn="uniform")
    oldhmm.add_data(y)
    states = oldhmm.states_list.pop()
    states.E_step()
    true_Ez = states.expected_states
    true_E_trans = states.expected_transcounts

    # Make an HMM with these parameters
    hmm = ssm.HMM(K, D, observations="diagonal_gaussian")
    hmm.transitions.log_Ps = np.log(A)
    hmm.observations.mus = C
    hmm.observations.sigmasq = sigma * np.ones((K, D))
    test_Ez, test_Ezzp1, _ = hmm.expected_states(y)
    test_E_trans = test_Ezzp1.sum(0)

    print(true_E_trans.round(3))
    print(test_E_trans.round(3))

    assert np.allclose(true_Ez, test_Ez)
    assert np.allclose(true_E_trans, test_E_trans)


def test_viterbi(T=1000, K=20, D=2):
    # Create a true HMM
    A = npr.rand(K, K)
    A /= A.sum(axis=1, keepdims=True)
    A = 0.75 * np.eye(K) + 0.25 * A
    C = npr.randn(K, D)
    sigma = 0.01

    # Sample from the true HMM
    z = np.zeros(T, dtype=int)
    y = np.zeros((T, D))
    for t in range(T):
        if t > 0:
            z[t] = np.random.choice(K, p=A[z[t-1]])
        y[t] = C[z[t]] + np.sqrt(sigma) * npr.randn(D)

    # Compare to pyhsmm answer
    from pyhsmm.models import HMM as OldHMM
    from pyhsmm.basic.distributions import Gaussian
    oldhmm = OldHMM([Gaussian(mu=C[k], sigma=sigma * np.eye(D)) for k in range(K)],
                  trans_matrix=A,
                  init_state_distn="uniform")
    oldhmm.add_data(y)
    states = oldhmm.states_list.pop()
    states.Viterbi()
    z_star = states.stateseq

    # Make an HMM with these parameters
    hmm = ssm.HMM(K, D, observations="diagonal_gaussian")
    hmm.transitions.log_Ps = np.log(A)
    hmm.observations.mus = C
    hmm.observations.sigmasq = sigma * np.ones((K, D))
    z_star2 = hmm.most_likely_states(y)

    assert np.allclose(z_star, z_star2)


def test_hmm_mp_perf(T=10000, K=100, D=20):
    # Make parameters
    pi0 = np.ones(K) / K
    Ps = npr.rand(T-1, K, K)
    Ps /= Ps.sum(axis=2, keepdims=True)
    ll = npr.randn(T, K)
    out1 = np.zeros((T, K))
    out2 = np.zeros((T, K))

    # Run the PyHSMM message passing code
    from pyhsmm.internals.hmm_messages_interface import messages_forwards_log, messages_backwards_log
    tic = time()
    messages_forwards_log(Ps, ll, pi0, out1)
    pyhsmm_dt = time() - tic
    print("PyHSMM Fwd: ", pyhsmm_dt, "sec")

    # Run the SSM message passing code
    from ssm.messages import forward_pass, backward_pass
    forward_pass(pi0, Ps, ll, out2) # Call once to compile, then time it
    tic = time()
    forward_pass(pi0, Ps, ll, out2)
    smm_dt = time() - tic
    print("SMM Fwd: ", smm_dt, "sec")
    assert np.allclose(out1, out2)

    # Backward pass
    tic = time()
    messages_backwards_log(Ps, ll, out1)
    pyhsmm_dt = time() - tic
    print("PyHSMM Bwd: ", pyhsmm_dt, "sec")

    backward_pass(Ps, ll, out2) # Call once to compile, then time it
    tic = time()
    backward_pass(Ps, ll, out2)
    smm_dt = time() - tic
    print("SMM (Numba) Bwd: ", smm_dt, "sec")
    assert np.allclose(out1, out2)


def test_hmm_likelihood_perf(T=10000, K=50, D=20):
    # Create a true HMM
    A = npr.rand(K, K)
    A /= A.sum(axis=1, keepdims=True)
    A = 0.75 * np.eye(K) + 0.25 * A
    C = npr.randn(K, D)
    sigma = 0.01

    # Sample from the true HMM
    z = np.zeros(T, dtype=int)
    y = np.zeros((T, D))
    for t in range(T):
        if t > 0:
            z[t] = np.random.choice(K, p=A[z[t-1]])
        y[t] = C[z[t]] + np.sqrt(sigma) * npr.randn(D)

    # Compare to pyhsmm answer
    from pyhsmm.models import HMM as OldHMM
    from pybasicbayes.distributions import Gaussian
    oldhmm = OldHMM([Gaussian(mu=C[k], sigma=sigma * np.eye(D)) for k in range(K)],
                  trans_matrix=A,
                  init_state_distn="uniform")

    states = oldhmm.add_data(y)
    tic = time()
    true_lkhd = states.log_likelihood()
    pyhsmm_dt = time() - tic
    print("PyHSMM: ", pyhsmm_dt, "sec. Val: ", true_lkhd)

    # Make an HMM with these parameters
    hmm = ssm.HMM(K, D, observations="gaussian")
    hmm.transitions.log_Ps = np.log(A)
    hmm.observations.mus = C
    hmm.observations._sqrt_Sigmas = np.sqrt(sigma) * np.array([np.eye(D) for k in range(K)])

    tic = time()
    test_lkhd = hmm.log_probability(y)
    smm_dt = time() - tic
    print("SMM HMM: ", smm_dt, "sec. Val: ", test_lkhd)

    # Make an ARHMM with these parameters
    arhmm = ssm.HMM(K, D, observations="ar")
    tic = time()
    arhmm.log_probability(y)
    arhmm_dt = time() - tic
    print("SSM ARHMM: ", arhmm_dt, "sec.")

    # Make an ARHMM with these parameters
    arhmm = ssm.HMM(K, D, observations="ar")
    tic = time()
    arhmm.expected_states(y)
    arhmm_dt = time() - tic
    print("SSM ARHMM Expectations: ", arhmm_dt, "sec.")


if __name__ == "__main__":
    # test_hmm_likelihood_perf()
    test_hmm_mp_perf()
