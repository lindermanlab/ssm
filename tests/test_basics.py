import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.models import HMM


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
            hmm = HMM(K, D, M=0, transitions=transitions, observations=observations)
            zsmpl, xsmpl = hmm.sample(T)

    # Sample with prefix
    for transitions in transition_names:
        for observations in observation_names:
            hmm = HMM(K, D, M=0, transitions=transitions, observations=observations)
            zpre, xpre = hmm.sample(3)
            zsmpl, xsmpl = hmm.sample(T, prefix=(zpre, xpre))

    # Sample with inputs
    for transitions in transition_names:
        for observations in observation_names:
            hmm = HMM(K, D, M=M, transitions=transitions, observations=observations)
            zpre, xpre = hmm.sample(3, input=npr.randn(3, M))
            zsmpl, xsmpl = hmm.sample(T, prefix=(zpre, xpre), input=npr.randn(T, M))

    # Sample without noise
    for transitions in transition_names:
        for observations in observation_names:
            hmm = HMM(K, D, M=M, transitions=transitions, observations=observations)
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
    hmm = HMM(K, D, observations="diagonal_gaussian")
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
    hmm = HMM(K, D, observations="diagonal_gaussian")
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
    hmm = HMM(K, D, observations="diagonal_gaussian")
    hmm.transitions.log_Ps = np.log(A)
    hmm.observations.mus = C
    hmm.observations.sigmasq = sigma * np.ones((K, D))
    z_star2 = hmm.most_likely_states(y)

    assert np.allclose(z_star, z_star2)
