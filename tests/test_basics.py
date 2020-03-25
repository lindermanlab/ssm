from time import time

import autograd.numpy as np
import autograd.numpy.random as npr
import scipy

import ssm


def test_sample(T=10, K=4, D=3, M=2):
    """
    Test that we can construct and sample an HMM
    with or withou, prefixes, noise, and noise.
    """
    transition_names = [
        "standard",
        "sticky",
        "constrained",
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


def test_constrained_hmm(T=100, K=3, D=3):
    hmm = ssm.HMM(K, D, M=0,
                  transitions="constrained",
                  observations="gaussian")
    z, x = hmm.sample(T)

    transition_mask = np.array([
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
    ]).astype(bool)
    init_Ps = np.random.rand(3, 3)
    init_Ps /= init_Ps.sum(axis=-1, keepdims=True)
    transition_kwargs = dict(
        transition_mask=transition_mask
    )
    fit_hmm = ssm.HMM(K, D, M=0,
                  transitions="constrained",
                  observations="gaussian",
                  transition_kwargs=transition_kwargs)
    fit_hmm.fit(x)
    learned_Ps = fit_hmm.transitions.transition_matrix
    assert np.all(learned_Ps[~transition_mask] == 0)


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


def test_trace_product():
    A = np.random.randn(100, 50, 10)
    B = np.random.randn(100, 10, 50)
    assert np.allclose(ssm.util.trace_product(A, B),
                       np.trace(A @ B, axis1=1, axis2=2))

    A = np.random.randn(50, 10)
    B = np.random.randn(10, 50)
    assert np.allclose(ssm.util.trace_product(A, B),
                       np.trace(A @ B))

    A = np.random.randn(1, 1)
    B = np.random.randn(1, 1)
    assert np.allclose(ssm.util.trace_product(A, B),
                       np.trace(A @ B))


def test_SLDSStructuredMeanField_entropy():
    """Test correctness of the entropy calculation for the
    SLDSStructuredMeanFieldVariationalPosterior class.

    """
    def entropy_mv_gaussian(J, h):
        mu = np.linalg.solve(J, h)
        sigma = np.linalg.inv(J)
        mv_normal = scipy.stats.multivariate_normal(mu, sigma)
        return mv_normal.entropy()

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
        ys = np.sin(2 * np.pi * np.arange(T) / 50)[:, None] * npr.randn(1, N) + 0.1 * npr.randn(T, N)

        return m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys

    def cumsum(v,strict=False):
        if not strict:
            return np.cumsum(v,axis=0)
        else:
            out = np.zeros_like(v)
            out[1:] = np.cumsum(v[:-1],axis=0)
            return out

    def bmat(blocks):
        rowsizes = [row[0].shape[0] for row in blocks]
        colsizes = [col[0].shape[1] for col in zip(*blocks)]
        rowstarts = cumsum(rowsizes,strict=True)
        colstarts = cumsum(colsizes,strict=True)

        nrows, ncols = sum(rowsizes), sum(colsizes)
        out = np.zeros((nrows,ncols))

        for i, (rstart, rsz) in enumerate(zip(rowstarts, rowsizes)):
            for j, (cstart, csz) in enumerate(zip(colstarts, colsizes)):
                out[rstart:rstart+rsz,cstart:cstart+csz] = blocks[i][j]
        return out

    def lds_to_dense_infoparams(params):
        m0, S0, As, Bs, Qs, Cs, Ds, Rs, us, ys = params
        mu_init = m0
        sigma_init = S0
        A, B, sigma_states = As, Bs, Qs
        C, D, sigma_obs = Cs, Ds, Rs
        data = ys
        inputs = us

        # Copied from PYLDS tests/test_dense.py
        T, n = data.shape[0], D.shape[0]

        # mu_init, sigma_init = model.mu_init, model.sigma_init
        # A, B, sigma_states = model.A, model.B,  model.sigma_states
        # C, D, sigma_obs = model.C, model.D, model.sigma_obs
        ss_inv = np.linalg.inv(sigma_states)

        h = np.zeros((T,n))
        h[0] += np.linalg.solve(sigma_init, mu_init)

        # Dynamics
        h[1:] += inputs[:-1].dot(B.T).dot(ss_inv)
        h[:-1] += -inputs[:-1].dot(B.T).dot(np.linalg.solve(sigma_states, A))

        # Emissions
        h += C.T.dot(np.linalg.solve(sigma_obs, data.T)).T
        h += -inputs.dot(D.T).dot(np.linalg.solve(sigma_obs, C))

        J = np.kron(np.eye(T),C.T.dot(np.linalg.solve(sigma_obs,C)))
        J[:n,:n] += np.linalg.inv(sigma_init)
        pairblock = bmat([[A.T.dot(ss_inv).dot(A), -A.T.dot(ss_inv)],
                        [-ss_inv.dot(A), ss_inv]])
        for t in range(0,n*(T-1),n):
            J[t:t+2*n,t:t+2*n] += pairblock

        return J.reshape(T*n,T*n), h.reshape(T*n)

    T, D, N, U = 100, 10, 10, 0
    params = make_lds_parameters(T, D, N, U)
    J_full, h_full = lds_to_dense_infoparams(params)

    ref_entropy = entropy_mv_gaussian(J_full, h_full)

    # Calculate entropy using kalman filter and posterior's entropy fn
    info_args = ssm.messages.convert_mean_to_info_args(*params)
    J_ini, h_ini, _, J_dyn_11,\
        J_dyn_21, J_dyn_22, h_dyn_1,\
        h_dyn_2, _, J_obs, h_obs, _ = info_args

    # J_obs[1:] += J_dyn_22
    # J_dyn_22[:] = 0
    log_Z, smoothed_mus, smoothed_Sigmas, ExxnT = ssm.messages.\
        kalman_info_smoother(*info_args)


    # Model is just a dummy model to simplify 
    # instantiating the posterior object.
    model = ssm.SLDS(N, 1, D, emissions="gaussian", dynamics="gaussian")
    datas = params[-1]
    post = ssm.variational.SLDSStructuredMeanFieldVariationalPosterior(model, datas)

    # Assign posterior to have info params that are the same as the ones used
    # in the reference entropy calculation.
    continuous_state_params = [dict(J_ini=J_ini,
                                    J_dyn_11=J_dyn_11,
                                    J_dyn_21=J_dyn_21,
                                    J_dyn_22=J_dyn_22,
                                    J_obs=J_obs,
                                    h_ini=h_ini,
                                    h_dyn_1=h_dyn_1,
                                    h_dyn_2=h_dyn_2,
                                    h_obs=h_obs)]
    post.continuous_state_params = continuous_state_params

    ssm_entropy = post._continuous_entropy()
    print("reference entropy: {}".format(ref_entropy))
    print("ssm_entropy: {}".format(ssm_entropy))
    assert np.allclose(ref_entropy, ssm_entropy)

if __name__ == "__main__":
    test_expectations()
    # test_hmm_likelihood_perf()
    # test_hmm_mp_perf()
    # test_constrained_hmm()
    # test_SLDSStructuredMeanField_entropy()
