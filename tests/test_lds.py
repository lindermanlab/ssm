import time
import ssm
from ssm.util import SEED
import copy
import scipy
import itertools as it

from autograd import elementwise_grad
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import hessian

from ssm.primitives import \
    blocks_to_bands, bands_to_blocks, transpose_banded, \
    solveh_banded, solve_banded, convert_lds_to_block_tridiag, \
    lds_log_probability, grad_cholesky_banded, cholesky_banded, \
    cholesky_lds, solve_lds, lds_sample, lds_mean, \
    convert_lds_to_block_tridiag


def make_lds_parameters(T=20, D=2):
    As = npr.randn(T-1, D, D)
    bs = npr.randn(T-1, D)
    Qi_sqrts = npr.randn(T-1, D, D)
    ms = npr.randn(T, D)
    Ri_sqrts = npr.randn(T, D, D)
    return As, bs, Qi_sqrts, ms, Ri_sqrts


def block_to_full(T, J_diag, J_lower_diag):
    D, _  = J_diag.shape

    # Solve the dense way
    J_full = np.zeros((T*D, T*D))
    for t in range(T):
        J_full[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag

    for t in range(T-1):
        J_full[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag.T
        J_full[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag

    return J_full


def make_block_tridiag(T, D):
    A = npr.randn(D, D)
    Q = np.eye(D)

    J_diag = np.linalg.inv(Q) + A.T.dot(np.linalg.solve(Q, A))
    J_lower_diag = -np.linalg.solve(Q, A)
    J_full = block_to_full(T, J_diag, J_lower_diag)

    return J_diag, J_lower_diag, J_full


def test_blocks_to_banded(T=5, D=3):
    """
    Test blocks_to_banded correctness
    """
    Ad = np.zeros((T, D, D))
    Aod = np.zeros((T-1, D, D))

    M = np.arange(1, D+1)[:, None] * 10 + np.arange(1, D+1)
    for t in range(T):
        Ad[t, :, :] = 100 * ((t+1)*10 + (t+1)) + M

    for t in range(T-1):
        Aod[t, :, :] = 100 * ((t+2)*10 + (t+1)) + M

    # print("Lower")
    # L = blocks_to_bands(Ad, Aod, lower=True)
    # print(L)

    # print("Upper")
    # U = blocks_to_bands(Ad, Aod, lower=False)
    # print(U)

    # Check inverse with random symmetric matrices
    Ad = npr.randn(T, D, D)
    Ad = (Ad + np.swapaxes(Ad, -1, -2)) / 2
    Aod = npr.randn(T-1, D, D)

    Ad2, Aod2 = bands_to_blocks(blocks_to_bands(Ad, Aod, lower=True), lower=True)
    assert np.allclose(np.tril(Ad), np.tril(Ad2))
    assert np.allclose(Aod, Aod2)

    Ad3, Aod3 = bands_to_blocks(blocks_to_bands(Ad, Aod, lower=False), lower=False)
    assert np.allclose(np.triu(Ad), np.triu(Ad3))
    assert np.allclose(Aod, Aod3)


def test_transpose_banded():
    """
    Test transpose_banded correctness
    """
    l, u = 1, 2
    ab = np.array([[0,  0, -1, -1, -1],
                   [0,  2,  2,  2,  2],
                   [5,  4,  3,  2,  1],
                   [1,  1,  1,  1,  0]]).astype(float)

    abT = transpose_banded((1, 2), ab)

    for i in range(l):
        assert np.allclose(abT[l-i-1, l-i:], ab[u+1+i, :-i-1])

    assert np.allclose(abT[l], ab[u])

    for i in range(u):
        assert np.allclose(abT[l+1+i, :-i-1], ab[u-i-1, i+1:])


def test_lds_log_probability(T=25, D=4):
    """
    Test lds_log_probability correctness
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)

    # Convert to dense matrix
    J_full = np.zeros((T*D, T*D))
    for t in range(T):
        J_full[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag[t]

    for t in range(T-1):
        J_full[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag[t].T
        J_full[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag[t]

    Sigma = np.linalg.inv(J_full)
    mu = Sigma.dot(h.ravel()).reshape((T, D))
    x = npr.randn(T, D)

    from scipy.stats import multivariate_normal
    ll_true = multivariate_normal.logpdf(x.ravel(), mu.ravel(), Sigma)

    # Solve with the banded solver
    ll_test = lds_log_probability(x, As, bs, Qi_sqrts, ms, Ri_sqrts)

    assert np.allclose(ll_true, ll_test), "True LL {} != Test LL {}".format(ll_true, ll_test)


def test_lds_mean(T=25, D=4):
    """
    Test lds_mean correctness
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)

    # Convert to dense matrix
    J_full = np.zeros((T*D, T*D))
    for t in range(T):
        J_full[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag[t]

    for t in range(T-1):
        J_full[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag[t].T
        J_full[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag[t]

    Sigma = np.linalg.inv(J_full)
    mu_true = Sigma.dot(h.ravel()).reshape((T, D))


    # Solve with the banded solver
    mu_test = lds_mean(As, bs, Qi_sqrts, ms, Ri_sqrts)

    assert np.allclose(mu_true, mu_test)


def test_lds_sample(T=25, D=4):
    """
    Test lds_sample correctness
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)

    # Convert to dense matrix
    J_full = np.zeros((T*D, T*D))
    for t in range(T):
        J_full[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag[t]

    for t in range(T-1):
        J_full[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag[t].T
        J_full[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag[t]

    z = npr.randn(T*D,)

    # Sample directly
    L = np.linalg.cholesky(J_full)
    xtrue = np.linalg.solve(L.T, z).reshape(T, D)
    xtrue += np.linalg.solve(J_full, h.reshape(T*D)).reshape(T, D)

    # Solve with the banded solver
    xtest = lds_sample(As, bs, Qi_sqrts, ms, Ri_sqrts, z=z)

    assert np.allclose(xtrue, xtest)


# Test the gradients
def test_blocks_to_banded_grad(T=25, D=4):
    """
    Test blocks_to_banded gradient
    """
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))

    check_grads(blocks_to_bands, argnum=0, modes=['rev'], order=1)(J_diag, J_lower_diag)
    check_grads(blocks_to_bands, argnum=1, modes=['rev'], order=1)(J_diag, J_lower_diag)


def test_transpose_banded_grad(T=25, D=4):
    """
    Test transpose_banded gradient
    """
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))
    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)

    check_grads(transpose_banded, argnum=1, modes=['rev'], order=1)((2*D-1, 0), J_banded)


def test_cholesky_banded_grad(T=10, D=4):
    """
    Test cholesky_banded gradient
    """
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    L = np.linalg.cholesky(J_full)
    dJ_bar_true = elementwise_grad(np.linalg.cholesky)(J_full)

    # Convert to lower bands
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))

    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
    L_banded = np.vstack([
        np.concatenate((np.diag(L, -d), np.zeros(d))) for d in range(2 * D)
        ])
    dJ_banded = grad_cholesky_banded(L_banded, J_banded)(np.ones_like(L_banded))

    assert np.allclose(np.diag(dJ_bar_true), dJ_banded[0])
    for d in range(1, 2 * D):
        assert np.allclose(np.diag(dJ_bar_true, -d), dJ_banded[d, :-d] / 2)


def test_solve_banded_grad(T=10, D=4):
    """
    Test solve_banded gradient
    """
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)

    L_full = np.linalg.cholesky(J_full)
    L_banded = np.vstack([[
        np.concatenate((np.diag(L_full, -d), np.zeros(d))) for d in range(2*D)
        ]])

    b = npr.randn(T * D)

    # Check gradient against that of regular solve.
    g_true = elementwise_grad(np.linalg.solve)(L_full, b)
    g_test = elementwise_grad(solve_banded, argnum=1)((2*D-1, 0), L_banded, b)
    assert np.allclose(np.diag(g_true), g_test[0])
    for d in range(1, 2 * D):
        assert np.allclose(np.diag(g_true, -d), g_test[d, :-d])

    check_grads(solve_banded, argnum=1, modes=['rev'], order=1)((2*D-1, 0), L_banded, b)
    check_grads(solve_banded, argnum=2, modes=['rev'], order=1)((2*D-1, 0), L_banded, b)


def test_solveh_banded_grad(T=10, D=4):
    """
    Test solveh_banded gradient
    """
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))
    b = npr.randn(T * D)

    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
    check_grads(solveh_banded, argnum=0, modes=['rev'], order=1)(J_banded, b, lower=True)
    check_grads(solveh_banded, argnum=1, modes=['rev'], order=1)(J_banded, b, lower=True)

    J_banded = blocks_to_bands(J_diag, np.swapaxes(J_lower_diag, -1, -2), lower=False)
    check_grads(solveh_banded, argnum=0, modes=['rev'], order=1)(J_banded, b, lower=False)
    check_grads(solveh_banded, argnum=1, modes=['rev'], order=1)(J_banded, b, lower=False)


def test_cholesky_lds_grad(T=10, D=4):
    """
    Test cholesky_lds gradient
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)

    check_grads(cholesky_lds, argnum=0, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(cholesky_lds, argnum=2, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(cholesky_lds, argnum=4, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts)


def test_solve_lds_grad(T=10, D=4):
    """
    Test solve_lds gradient
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    v = npr.randn(T, D)

    check_grads(solve_lds, argnum=0, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, v)
    check_grads(solve_lds, argnum=2, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, v)
    check_grads(solve_lds, argnum=4, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, v)
    check_grads(solve_lds, argnum=5, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, v)


def test_lds_log_probability_grad(T=10, D=2):
    """
    Test lds_log_probability gradient
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    x = npr.randn(T, D)

    check_grads(lds_log_probability, argnum=0, modes=['rev'], order=1)(x, As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(lds_log_probability, argnum=1, modes=['rev'], order=1)(x, As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(lds_log_probability, argnum=2, modes=['rev'], order=1)(x, As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(lds_log_probability, argnum=3, modes=['rev'], order=1)(x, As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(lds_log_probability, argnum=4, modes=['rev'], order=1)(x, As, bs, Qi_sqrts, ms, Ri_sqrts)
    check_grads(lds_log_probability, argnum=5, modes=['rev'], order=1)(x, As, bs, Qi_sqrts, ms, Ri_sqrts)


def test_lds_sample_grad(T=10, D=2):
    """
    Test lds_sample gradient
    """
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    z = npr.randn(T, D)

    check_grads(lds_sample, argnum=0, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, z=z)
    check_grads(lds_sample, argnum=1, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, z=z)
    check_grads(lds_sample, argnum=2, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, z=z)
    check_grads(lds_sample, argnum=3, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, z=z)
    check_grads(lds_sample, argnum=4, modes=['rev'], order=1)(As, bs, Qi_sqrts, ms, Ri_sqrts, z=z)


def test_lds_log_probability_perf(T=1000, D=10, N_iter=10):
    """
    Compare performance of banded method vs message passing in pylds.
    """
    print("Comparing methods for T={} D={}".format(T, D))

    from pylds.lds_messages_interface import kalman_info_filter, kalman_filter

    # Convert LDS parameters into info form for pylds
    As, bs, Qi_sqrts, ms, Ri_sqrts = make_lds_parameters(T, D)
    Qis = np.matmul(Qi_sqrts, np.swapaxes(Qi_sqrts, -1, -2))
    Ris = np.matmul(Ri_sqrts, np.swapaxes(Ri_sqrts, -1, -2))
    x = npr.randn(T, D)

    print("Timing banded method")
    start = time.time()
    for itr in range(N_iter):
        lds_log_probability(x, As, bs, Qi_sqrts, ms, Ri_sqrts)
    stop = time.time()
    print("Time per iter: {:.4f}".format((stop - start) / N_iter))

    # Compare to Kalman Filter
    mu_init = np.zeros(D)
    sigma_init = np.eye(D)
    Bs = np.ones((D, 1))
    sigma_states = np.linalg.inv(Qis)
    Cs = np.eye(D)
    Ds = np.zeros((D, 1))
    sigma_obs = np.linalg.inv(Ris)
    inputs = bs
    data = ms

    print("Timing PyLDS message passing (kalman_filter)")
    start = time.time()
    for itr in range(N_iter):
        kalman_filter(mu_init, sigma_init,
            np.concatenate([As, np.eye(D)[None, :, :]]), Bs, np.concatenate([sigma_states, np.eye(D)[None, :, :]]),
            Cs, Ds, sigma_obs, inputs, data)
    stop = time.time()
    print("Time per iter: {:.4f}".format((stop - start) / N_iter))

    # Info form comparison
    J_init = np.zeros((D, D))
    h_init = np.zeros(D)
    log_Z_init = 0

    J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(As, bs, Qi_sqrts, ms, Ri_sqrts)
    J_pair_21 = J_lower_diag
    J_pair_22 = J_diag[1:]
    J_pair_11 = J_diag[:-1]
    J_pair_11[1:] = 0
    h_pair_2 = h[1:]
    h_pair_1 = h[:-1]
    h_pair_1[1:] = 0
    log_Z_pair = 0

    J_node = np.zeros((T, D, D))
    h_node = np.zeros((T, D))
    log_Z_node = 0

    print("Timing PyLDS message passing (kalman_info_filter)")
    start = time.time()
    for itr in range(N_iter):
        kalman_info_filter(J_init, h_init, log_Z_init,
            J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair,
            J_node, h_node, log_Z_node)
    stop = time.time()
    print("Time per iter: {:.4f}".format((stop - start) / N_iter))

def test_lds_sample_and_fit(T=100, N=15, K=3, D=10, num_cases=25):

    TRANSITIONS_NAMES = [
        "stationary",
        "sticky",
        "inputdriven",
        "recurrent",
        "recurrent_only",
        # "rbf_recurrent",
        # "nn_recurrent",
        ]

    DYNAMICS_NAMES = [
        "none",
        "gaussian",
        "diagonal_gaussian",
        "studentst",
        "diagonal_t",
        ]

    # Exclude the identity emissions (for now)
    # because they require N == D
    EMISSIONS_NAMES = [
        "gaussian",
        "gaussian_orthog",
        "gaussian_nn",
        "studentst",
        "studentst_orthog",
        "studentst_nn",
        "poisson",
        "poisson_orthog",
        "poisson_nn",
        "bernoulli",
        "bernoulli_orthog",
        "bernoulli_nn",
        # "autoregressive",
        # "autoregressive_orthog",
        # "autoregressive_nn",
        ]
    METHODS = ["bbvi"]

    # method_name --> allowable posteriors
    POSTERIORS = {
        "bbvi": ["mf", "lds"],
        # "laplace_em": ["structured_meanfield"]
    }

    test_cases = list(
        it.product(DYNAMICS_NAMES,
                   EMISSIONS_NAMES,
                   TRANSITIONS_NAMES,
                   METHODS)
    )

    # Choose a random subset of combinations
    test_case_indices = npr.choice(len(test_cases), size=num_cases)

    # Test SLDS and RSLDS
    print("Testing SLDS and RSLDS...")
    for idx in test_case_indices:
        dynamics, emissions, transitions, method = test_cases[idx]
        for posterior in POSTERIORS[method]:
            npr.seed(seed=SEED)
            print("Fitting: "
                    "transitions = {},"
                    "dynamics = {}, "
                    "emissions = {}, "
                    "method = {}, "
                    "posterior = {}, ".format(
                    transitions,
                    dynamics,
                    emissions,
                    method,
                    posterior
                )
            )
            true_slds = ssm.SLDS(N, K, D,
                                transitions=transitions,
                                dynamics=dynamics,
                                emissions=emissions)
            z, x, y = true_slds.sample(T)

            fit_slds = ssm.SLDS(N, K, D,
                                transitions=transitions,
                                dynamics=dynamics,
                                emissions=emissions)
            fit_slds.fit(y,
                         method=method,
                         variational_posterior=posterior,
                         num_init_iters=2,
                         num_iters=2)


def lbfgs_newton_perf_comparison(T=100, N=15, K=3, D=10, ntrials=5, n_iters=20):
    np.random.seed(seed=123)
    true_slds = ssm.SLDS(N, K, D,
                            transitions="recurrent",
                            dynamics="gaussian",
                            emissions="gaussian")
    z, x, y = true_slds.sample(T)

    fit_slds = ssm.SLDS(N, K, D,
                        transitions="recurrent",
                        dynamics="gaussian",
                        emissions="gaussian")
    # Make sure all params are starting at the same value
    newtons_lds = copy.deepcopy(fit_slds)
    lbfgs_lds = copy.deepcopy(fit_slds)

    newton_time = 0
    for i in range(ntrials):
        start = time.time()
        newtons_lds.fit(y,
                    initialize=False,
                    num_iters=n_iters,
                    continuous_optimizer="newton")
        end = time.time()
        newton_time += (end - start) / n_iters
    newton_time /= ntrials
    print("Avg time/iter with newton's method: {:.4f}".format(newton_time))

    lbfgs_time = 0
    for i in range(ntrials):
        start = time.time()
        lbfgs_lds.fit(y,
                    initialize=False,
                    num_iters=n_iters,
                    continuous_optimizer="lbfgs")
        end = time.time()
        lbfgs_time += (end - start) / n_iters
    lbfgs_time /= ntrials
    print("Avg time/iter with lbfgs: {:.4f}".format(lbfgs_time))

def test_laplace_em(T=100, N=15, K=3, D=10, num_cases=25):
    # Check that laplace-em works for each transition and emission model
    # so long as the dynamics are linear-gaussian.
    DYNAMICS_NAMES = ["gaussian"]
    EMISSIONS_NAMES = ["gaussian",
                       "gaussian_orthog",
                       "poisson",
                       "poisson_orthog",
                       "bernoulli",
                       "bernoulli_orthog"]
    TRANSITIONS_NAMES = ["stationary",
                         "sticky",
                         "inputdriven",
                         "recurrent",
                         "recurrent_only"]
    INPUT_DIMS = [0, 1]

    test_cases = list(
        it.product(DYNAMICS_NAMES,
                   EMISSIONS_NAMES,
                   TRANSITIONS_NAMES,
                   INPUT_DIMS)
    )

    # Choose a random subset of combinations
    test_case_indices = npr.choice(len(test_cases), size=num_cases)

    # Test SLDS and RSLDS
    print("Testing SLDS and RSLDS...")
    for idx in test_case_indices:
        dynamics, emissions, transitions, input_dim = test_cases[idx]
        true_slds = ssm.SLDS(N, K, D, M=input_dim,
                                transitions=transitions,
                                dynamics="gaussian",
                                emissions=emissions)

        # Test with a random number of data arrays
        num_datas = npr.randint(1, 5)
        Ts = T + npr.randint(20, size=num_datas)
        us = [npr.randn(Ti, input_dim) for Ti in Ts]
        datas = [true_slds.sample(Ti, input=u) for Ti, u in zip(Ts, us)]
        zs, xs, ys = list(zip(*datas))

        # Fit an SLDS to the data
        fit_slds = ssm.SLDS(N, K, D, M=input_dim,
                            transitions=transitions,
                            dynamics="gaussian",
                            emissions=emissions)
        try:
            fit_slds.fit(ys,
                            inputs=us,
                            initialize=True,
                            num_init_iters=2,
                            num_iters=5)

        # So that we can still interrupt the test.
        except KeyboardInterrupt:
            raise

        # So that we know which test case fails...
        except:
            print("Error during fit with Laplace-EM. Failed with:")
            print("Emissions = {}".format(emissions))
            print("Transitions = {}".format(transitions))
            raise

def test_laplace_em_hessian(N=5, K=3, D=2, T=20):
    for transitions in ["standard", "recurrent", "recurrent_only"]:
        for emissions in ["gaussian_orthog", "gaussian"]:
            print("Checking analytical hessian for transitions={},  "
                  "and emissions={}".format(transitions, emissions)
            )
            slds = ssm.SLDS(N, K, D, transitions=transitions,
                            dynamics="gaussian",
                            emissions=emissions)
            z, x, y = slds.sample(T)
            new_slds = ssm.SLDS(N, K, D, transitions="standard",
                            dynamics="gaussian",
                            emissions=emissions)

            inputs = [np.zeros((T, 0))]
            masks = [np.ones_like(y)]
            tags = [None]
            method = "laplace_em"
            datas = [y]
            num_samples = 1

            def neg_expected_log_joint_wrapper(x_vec, T, D):
                x = x_vec.reshape(T, D)
                return new_slds._laplace_neg_expected_log_joint(datas[0],
                                                                inputs[0],
                                                                masks[0],
                                                                tags[0],
                                                                x,
                                                                Ez,
                                                                Ezzp1)
            variational_posterior = new_slds._make_variational_posterior("structured_meanfield",
                                                                        datas, inputs, masks, tags, method)
            new_slds._fit_laplace_em_discrete_state_update(
                            variational_posterior, datas, inputs, masks, tags, num_samples)
            Ez, Ezzp1, _ = variational_posterior.discrete_expectations[0]

            x = variational_posterior.mean_continuous_states[0]
            scale = x.size
            J_diag, J_lower_diag = new_slds._laplace_hessian_neg_expected_log_joint(datas[0],
                                                                inputs[0],
                                                                masks[0],
                                                                tags[0],
                                                                x,
                                                                Ez,
                                                                Ezzp1)
            dense_hessian = scipy.linalg.block_diag(*[x for x in J_diag])
            dense_hessian[D:, :-D] += scipy.linalg.block_diag(*[x for x in J_lower_diag])
            dense_hessian[:-D, D:] += scipy.linalg.block_diag(*[x.T for x in J_lower_diag])

            true_hess = hessian(neg_expected_log_joint_wrapper)(x.reshape(-1), T, D)
            assert np.allclose(true_hess, dense_hessian)
            print("Hessian passed.")

            # Also check that computation of H works.
            h_dense = dense_hessian @ x.reshape(-1)
            h_dense = h_dense.reshape(T, D)

            J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = new_slds._laplace_neg_hessian_params(datas[0],
                                                                inputs[0],
                                                                masks[0],
                                                                tags[0],
                                                                x,
                                                                Ez,
                                                                Ezzp1)
            h_ini, h_dyn_1, h_dyn_2, h_obs = new_slds._laplace_neg_hessian_params_to_hs(x,
                J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs
            )

            h = h_obs.copy()
            h[0] += h_ini
            h[:-1] += h_dyn_1
            h[1:] += h_dyn_2

            assert np.allclose(h, h_dense)


if __name__ == "__main__":
    # test_blocks_to_banded()
    # test_transpose_banded()
    # test_lds_log_probability()
    # test_lds_mean()
    # test_lds_sample()
    # test_blocks_to_banded_grad()
    # test_transpose_banded_grad()
    # test_cholesky_banded_grad()
    # test_solve_banded_grad()
    # test_solveh_banded_grad()
    # test_cholesky_lds_grad()
    # test_solve_lds_grad()
    # test_lds_log_probability_grad()
    # test_lds_sample_grad()
    # for D in range(2, 21, 2):
    #     test_lds_log_probability_perf(T=1000, D=D)
    # for T in [100, 1000]:
    #     print("Performance comparison for LBFGS vs. Newton's method "
    #           "with T={}".format(T))
    #     lbfgs_newton_perf_comparison(T=T, N=5, K=1, D=30)
    test_laplace_em_hessian()
