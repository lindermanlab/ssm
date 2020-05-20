import time
import ssm
import numpy as np
import copy
import scipy
import timeit

from ssm.util import SEED
from ssm.primitives import lds_log_probability
from test_lds import make_lds_parameters


# Params formatted as N, D, T, num_datas
MULT_DATA_PARAMS = (200, 10, 50, 100)
SINGLE_DATA_PARAMS = (200, 10, 5000, 1)


def time_laplace_em_end2end(ncalls=5):
    print("Benchmarking 1 iter of laplace-em fitting on Vanilla LDS.")
    params_list = [SINGLE_DATA_PARAMS, MULT_DATA_PARAMS]
    for (N, D, T, num_datas) in params_list:
        lds_true = ssm.LDS(N, D, dynamics="gaussian", emissions="gaussian")
        datas = [lds_true.sample(T)[1] for _ in range(num_datas)]

        lds_new = ssm.LDS(N, D, dynamics="gaussian", emissions="gaussian")
        print("N, D, T, num_datas: = ", N, D, T, num_datas)
        total = timeit.timeit(lambda: lds_new.fit(datas, initialize=False, num_iters=1),
            number=ncalls)
        print("Avg time per call: %f" % (total / ncalls))

def time_lds_sample(ncalls=20):
    print("Testing continuous sample performance:")
    params_list = [SINGLE_DATA_PARAMS, MULT_DATA_PARAMS]
    for (N, D, T, num_datas) in params_list:
        lds_true = ssm.LDS(N, D, dynamics="gaussian", emissions="gaussian")
        datas = [lds_true.sample(T)[1] for _ in range(num_datas)]
        
        # Calling fit will return a variational posterior object.
        # This is simpler than creating one ourselves.
        _, posterior = lds_true.fit(datas,
            initialize=False, num_iters=1, method="laplace_em")

        # Now we test the speed of sampling from this object.
        print("N, D, T, num_datas: = ", N, D, T, num_datas)
        total = timeit.timeit(lambda: posterior.sample_continuous_states(),
            number=ncalls)
        print("Avg time per call: %f" % (total / ncalls))


def time_lds_log_probability_perf(T=1000, D=10, N_iter=10):
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

if __name__ == "__main__":
    time_laplace_em_end2end()
    # time_lds_sample()
    # time_lds_log_probability_perf()