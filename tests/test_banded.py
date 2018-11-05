from autograd import elementwise_grad
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads

from ssm.cstats import transpose_banded

from ssm.primitives import blocks_to_bands, bands_to_blocks, \
       solve_block_tridiag, logdet_block_tridiag, \
       cholesky_block_tridiag, sample_block_tridiag, lds_normalizer, \
       solveh_banded, solve_banded, _convert_lds_to_block_tridiag, \
       grad_cholesky_banded, cholesky_banded, cholesky_lds, solve_lds


def test_blocks_to_banded(T=5, D=3):
    """
    Test that we get the right conversion out
    """
    Ad = np.zeros((T, D, D))
    Aod = np.zeros((T-1, D, D))

    M = np.arange(1, D+1)[:, None] * 10 + np.arange(1, D+1)
    for t in range(T):
        Ad[t, :, :] = 100 * ((t+1)*10 + (t+1)) + M

    for t in range(T-1):
        Aod[t, :, :] = 100 * ((t+2)*10 + (t+1)) + M

    print("Lower")
    L = blocks_to_bands(Ad, Aod, lower=True)
    print(L)

    print("Upper")
    U = blocks_to_bands(Ad, Aod, lower=False)
    print(U)

    # Check inverse with random symmetric matrices
    Ad = npr.randn(T, D, D)
    Ad = (Ad + np.swapaxes(Ad, -1, -2)) / 2
    Aod = npr.randn(T-1, D, D)

    Ad2, Aod2 = bands_to_blocks(blocks_to_bands(Ad, Aod, lower=True), lower=True)
    assert np.allclose(Ad, Ad2)
    assert np.allclose(Aod, Aod2)

    Ad3, Aod3 = bands_to_blocks(blocks_to_bands(Ad, Aod, lower=False), lower=False)
    assert np.allclose(Ad, Ad3)
    assert np.allclose(Aod, Aod3)


def test_transpose_banded():
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

def make_block_tridiag(T, D):
    A = npr.randn(D, D)
    Q = np.eye(D)

    J_diag = np.linalg.inv(Q) + A.T.dot(np.linalg.solve(Q, A))
    J_lower_diag = -np.linalg.solve(Q, A)

    # Solve the dense way
    J_full = np.zeros((T*D, T*D))
    for t in range(T):
        J_full[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag

    for t in range(T-1):
        J_full[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag.T
        J_full[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag

    return J_diag, J_lower_diag, J_full


def test_solve_block_tridiag(T=25, D=4):
    """
    Compare block tridiag and full linear solve 
    """
    v = npr.randn(T*D)
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    xtrue = np.linalg.solve(J_full, v)

    # Solve the sparse way
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))
    xtest = solve_block_tridiag(J_diag, J_lower_diag, v)

    assert np.allclose(xtrue, xtest)


def test_logdet_block_tridiag(T=25, D=4):
    """
    Compare block tridiag and full log determinant
    """
    v = npr.randn(T*D)
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    xtrue = np.linalg.slogdet(J_full)[1]

    # Solve with the banded solver
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))
    xtest = logdet_block_tridiag(J_diag, J_lower_diag)

    assert np.allclose(xtrue, xtest)


def test_sample_block_tridiag(T=25, D=4, S=3):
    """
    Compare block tridiag and full sampling
    """
    z = npr.randn(T*D, S)
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    L = np.linalg.cholesky(J_full)
    xtrue = np.linalg.solve(L.T, z).T.reshape(S, T, D)

    # Solve with the banded solver
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))
    xtest = sample_block_tridiag(J_diag, J_lower_diag, lower=True, size=S, z=z)

    assert np.allclose(xtrue, xtest)


def test_lds_normalizer(T=25, D=4, S=3):
    """
    Compare block tridiag and full sampling
    """
    As = npr.randn(T, D, D)
    Qinv_halves = np.tile(np.eye(D)[None, :, :], (T, 1, 1))
    J_diag, J_lower_diag = _convert_lds_to_block_tridiag(As, Qinv_halves)

    # Convert to dense matrix
    J_full = np.zeros((T*D, T*D))
    for t in range(T):
        J_full[t*D:(t+1)*D, t*D:(t+1)*D] = J_diag[t]

    for t in range(T-1):
        J_full[t*D:(t+1)*D, (t+1)*D:(t+2)*D] = J_lower_diag[t].T
        J_full[(t+1)*D:(t+2)*D, t*D:(t+1)*D] = J_lower_diag[t]
    
    h = npr.randn(T, D)
    Sigma = np.linalg.inv(J_full)
    mu = Sigma.dot(h.ravel()).reshape((T, D))
    x = npr.randn(T, D)

    from scipy.stats import multivariate_normal
    ll_true = multivariate_normal.logpdf(x.ravel(), mu.ravel(), Sigma)

    # Solve with the banded solver
    ll_test = lds_normalizer(x, As, Qinv_halves, h)

    assert np.allclose(ll_true, ll_test)


def test_blocks_to_banded_primitive(T=25, D=4):
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))

    check_grads(blocks_to_bands, argnum=0, modes=['rev'], order=1)(J_diag, J_lower_diag)
    check_grads(blocks_to_bands, argnum=1, modes=['rev'], order=1)(J_diag, J_lower_diag)


def test_cholesky_banded_primitive(T=10, D=4):
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


def test_solve_banded_primitive(T=10, D=4):
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    
    L_full = np.linalg.cholesky(J_full)
    L_banded = np.vstack([[
        np.concatenate((np.diag(L_full, -d), np.zeros(d))) for d in range(2*D)
        ]])

    b = npr.randn(T * D)

    # The gradient wrt A is tricky to do with numerical gradients 
    # because A must be invertible.  Instead, check against regular solves.
    g_true = elementwise_grad(np.linalg.solve)(L_full, b)
    g_test = elementwise_grad(solve_banded, argnum=1)((2*D-1, 0), L_banded, b)
    assert np.allclose(np.diag(g_true), g_test[0])
    for d in range(1, 2 * D):
        assert np.allclose(np.diag(g_true, -d), g_test[d, :-d])

    check_grads(solve_banded, argnum=1, modes=['rev'], order=1)((2*D-1, 0), L_banded, b)
    check_grads(solve_banded, argnum=2, modes=['rev'], order=1)((2*D-1, 0), L_banded, b)


def test_solveh_banded_primitive(T=10, D=4):
    J_diag, J_lower_diag, J_full = make_block_tridiag(T, D)
    J_diag = np.tile(J_diag[None, :, :], (T, 1, 1))
    J_lower_diag = np.tile(J_lower_diag[None, :, :], (T-1, 1, 1))
    J_banded = blocks_to_bands(J_diag, J_lower_diag, lower=True)
    b = npr.randn(T * D)

    check_grads(solveh_banded, argnum=0, modes=['rev'], order=1)(J_banded, b, lower=True)
    check_grads(solveh_banded, argnum=1, modes=['rev'], order=1)(J_banded, b, lower=True)


def test_cholesky_lds_primitive(T=10, D=4):
    As = npr.randn(T, D, D)
    Qinv_halves = np.tile(np.eye(D)[None, :, :], (T, 1, 1))
    check_grads(cholesky_lds, argnum=0, modes=['rev'], order=1)(As, Qinv_halves)


def test_solve_lds_primitive(T=10, D=4):
    As = npr.randn(T, D, D)
    Qinv_halves = np.tile(np.eye(D)[None, :, :], (T, 1, 1))
    b = npr.randn(T, D)
    
    check_grads(solve_lds, argnum=1, modes=['rev'], order=1)(As, Qinv_halves, b)


def test_lds_normalizer_grad(T=10, D=2):
    As = npr.randn(T, D, D)
    Qinv_halves = np.tile(np.eye(D)[None, :, :], (T, 1, 1))
    h = npr.randn(T, D)
    x = npr.randn(T, D)    

    check_grads(lds_normalizer, argnum=0, modes=['rev'], order=1)(x, As, Qinv_halves, h)
    check_grads(lds_normalizer, argnum=1, modes=['rev'], order=1)(x, As, Qinv_halves, h)
    check_grads(lds_normalizer, argnum=2, modes=['rev'], order=1)(x, As, Qinv_halves, h)
    check_grads(lds_normalizer, argnum=3, modes=['rev'], order=1)(x, As, Qinv_halves, h)


if __name__ == "__main__":
    # test_blocks_to_banded()
    test_transpose_banded()
    test_solve_block_tridiag()
    test_logdet_block_tridiag()
    test_sample_block_tridiag()
    test_lds_normalizer()

    test_blocks_to_banded_primitive()
    test_cholesky_banded_primitive()
    test_solve_banded_primitive()
    test_solveh_banded_primitive()
    
    test_cholesky_lds_primitive()
    test_solve_lds_primitive()
    test_lds_normalizer_grad()
    