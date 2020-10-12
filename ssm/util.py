from warnings import warn
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad

from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import gammaln, digamma, polygamma

SEED = hash("ssm") % (2**32)
LOG_EPS = 1e-16
DIV_EPS = 1e-16

def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def rle(stateseq):
    """
    Compute the run length encoding of a discrete state sequence.

    E.g. the state sequence [0, 0, 1, 1, 1, 2, 3, 3]
         would be encoded as ([0, 1, 2, 3], [2, 3, 1, 2])

    [Copied from pyhsmm.util.general.rle]

    Parameters
    ----------
    stateseq : array_like
        discrete state sequence

    Returns
    -------
    ids : array_like
        integer identities of the states

    durations : array_like (int)
        length of time in corresponding state
    """
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)


def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def ensure_args_are_lists(f):
    def wrapper(self, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)

        if inputs is None:
            inputs = [np.zeros((data.shape[0],) + M) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_variational_args_are_lists(f):
    def wrapper(self, arg0, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        try:
            M = (self.M,) if isinstance(self.M, int) else self.M
        except:
            # self does not have M if self is a variational posterior object
            # in that case, arg0 is a model, which does have an M parameter
            M = (arg0.M,) if isinstance(arg0.M, int) else arg0.M

        assert isinstance(M, tuple)

        if inputs is None:
            inputs = [np.zeros((data.shape[0],) + M) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, arg0, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_args_not_none(f):
    def wrapper(self, data, input=None, mask=None, tag=None, **kwargs):
        assert data is not None

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)
        input = np.zeros((data.shape[0],) + M) if input is None else input

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def ensure_slds_args_not_none(f):
    def wrapper(self, variational_mean, data, input=None, mask=None, tag=None, **kwargs):
        assert variational_mean is not None
        assert data is not None
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)
        input = np.zeros((data.shape[0],) + M) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, variational_mean, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper

def ssm_pbar(num_iters, verbose, description, prob):
    '''Return either progress bar or regular list for iterating. Inputs are:

      num_iters (int)
      verbose (int)     - if == 2, return trange object, else returns list
      description (str) - description for progress bar
      prob (float)      - values to initialize description fields at

    '''
    if verbose == 2:
        pbar = trange(num_iters)
        pbar.set_description(description.format(*prob))
    else:
        pbar = range(num_iters)
    return pbar


def logistic(x):
    return 1. / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def softplus(x):
    return np.log1p(np.exp(x))


def inv_softplus(y):
    return np.log(np.exp(y) - 1)


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def relu(x):
    return np.maximum(0, x)


def replicate(x, state_map, axis=-1):
    """
    Replicate an array of shape (..., K) according to the given state map
    to get an array of shape (..., R) where R is the total number of states.

    Parameters
    ----------
    x : array_like, shape (..., K)
        The array to be replicated.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R)
    """
    assert state_map.ndim == 1
    assert np.all(state_map >= 0) and np.all(state_map < x.shape[-1])
    return np.take(x, state_map, axis=axis)

def collapse(x, state_map, axis=-1):
    """
    Collapse an array of shape (..., R) to shape (..., K) by summing
    columns that map to the same state in [0, K).

    Parameters
    ----------
    x : array_like, shape (..., R)
        The array to be collapsed.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R)
    """
    R = x.shape[axis]
    assert state_map.ndim == 1 and state_map.shape[0] == R
    K = state_map.max() + 1
    return np.concatenate([np.sum(np.take(x, np.where(state_map == k)[0], axis=axis),
                                  axis=axis, keepdims=True)
                           for k in range(K)], axis=axis)


def check_shape(var, var_name, desired_shape):
    assert var.shape == desired_shape, "Variable {} is of wrong shape. "\
        "Expected {}, found {}.".format(var_name, desired_shape, var.shape)


def trace_product(A, B):
    """ Compute trace of the matrix product A*B efficiently.

    A, B can be 2D or 3D arrays, in which case the trace is computed along
    the last two axes. In this case, the function will return an array.
    Computed using the fact that tr(AB) = sum_{ij}A_{ij}B_{ji}.
    """
    ndimsA = A.ndim
    ndimsB = B.ndim
    assert ndimsA == ndimsB, "Both A and B must have same number of dimensions."
    assert ndimsA <= 3, "A and B must have 3 or fewer dimensions"

    # We'll take the trace along the last two dimensions.
    BT = np.swapaxes(B, -1, -2)
    return np.sum(A*BT, axis=(-1, -2))
