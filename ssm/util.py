import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.optimize import linear_sum_assignment

def compute_state_overlap(z1, z2):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0
    
    K = max(z1.max(), z2.max()) + 1
    overlap = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap

def find_permutation(z1, z2, K=None):
    overlap = compute_state_overlap(z1, z2)
    K_data = overlap.shape[0]
    
    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K_data)), "All indices should have been matched!"
    assert len(perm) == K_data

    # Check if the overlap matrix is smaller than K
    # If so, pad as necessary
    if K is not None and K_data < K:
        perm = np.concatenate((perm, np.arange(K_data, K)))

    return perm

def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def ensure_args_are_lists(f):
    def wrapper(self, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], self.M)) for data in datas]
        elif not isinstance(inputs, list):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, list):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, list):
            tags = [tags]

        return f(self, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_elbo_args_are_lists(f):
    def wrapper(self, variational_params, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], self.M)) for data in datas]
        elif not isinstance(inputs, list):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, list):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, list):
            tags = [tags]

        return f(self, variational_params, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_args_not_none(f):
    def wrapper(self, data, input=None, mask=None, tag=None, **kwargs):
        assert data is not None
        input = np.zeros((data.shape[0], self.M)) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def ensure_slds_args_not_none(f):
    def wrapper(self, variational_mean, data, input=None, mask=None, tag=None, **kwargs):
        assert variational_mean is not None
        assert data is not None
        input = np.zeros((data.shape[0], self.M)) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, variational_mean, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def interpolate_data(data, mask):
    assert data.shape == mask.shape and mask.dtype == bool
    T, N = data.shape
    interp_data = data.copy()
    if np.any(~mask):
        for n in range(N):
            if np.sum(mask[:,n]) >= 2:
                t_missing = np.arange(T)[~mask[:,n]]
                t_given = np.arange(T)[mask[:,n]]
                y_given = data[mask[:,n], n]
                interp_data[~mask[:,n], n] = np.interp(t_missing, t_given, y_given)
            else:
                # Can't do much if we don't see anything... just set it to zero
                interp_data[~mask[:,n], n] = 0
    return interp_data


def logistic(x):
    return 1. / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def softplus(x):
    return np.log(1 + np.exp(x))


def inv_softplus(y):
    return np.log(np.exp(y) - 1)


