from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.optimizers import unflatten_optimizer

from scipy.optimize import linear_sum_assignment

@unflatten_optimizer
def adam_with_convergence_check(grad, x, callback=None, max_iters=10000, 
    step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, tol=1e-6,
    ):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    converged = False
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(max_iters):
        g = grad(x, i)
        if callback: 
            callback(x, i, g)

        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        dx = -step_size*mhat/(np.sqrt(vhat) + eps)
        x = x + dx

        # check for convergence
        if np.mean(abs(dx)) < tol:
            converged = True
            break 

    if not converged:
        warn("Adam failed to converge in {} iterations.".format(max_iters))

    return x

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
    assert K1 <= K2, "Can only find permutation from more states to fewer"
    
    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"
    
    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

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
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], self.M)) for data in datas]
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
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], 0)) for data in datas]
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


def logistic(x):
    return 1. / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def softplus(x):
    return np.log(1 + np.exp(x))


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


def generalized_newton_studentst_dof(E_tau, E_logtau, nu0=1, max_iter=100, nu_min=1e-3, nu_max=20, tol=1e-8, verbose=False):
    """
    Generalized Newton's method for the degrees of freedom parameter, nu, 
    of a Student's t distribution.  See the notebook in the doc/students_t
    folder for a complete derivation. 
    """
    from scipy.special import digamma, polygamma
    delbo = lambda nu: 1/2 * (1 + np.log(nu/2)) - 1/2 * digamma(nu/2) + 1/2 * E_logtau - 1/2 * E_tau
    ddelbo = lambda nu: 1/(2 * nu) - 1/4 * polygamma(1, nu/2)

    dnu = np.inf
    nu = nu0
    for itr in range(max_iter):
        if abs(dnu) < tol:
            break
            
        if nu < nu_min or nu > nu_max:
            warn("generalized_newton_studentst_dof fixed point grew beyond "
                 "bounds [{},{}].".format(nu_min, nu_max))
            nu = np.clip(nu, nu_min, nu_max)
            break
        
        # Perform the generalized Newton update
        a = -nu**2 * ddelbo(nu)
        b = delbo(nu) - a / nu
        assert a > 0 and b < 0, "generalized_newton_studentst_dof encountered invalid values of a,b"
        dnu = -a / b - nu
        nu = nu + dnu

    if itr == max_iter - 1:
        warn("generalized_newton_studentst_dof failed to converge" 
             "at tolerance {} in {} iterations.".format(tol, itr))

    return nu
