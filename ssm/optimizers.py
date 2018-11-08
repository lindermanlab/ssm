"""
Single step of a variety of optimization routines. 
Modified from autograd.misc.optimizers.

The function being optimized must take two arguments,
an input value and an iteration number.  
"""
import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

def unflatten_optimizer_step(step):
    """
    Wrap an optimizer step function that operates on flat 1D arrays
    with a version that handles trees of nested containers, 
    i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
    """
    @wraps(step)
    def _step(grad, x, itr, state=None, *args, **kwargs):
        _x, unflatten = flatten(x)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        _next_x, _next_g, _next_state = step(_grad, _x, itr, state=state, *args, **kwargs)
        return unflatten(_next_x), _next_g, _next_state
    return _step


def convex_combination(curr, target, alpha):
    """
    Output next = (1-alpha) * target + alpha * curr
    where target, curr, and next can be trees of nested
    containers with arrays/scalars at the leaves.
    Assume curr and target have the same structure.
    """
    assert alpha >= 0 and alpha <= 1
    _curr, unflatten = flatten(curr)
    _target, _ = flatten(target)
    return unflatten(alpha * _curr + (1-alpha) * _target)


@unflatten_optimizer_step
def sgd_step(grad, x, itr, state=None, step_size=0.1, mass=0.9):
    # Stochastic gradient descent with momentum.
    velocity = state if state is not None else np.zeros(len(x))
    g = grad(x, itr)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, g, velocity


@unflatten_optimizer_step
def rmsprop_step(grad, x, itr, state=None, step_size=0.1, gamma=0.9, eps=10**-8):
    # Root mean squared prop: See Adagrad paper for details.
    avg_sq_grad = np.ones(len(x)) if state is None else state
    g = grad(x, itr)
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
    x = x - (step_size * g) / (np.sqrt(avg_sq_grad) + eps)
    return x, g, avg_sq_grad


@unflatten_optimizer_step
def adam_step(grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    """
    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    g = grad(x, itr)
    m = (1 - b1) * g      + b1 * m    # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
    mhat = m / (1 - b1**(itr + 1))    # Bias correction.
    vhat = v / (1 - b2**(itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, g, (m, v)
