"""
Single step of a variety of optimization routines.
Modified from autograd.misc.optimizers.

The function being optimized must take two arguments,
an input value and an iteration number.
"""
from functools import partial
from warnings import warn

from autograd import grad, value_and_grad
import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

from scipy.optimize import minimize

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


def unflatten_optimizer_step(step):
    """
    Wrap an optimizer step function that operates on flat 1D arrays
    with a version that handles trees of nested containers,
    i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
    """
    @wraps(step)
    def _step(value_and_grad, x, itr, state=None, *args, **kwargs):
        _x, unflatten = flatten(x)
        def _value_and_grad(x, i):
            v, g = value_and_grad(unflatten(x), i)
            return v, flatten(g)[0]
        _next_x, _next_val, _next_g, _next_state = \
            step(_value_and_grad, _x, itr, state=state, *args, **kwargs)
        return unflatten(_next_x), _next_val, _next_g, _next_state
    return _step


@unflatten_optimizer_step
def sgd_step(value_and_grad, x, itr, state=None, step_size=0.1, mass=0.9):
    # Stochastic gradient descent with momentum.
    velocity = state if state is not None else np.zeros(len(x))
    val, g = value_and_grad(x, itr)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, val, g, velocity

@unflatten_optimizer_step
def rmsprop_step(value_and_grad, x, itr, state=None, step_size=0.1, gamma=0.9, eps=10**-8):
    # Root mean squared prop: See Adagrad paper for details.
    avg_sq_grad = np.ones(len(x)) if state is None else state
    val, g = value_and_grad(x, itr)
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
    x = x - (step_size * g) / (np.sqrt(avg_sq_grad) + eps)
    return x, val, g, avg_sq_grad


@unflatten_optimizer_step
def adam_step(value_and_grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    """
    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    val, g = value_and_grad(x, itr)
    m = (1 - b1) * g      + b1 * m    # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
    mhat = m / (1 - b1**(itr + 1))    # Bias correction.
    vhat = v / (1 - b2**(itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, val, g, (m, v)


def _generic_sgd(method, loss, x0, callback=None, num_iters=200, state=None, full_output=False, **kwargs):
    """
    Generic stochastic gradient descent step.
    """
    step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[method]

    # Initialize outputs
    x, losses, grads = x0, [], []
    for itr in range(num_iters):
        x, val, g, state = step(value_and_grad(loss), x, itr, state, **kwargs)
        losses.append(val)
        grads.append(g)

    if full_output:
        return x, state
    else:
        return x


def _generic_minimize(method, loss, x0, verbose=False, num_iters=1000, state=None, full_output=False, suppress_warnings=False, **kwargs):
    """
    Minimize a given loss function with scipy.optimize.minimize.
    """
    # Flatten the loss
    _x0, unflatten = flatten(x0)
    _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)

    if verbose:
        print("Fitting with {}.".format(method))

    # Specify callback for fitting
    itr = [0]
    def callback(x_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0], loss(unflatten(x_flat), -1)))

    # Call the optimizer.
    # HACK: Pass in -1 as the iteration.
    result = minimize(_objective, _x0, args=(-1,), jac=grad(_objective),
                      method=method,
                      callback=callback if verbose else None,
                      options=dict(maxiter=num_iters, disp=verbose),
                      **kwargs)
    if verbose:
        print("{} completed with message: \n{}".format(method, result.message))

    if not suppress_warnings and not result.success:
        warn("{} failed with message:\n{}".format(method, result.message))

    if full_output:
        return unflatten(result.x), result
    else:
        return unflatten(result.x)

# Define optimizers
sgd = partial(_generic_sgd, "sgd")
rmsprop = partial(_generic_sgd, "rmsprop")
adam = partial(_generic_sgd, "adam")
bfgs = partial(_generic_minimize, "BFGS")
lbfgs = partial(_generic_minimize, "L-BFGS-B")
