"""
Single step of a variety of optimization routines.
Modified from autograd.misc.optimizers.

The function being optimized must take two arguments,
an input value and an iteration number.
"""
from functools import partial
from warnings import warn

import numpy as onp
import scipy.optimize
from jax import grad, jit
from jax.tree_util import tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from itertools import count

# from autograd import grad, value_and_grad
# import autograd.numpy as np
# from autograd.misc import flatten
# from autograd.wrap_util import wraps
# from scipy.optimize import minimize


def minimize(fun, x0,
             method=None,
             args=(),
             bounds=None,
             constraints=(),
             tol=None,
             callback=None,
             options=None):
    """
    A simple wrapper for scipy.optimize.minimize using JAX.

    Args:
        fun: The objective function to be minimized, written in JAX code
        so that it is automatically differentiable.  It is of type,
            ```fun: x, *args -> float```
        where `x` is a PyTree and args is a tuple of the fixed parameters needed
        to completely specify the function.

        x0: Initial guess represented as a JAX PyTree.

        args: tuple, optional. Extra arguments passed to the objective function
        and its derivative.  Must consist of valid JAX types; e.g. the leaves
        of the PyTree must be floats.

        _The remainder of the keyword arguments are inherited from
        `scipy.optimize.minimize`, and their descriptions are copied here for
        convenience._

        method : str or callable, optional
        Type of solver.  Should be one of
            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object (added in version 0.14.0),
              see below for description.
        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.

        bounds : sequence or `Bounds`, optional
            Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
            trust-constr methods. There are two ways to specify the bounds:
                1. Instance of `Bounds` class.
                2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                is used to specify no bound.
            Note that in order to use `bounds` you will need to manually flatten
            them in the same order as your inputs `x0`.

        constraints : {Constraint, dict} or List of {Constraint, dict}, optional
            Constraints definition (only for COBYLA, SLSQP and trust-constr).
            Constraints for 'trust-constr' are defined as a single object or a
            list of objects specifying constraints to the optimization problem.
            Available constraints are:
                - `LinearConstraint`
                - `NonlinearConstraint`
            Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
            Each dictionary with fields:
                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (only for SLSQP).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.
            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.
            Note that COBYLA only supports inequality constraints.

            Note that in order to use `constraints` you will need to manually flatten
            them in the same order as your inputs `x0`.

        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific
            options.

        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:
                maxiter : int
                    Maximum number of iterations to perform. Depending on the
                    method each iteration may use several function evaluations.
                disp : bool
                    Set to True to print convergence messages.
            For method-specific options, see :func:`show_options()`.

        callback : callable, optional
            Called after each iteration. For 'trust-constr' it is a callable with
            the signature:
                ``callback(xk, OptimizeResult state) -> bool``
            where ``xk`` is the current parameter vector represented as a PyTree,
             and ``state`` is an `OptimizeResult` object, with the same fields
            as the ones from the return. If callback returns True the algorithm
            execution is terminated.

            For all the other methods, the signature is:
                ```callback(xk)```
            where `xk` is the current parameter vector, represented as a PyTree.

    Returns:
        res : The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are:
            ``x``: the solution array, represented as a JAX PyTree
            ``success``: a Boolean flag indicating if the optimizer exited successfully
            ``message``: describes the cause of the termination.
        See `scipy.optimize.OptimizeResult` for a description of other attributes.

    """
    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)

    # Wrap the objective function to consume flat _original_
    # numpy arrays and produce scalar outputs.
    def fun_wrapper(x_flat, *args):
        x = unravel(x_flat)
        return float(fun(x, *args))

    # Wrap the gradient in a similar manner
    jac = jit(grad(fun))
    def jac_wrapper(x_flat, *args):
        x = unravel(x_flat)
        g_flat, _ = ravel_pytree(jac(x, *args))
        return onp.array(g_flat)

    # Wrap the callback to consume a pytree
    def callback_wrapper(x_flat, *args):
        if callback is not None:
            x = unravel(x_flat)
            return callback(x, *args)

    # Minimize with scipy
    results = scipy.optimize.minimize(fun_wrapper,
                                      x0_flat,
                                      args=args,
                                      method=method,
                                      jac=jac_wrapper,
                                      callback=callback_wrapper,
                                      bounds=bounds,
                                      constraints=constraints,
                                      tol=tol,
                                      options=options)

    # pack the output back into a PyTree
    results["x"] = unravel(results["x"])
    return results

def convex_combination(curr_params, new_params, step_size):
    """
    Output next = step_size * target + (1-step_size) * curr
    where target, curr, and next can be PyTree's of nested
    containers with arrays/scalars at the leaves.
    Assume curr and target have the same structure.
    """
    assert step_size >= 0 and step_size <= 1
    _curr_params, unravel = ravel_pytree(curr_params)
    _new_params, _ = ravel_pytree(new_params)
    return unravel((1 - step_size) * _curr_params + step_size * _new_params)


## OLD


# def unflatten_optimizer_step(step):
#     """
#     Wrap an optimizer step function that operates on flat 1D arrays
#     with a version that handles trees of nested containers,
#     i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
#     """
#     @wraps(step)
#     def _step(value_and_grad, x, itr, state=None, *args, **kwargs):
#         _x, unflatten = flatten(x)
#         def _value_and_grad(x, i):
#             v, g = value_and_grad(unflatten(x), i)
#             return v, flatten(g)[0]
#         _next_x, _next_val, _next_g, _next_state = \
#             step(_value_and_grad, _x, itr, state=state, *args, **kwargs)
#         return unflatten(_next_x), _next_val, _next_g, _next_state
#     return _step


# @unflatten_optimizer_step
# def sgd_step(value_and_grad, x, itr, state=None, step_size=0.1, mass=0.9):
#     # Stochastic gradient descent with momentum.
#     velocity = state if state is not None else np.zeros(len(x))
#     val, g = value_and_grad(x, itr)
#     velocity = mass * velocity - (1.0 - mass) * g
#     x = x + step_size * velocity
#     return x, val, g, velocity

# @unflatten_optimizer_step
# def rmsprop_step(value_and_grad, x, itr, state=None, step_size=0.1, gamma=0.9, eps=10**-8):
#     # Root mean squared prop: See Adagrad paper for details.
#     avg_sq_grad = np.ones(len(x)) if state is None else state
#     val, g = value_and_grad(x, itr)
#     avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
#     x = x - (step_size * g) / (np.sqrt(avg_sq_grad) + eps)
#     return x, val, g, avg_sq_grad


# @unflatten_optimizer_step
# def adam_step(value_and_grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
#     """
#     Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
#     It's basically RMSprop with momentum and some correction terms.
#     """
#     m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
#     val, g = value_and_grad(x, itr)
#     m = (1 - b1) * g      + b1 * m    # First  moment estimate.
#     v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
#     mhat = m / (1 - b1**(itr + 1))    # Bias correction.
#     vhat = v / (1 - b2**(itr + 1))
#     x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
#     return x, val, g, (m, v)


# def _generic_sgd(method, loss, x0, callback=None, num_iters=200, state=None, full_output=False, **kwargs):
#     """
#     Generic stochastic gradient descent step.
#     """
#     step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[method]

#     # Initialize outputs
#     x, losses, grads = x0, [], []
#     for itr in range(num_iters):
#         x, val, g, state = step(value_and_grad(loss), x, itr, state, **kwargs)
#         losses.append(val)
#         grads.append(g)

#     if full_output:
#         return x, state
#     else:
#         return x


# def _generic_minimize(method, loss, x0,
#                       verbose=False,
#                       num_iters=1000,
#                       tol=1e-4,
#                       state=None,
#                       full_output=False,
#                       suppress_warnings=False,
#                       **kwargs):
#     """
#     Minimize a given loss function with scipy.optimize.minimize.
#     """
#     # Flatten the loss
#     _x0, unflatten = flatten(x0)
#     _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)

#     if verbose:
#         print("Fitting with {}.".format(method))

#     # Specify callback for fitting
#     itr = [0]
#     def callback(x_flat):
#         itr[0] += 1
#         print("Iteration {} loss: {:.3f}".format(itr[0], loss(unflatten(x_flat), -1)))

#     # Wrap the gradient to avoid NaNs
#     def safe_grad(x, itr):
#         g = grad(_objective)(x, itr)
#         g[~np.isfinite(g)] = 1e8
#         return g

#     # Call the optimizer.  Pass in -1 as the iteration since it is unused.
#     result = minimize(_objective, _x0, args=(-1,),
#                       jac=safe_grad,
#                       method=method,
#                       callback=callback if verbose else None,
#                       options=dict(maxiter=num_iters, disp=verbose),
#                       tol=tol,
#                       **kwargs)
#     if verbose:
#         print("{} completed with message: \n{}".format(method, result.message))

#     if not suppress_warnings and not result.success:
#         warn("{} failed with message:\n{}".format(method, result.message))

#     if full_output:
#         return unflatten(result.x), result
#     else:
#         return unflatten(result.x)

# # Define optimizers
# sgd = partial(_generic_sgd, "sgd")
# rmsprop = partial(_generic_sgd, "rmsprop")
# adam = partial(_generic_sgd, "adam")
# bfgs = partial(_generic_minimize, "BFGS")
# lbfgs = partial(_generic_minimize, "L-BFGS-B")


# # Special optimizer for function with block-tridiagonal hessian
# def newtons_method_block_tridiag_hessian(
#     x0, obj, grad_func, hess_func,
#     tolerance=1e-4, maxiter=100):
#     """
#     Newton's method to minimize a positive definite function with a
#     block tridiagonal Hessian matrix.
#     Algorithm 9.5, Boyd & Vandenberghe, 2004.
#     """
#     x = x0
#     is_converged = False
#     count = 0
#     while not is_converged:
#         H_diag, H_lower_diag = hess_func(x)
#         g = grad_func(x)
#         dx = -1.0 * solve_symm_block_tridiag(H_diag, H_lower_diag, g)
#         lambdasq = np.dot(g.ravel(), -1.0*dx.ravel())
#         if lambdasq / 2.0 <= tolerance:
#             is_converged = True
#             break
#         stepsize = backtracking_line_search(x, dx, obj, g)
#         x = x + stepsize * dx
#         count += 1
#         if count > maxiter:
#             break

#     if not is_converged:
#         warn("Newton's method failed to converge in {} iterations. "
#              "Final mean abs(dx): {}".format(maxiter, np.mean(np.abs(dx))))

#     return x


# def backtracking_line_search(x0, dx, obj, g, stepsize = 1.0, min_stepsize=1e-8,
#                              alpha=0.2, beta=0.7):
#     """
#     A backtracking line search for the step size in Newton's method.
#     Algorithm 9.2, Boyd & Vandenberghe, 2004.
#     - dx is the descent direction
#     - g is the gradient evaluated at x0
#     - alpha in (0,0.5) is fraction of decrease in objective predicted  by
#         a linear extrapolation that we will accept
#     - beta in (0,1) is step size reduction factor
#     """
#     x = x0

#     # criterion: stop when f(x + stepsize * dx) < f(x) + \alpha * stepsize * f'(x)^T dx
#     f_term = obj(x)
#     grad_term = alpha * np.dot(g.ravel(), dx.ravel())

#     # decrease stepsize until criterion is met
#     # or stop at minimum step size
#     while stepsize > min_stepsize:
#         fx = obj(x+ stepsize*dx)
#         if np.isnan(fx) or fx > f_term + grad_term*stepsize:
#             stepsize *= beta
#         else:
#             break

#     return stepsize
