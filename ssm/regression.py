"""
General purpose classes for (generalized) linear regression observation models.
"""
from autograd import elementwise_grad
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.linalg import block_diag
from autograd.scipy.special import logsumexp, gammaln
from scipy.special import polygamma, digamma
from scipy.optimize import minimize
from warnings import warn
from ssm.util import check_shape

mean_functions = dict(
    identity=lambda x: x,
    logistic=lambda x: 1 / (1 + np.exp(-x)),
    exp=lambda x: np.exp(x),
    softplus=lambda x: np.log(1 + np.exp(x))
    )

partition_functions = dict(
    gaussian=lambda eta: 0.5 * np.dot(eta, eta),
    bernoulli=lambda eta: np.log1p(np.exp(eta)),
    poisson=lambda eta: np.exp(eta),
    negative_binomial=lambda eta, r: -r * np.log(1 - np.exp(eta))
    )

canonical_link_functions = dict(
    gaussian=lambda mu: mu,
    bernoulli=lambda mu: np.log(mu / (1-mu)),
    poisson=lambda mu: np.log(mu),
    negative_binomial=lambda mu, r: np.log(mu / (mu + r))
    )

model_kwarg_descriptions = dict(
    gaussian=dict(),
    bernoulli=dict(),
    poisson=dict(),
    negative_binomial=dict(r="The \"number of failures\" parameterizing the negative binomial distribution.")
    )


def fit_linear_regression(Xs, ys,
                          weights=None,
                          fit_intercept=True,
                          expectations=None,
                          prior_ExxT=None,
                          prior_ExyT=None,
                          nu0=1,
                          Psi0=1
                          ):
    """
    Fit a linear regression y_i ~ N(Wx_i + b, diag(S)) for W, b, S.

    Params
    ------
    Xs: array or list of arrays, each element is N x D,
        where N is the number of data points, and D is the
        dimension of x_i.
    ys: array or list of arrays, each element is N x P,
        where p is the dimension of y_i.
    weights: optional, list of scalars weighting each observation.
                Must be same length as Xs and ys.
    fit_intercept:  if False drop b.
    expectations: optional, tuple of sufficient statistics for the
                    regression. If provided, Xs and ys will be ignored,
                    and the regression is calculated only from the
                    sufficient statistics. Tuple should be of the form
                    (Exx, Exy, Eyy, weight_sum).
    prior_ExxT: D x D array. optional. Only used when expectations=None.
    prior_ExyT: D x P array. optional. Only used when expectations=None.
    nu0: prior on covariance from MNIW distribution.
    psi0: prior on covariance from MNIW distribution.

    Returns
    -------
    W, b, Sigmas: when fit_intercept=True.
    W, Sigmas: when fit_intercept=False.
    """
    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = ys if isinstance(ys, (list, tuple)) else [ys]
    assert len(Xs) == len(ys)

    p, d = Xs[0].shape[1], ys[0].shape[1]
    assert all([X.shape[1] == p for X in Xs])
    assert all([y.shape[1] == d for y in ys])
    assert all([X.shape[0] == y.shape[0] for X, y in zip(Xs, ys)])

    # Check the weights.  Default to all ones.
    if weights is not None:
        weights = weights if isinstance(weights, (list, tuple)) else [weights]
    else:
        weights = [np.ones(X.shape[0]) for X in Xs]

    x_dim = p + int(fit_intercept)
    ExxT = np.zeros((x_dim, x_dim))
    ExyT = np.zeros((x_dim, d))
    EyyT = np.zeros((d, d))
    weight_sum = 0
    if expectations is None:

        # Compute the posterior. The priors must include a prior for the
        # intercept term, if given.
        if prior_ExxT is not None and prior_ExyT is not None:
            check_shape(prior_ExxT, "prior_ExxT", (x_dim, x_dim))
            check_shape(prior_ExyT, "prior_ExyT", (x_dim, d))
            ExxT[:, :] = prior_ExxT
            ExyT[:, :] = prior_ExyT

        for X, y, weight in zip(Xs, ys, weights):
            X = np.column_stack((X, np.ones(X.shape[0]))) if fit_intercept else X
            weight_sum += np.sum(weight)
            weight = weight[:, None] if weight.ndim == 1 else weight
            weighted_x = X * weight
            weighted_y = y * weight
            ExxT += weighted_x.T @ X
            ExyT += weighted_x.T @ y
            EyyT += weighted_y.T @ y
    else:
        ExxT, ExyT, EyyT, weight_sum = expectations
        check_shape(ExxT, "ExxT", (x_dim, x_dim))
        check_shape(ExyT, "ExyT", (x_dim, d))
        check_shape(EyyT, "EyyT", (d, d))

    # Solve for the MAP estimate
    W_full = np.linalg.solve(ExxT, ExyT).T
    if fit_intercept:
        W, b = W_full[:, :-1], W_full[:, -1]
    else:
        W = W_full
        b = 0

    # Compute expected error for covariance matrix estimate
    # E[(y - Ax)(y - Ax)^T]
    expected_err = EyyT - W_full @ ExyT - ExyT.T @ W_full.T + W_full @ ExxT @ W_full.T
    nu = nu0 + weight_sum

    # Get MAP estimate of posterior covariance
    Sigma = (expected_err + Psi0 * np.eye(d)) / (nu + d + 1)
    if fit_intercept:
        return W, b, Sigma
    else:
        return W, Sigma


def fit_scalar_glm(Xs, ys,
                   model="bernoulli",
                   mean_function="logistic",
                   model_hypers={},
                   fit_intercept=True,
                   weights=None,
                   X_variances=None,
                   prior=None,
                   proximal_point=None,
                   threshold=1e-6,
                   step_size=1,
                   max_iter=50,
                   verbose=False):
    """
    Fit a GLM with vector inputs X and scalar outputs y.
    The user provides the inputs, outputs, the model type
    (i.e. the conditional distribution of the data), and
    the mean function that maps linear weighted inputs
    to the expected value of the output.

    The following models are supported:

        - Gaussian
        - Bernoulli
        - Poisson
        - Negative binomial (fixed r)

    Arguments
    ---------

    Xs: array of shape (n, p) or list of arrays with shapes
        [(n_1, p), (n_2, p), ..., (n_M, p)] containing
        covariates for the GLM.

    ys: array of shape (n,) or list of arrays with shapes
        [(n_1,), (n_2,), ..., (n_M,)] containing the scalar
        outputs of the GLM.

    model: string specifying the conditional distribution of
        of the data.  Currently supported values are:
            - "gaussian"
            - "bernoulli"
            - "poisson"
            - "negative binomial"

    mean_function: string or lambda function specifying the
        mapping from the projected data to the mean of the output.
        Currently supported values are:
            - "identity"
            - "logistic"
            - "exp"
            - "softplus"
        It is up to the user to make sure that the chosen mean
        function has the correct range for the corresponding model.
        For example, model="bernoulli" and mean_function="exp" will
        fail.

    model_hypers: dictionary of hyperparameters for the model.
        For example, the negative binomial requires an extra
        hyperparameter for the "number of failures".  For valid
        values of the `model_hypers`, see
        ssm.regression.model_kwarg_descriptions.

    fit_intercept: bool specifying whether or not to fit an intercept
        term. If True, the output will include the weights (an array
        of length p), and a scalar intercept value.

    weights: array of shape (n,) or list of arrays with shapes
        [(n_1,), (n_2,), ..., (n_M,)] containing non-negative weights
        associated with each data point.  For example, these are
        used when fitting mixtures of GLMs with the EM algorithm.

    X_variances: array of shape (n, p, p) or list of arrays with shapes
        [(n_1, p, p), (n_2, p, p), ..., (n_M, p, p)] containing
        the covariance of given covariates.  These are used when
        the data itself is uncertain, but where we have distributions
        q(X) and q(y) on the inputs and outputs, respectively. (We assume
        X and y are independent.)  In this case, Xs and ys are treated as
        the marginal means E[X] and E[y] respectively.  To fit the GLM,
        we also need the marginal covariances of the inputs.  These are
        specified here as an array of covariance matrices, or as a list
        of arrays of covariance matrices, one for each data point.

    prior: tuple of (mean, variance) of a Gaussian prior on the weights of
        the GLM.  The mean must be a scalar or an array of shape (p,) if
        fit_intercept is False or (p+1,) otherwise.  If scalar, it is
        multiplied by a vector of ones.  The variance can be a positive
        scalar or a (p, p) or (p+1, p+1) matrix, depending again on whether
        fit_intercept is True.

    proximal_point: tuple of (array, positive scalar) for the proximal
        point algorithm.  The array must be of shape (p,) if fit_intercept
        is False or (p+1,) otherwise.  It specifies the current value of
        the parameters that we should not deviate too far from.  The positive
        scalar specifies the inverse strength of this regularization.  As
        this values goes to zero, the fitted value must be exactly the
        proximal point given in the array. Effectively, these specify an
        another Gaussian prior, which will multiplied with the prior above.

    threshold: positive scalar value specifying the mean absolute deviation in
        weights required for convergence.

    step_size: scalar value in (0, 1] specifying the linear combination of the
        next weights and current weights.  A step size of 1 means that each
        iteration goes all the way to the mode of the quadratic approximation.

    max_iter: int, maximum number of iterations of the Newton-Raphson algorithm.

    verbose: bool, whether or not to print diagnostic messages.
    """
    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = ys if isinstance(ys, (list, tuple)) else [ys]
    assert len(Xs) == len(ys)

    p = Xs[0].shape[1]
    assert all([y.ndim == 1 for y in ys])
    assert all([X.shape[1] == p for X in Xs])
    assert all([y.shape[0] == X.shape[0] for X, y in zip(Xs, ys)])

    # Check the weights.  Default to all ones.
    if weights is not None:
        weights = weights if isinstance(weights, (list, tuple)) else [weights]
        assert all([weight.shape == (X.shape[0],) for X, weight in zip(Xs, weights)])

    else:
        weights = [np.ones(X.shape[0]) for X in Xs]

    # If the inputs are uncertain, the user may specify the marginal variance
    # of the data points.  These must be an array of (p, p) covariance matrices.
    if X_variances is not None:
        X_variances = X_variances if isinstance(X_variances, (list, tuple)) else [X_variances]
        assert all([X_var.shape == (X.shape[0], p, p) for X, X_var in zip(Xs, X_variances)])
    else:
        X_variances = [np.zeros((X.shape[0], p, p)) for X in Xs]

    # Add a column to X if fitting the intercept as well
    # Note: this could be memory intensive, but the code is a lot simpler.
    if fit_intercept:
        Xs = [np.column_stack((X, np.ones(X.shape[0]))) for X in Xs]
        new_X_variances = [np.zeros((X.shape[0], p+1, p+1)) for X in Xs]
        for X_var, new_X_var in zip(X_variances, new_X_variances):
            new_X_var[:, :p, :p] = X_var
        X_variances = new_X_variances
        p += 1

    # Check the model specification
    model = model.lower()
    assert model in ("gaussian", "bernoulli", "poisson", "negative_binomial")

    # Initialize the prior
    if prior is None:
        prior_mean = np.zeros(p)
        prior_precision = np.zeros((p, p))
    else:
        assert isinstance(prior, (tuple, list)) and len(prior) == 2
        prior_mean, prior_variance = prior
        if np.isscalar(prior_mean):
            prior_mean = prior_mean * np.ones(p)
        else:
            assert prior_mean.shape == (p,)

        if np.isscalar(prior_variance):
            assert prior_variance > 0
            prior_precision = 1 / prior_variance * np.eye(p)
        else:
            assert prior_variance.shape == (p, p)
            prior_precision = np.linalg.inv(prior_variance)

    # Incorporate the proximal point into the prior, if specified.
    if proximal_point is not None:
        # Make sure the point and the regularization strength are both specified.
        assert isinstance(proximal_point, (tuple, list)) and len(proximal_point) == 2
        point, alpha = proximal_point
        assert point.shape == (p,)
        assert np.isscalar(alpha) and alpha > 0

        # Combine the proximal point regularizer with the Gaussian prior.
        new_precision = prior_precision + 1 / alpha * np.eye(p)
        prior_mean = np.linalg.solve(new_precision, np.dot(prior_precision, prior_mean) + point / alpha)
        prior_precision = new_precision

    # Get the partition function (A) and mean function (f).
    # These determine the mapping from inputs to natural parameters (g).
    A = lambda eta: partition_functions[model](eta, **model_hypers)
    f = mean_functions[mean_function] if isinstance(mean_function, str) else mean_function
    g = lambda u: canonical_link_functions[model](f(u), **model_hypers)

    # Compute necessary derivatives for IRLS
    # When y is a scalar, these are all R^1 ->R^1 scalar functions
    df = elementwise_grad(f)
    dg = elementwise_grad(g)
    d2g = elementwise_grad(dg)
    dA = elementwise_grad(A)
    d2A = elementwise_grad(dA)

    # Construct the linear approximation for the gradient in the case of uncertain inputs
    h = lambda x, y, theta: g()

    # Initialize the weights, theta
    theta = np.zeros(p)
    dtheta = np.inf
    converged = False
    for itr in range(max_iter):
        if verbose:
            print("Iteration ", itr, "delta theta: ", dtheta)

        # Check convergence
        converged = dtheta < threshold
        if converged:
            print("Converged in ", itr, " iterations.")
            break

        # Compute the negative Hessian (J) and the gradient (h) of the objective
        J = prior_precision.copy()
        h = -np.dot(prior_precision, (theta - prior_mean))

        for X, y, weight, X_var in zip(Xs, ys, weights, X_variances):

            # Project inputs with current parameters and get predicted values
            u = np.dot(X, theta)
            yhat = f(u)

            # Compute the weights G and R
            G = dg(u)
            R = d2g(u) * (yhat - y) + G**2 * d2A(g(u))

            # Linearize the gradient for uncertain data
            H = G * (y - yhat)
            # dH = d2g(u) * (y - yhat) - dg(u) * df(u)
            dH = G * (y - yhat) - G**2 * d2A(g(u))  # nearly the same as R!

            # Update the negative Hessian
            weighted_X = X * R[:, None] * weight[:, None]
            J += np.dot(weighted_X.T, X)
            J += np.einsum('npq,n->pq', X_var, R)

            # Update the gradient
            h += np.dot(weighted_X.T, H / R)
            h += np.einsum('npq,n,q-> p', X_var, dH, theta)

        # Solve for the Newton update
        # (current parameters + negative Hessian^{-1} gradient)
        next_theta = theta + np.linalg.solve(J, h)

        # Check for convergence
        dtheta = np.mean(abs(next_theta - theta))
        theta = (1 - step_size) * theta + step_size * next_theta

    # Output warning if terminated without convergence
    if not converged:
        warn("Newtons method failed to converge in {} iterations.".format(max_iter))

    # Return the weights and intercept if necessary
    if fit_intercept:
        return theta[:-1], theta[-1]
    else:
        return theta


def fit_multiclass_logistic_regression(X, y,
                                       bias=None, K=None, W0=None, mu0=0, sigmasq0=1,
                                       verbose=False, maxiter=1000):
    """
    Fit a multiclass logistic regression

        y_i ~ Cat(softmax(W x_i))

    y is a one hot vector in {0, 1}^K
    x_i is a vector in R^D
    W is a matrix R^{K x D}

    The log likelihood is,

        L(W) = sum_i sum_k y_ik * w_k^T x_i - logsumexp(W x_i)

    The prior is w_k ~ Norm(mu0, diag(sigmasq0)).
    """
    N, D = X.shape
    assert y.shape[0] == N

    # Make sure y is one hot
    if y.ndim == 1 or y.shape[1] == 1:
        assert y.dtype == int and y.min() >= 0
        K = y.max() + 1 if K is None else K
        y_oh = np.zeros((N, K), dtype=int)
        y_oh[np.arange(N), y] = 1

    else:
        K = y.shape[1]
        assert y.min() == 0 and y.max() == 1 and np.allclose(y.sum(1), 1)
        y_oh = y

    # Check that bias is correct shape
    if bias is not None:
        assert bias.shape == (K,) or bias.shape == (N, K)
    else:
        bias = np.zeros((K,))

    def loss(W_flat):
        W = np.reshape(W_flat, (K, D))
        scores = np.dot(X, W.T) + bias
        lp = np.sum(y_oh * scores) - np.sum(logsumexp(scores, axis=1))
        prior = np.sum(-0.5 * (W - mu0)**2 / sigmasq0)
        return -(lp + prior) / N

    W0 = W0 if W0 is not None else np.zeros((K, D))
    assert W0.shape == (K, D)

    itr = [0]
    def callback(W_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0], loss(W_flat)))

    result = minimize(loss, np.ravel(W0), jac=grad(loss),
                      method="BFGS",
                      callback=callback if verbose else None,
                      options=dict(maxiter=maxiter, disp=verbose))

    W = np.reshape(result.x, (K, D))
    return W


def generalized_newton_studentst_dof(E_tau, E_logtau, nu0=2, a_nu=3, b_nu=3/2,
                                     max_iter=100, nu_min=1e-8, nu_max=100, tol=1e-8,
                                     verbose=False):
    """
    Generalized Newton's method for the degrees of freedom parameter, nu,
    of a Student's t distribution.  See the notebook in the doc/students_t
    folder for a complete derivation.

    Include a Gamma prior nu ~ Ga(a_nu, b_nu), corresponding to regularizer

    R(nu) = (a_nu - 1) * np.log(nu) - b_nu * nu
    R'(nu) = (a_nu - 1) / nu - b_nu
    R''(nu) = (1 - a_nu) / nu**2
    """
    assert a_nu > 1, "Gamma prior nu ~ Ga(a_nu, b_nu) must be log concave; i.e. a_nu must be > 1."
    delbo = lambda nu: 1/2 * (1 + np.log(nu/2)) - 1/2 * digamma(nu/2) \
            + 1/2 * E_logtau - 1/2 * E_tau + (a_nu - 1) / nu - b_nu
    ddelbo = lambda nu: 1/(2 * nu) - 1/4 * polygamma(1, nu/2) + (1 - a_nu) / nu**2

    dnu = np.inf
    nu = nu0
    for itr in range(max_iter):
        if abs(dnu) < tol:
            break

        if nu < nu_min or nu > nu_max:
            warn("generalized_newton_studentst_dof fixed point grew beyond "
                 "bounds [{},{}] to {}.".format(nu_min, nu_max, nu))
            nu = np.clip(nu, nu_min, nu_max)
            break

        # Perform the generalized Newton update
        a = -nu**2 * ddelbo(nu)
        b = delbo(nu) - a / nu
        assert a > 0 and b < 0, \
               "generalized_newton_studentst_dof failed due to nonconcave optimization. \
               Try strengthening prior via parameters a_nu and b_nu."
        dnu = -a / b - nu
        nu = nu + dnu

    if itr == max_iter - 1:
        warn("generalized_newton_studentst_dof failed to converge"
             "at tolerance {} in {} iterations.".format(tol, itr))

    return nu


def fit_negative_binomial_integer_r(xs, r_min=1, r_max=20):
    """
    Fit a negative binomial distribution NB(r, p) to data xs,
    under the constraint that the shape r is an integer.

    The durations are 1 + a negative binomial random variable.
    """
    assert isinstance(xs, np.ndarray) and xs.ndim == 1 and xs.min() >= 1
    xs -= 1
    N = len(xs)
    x_sum = np.sum(xs)

    p_star = lambda r: np.clip(x_sum / (N * r + x_sum), 1e-8, 1-1e-8)

    def nb_marginal_likelihood(r):
        # Compute the log likelihood of data with shape r and
        # MLE estimate p = sum(xs) / (N*r + sum(xs))
        ll = np.sum(gammaln(xs + r)) - np.sum(gammaln(xs + 1)) - N * gammaln(r)
        ll += np.sum(xs * np.log(p_star(r))) + N * r * np.log(1 - p_star(r))
        return ll

    # Search for the optimal r. If variance of xs exceeds the mean, the MLE exists.
    rs = np.arange(r_min, r_max+1)
    mlls = [nb_marginal_likelihood(r) for r in rs]
    r_star = rs[np.argmax(mlls)]

    return r_star, p_star(r_star)


if __name__ == "__main__":
    # Try it out with logistic regression
    npr.seed(3)
    n = 100000
    p = 20
    X = npr.randn(n, p)
    w = npr.randn(p)
    b = -3
    u = X.dot(w) + b

    # print("gaussian")
    # y = npr.randn(n) + u
    # what, bhat = fit_scalar_glm(X, y, model="gaussian", mean_function="identity")
    # print(w, b)
    # print(what, bhat)
    # print("")

    # print("gaussian with uncertain inputs")
    # y = npr.randn(n) + u
    # what, bhat = fit_scalar_glm(X, y, model="gaussian", mean_function="identity",
    #     X_variances=np.tile(0.1 * np.eye(p)[None, ...], (n, 1, 1)))
    # print(w, b)
    # print(what, bhat)
    # print("")

    # print("logistic regression")
    # y = npr.rand(n) < 1 / (1 + np.exp(-u))
    # what, bhat = fit_scalar_glm(X, y, model="bernoulli", mean_function="logistic")
    # print(w, b)
    # print(what, bhat)
    # print("")

    # print("poisson / exp")
    # y = npr.poisson(np.exp(u))
    # what, bhat = fit_scalar_glm(X, y, model="poisson", mean_function="exp", prior=(0, 10))
    # print(w, b)
    # print(what, bhat)
    # print("")

    # print("poisson / softplus")
    # y = npr.poisson(np.log1p(np.exp(u)))
    # what, bhat = fit_scalar_glm(X, y, model="poisson", mean_function="softplus")
    # print("true: ", w, b)
    # print("inf:  ", what, bhat)
    # print("")

    # r = 3
    # print("negative_binomial / logistic; r=", r)
    # y = npr.negative_binomial(r, 1 - 1 / (1 + np.exp(-u)))
    # what, bhat = fit_scalar_glm(X, y, model="negative_binomial", mean_function="exp", model_hypers=dict(r=r))
    # print("true: ", w, b)
    # print("inf:  ", what, bhat)
    # print("")

    print("poisson / softplus with uncertain data")
    y = npr.poisson(np.log1p(np.exp(u)))
    what, bhat = fit_scalar_glm(X, y, model="poisson", mean_function="softplus",
        X_variances=np.tile(0.5 * np.eye(p)[None, ...], (n, 1, 1)))
    print("true: ", w, b)
    print("inf:  ", what, bhat)
    print("")
