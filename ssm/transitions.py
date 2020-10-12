from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet
from autograd import hessian

from ssm.util import one_hot, logistic, relu, rle, ensure_args_are_lists, LOG_EPS, DIV_EPS
from ssm.regression import fit_multiclass_logistic_regression, fit_negative_binomial_integer_r
from ssm.stats import multivariate_normal_logpdf
from ssm.optimizers import adam, bfgs, lbfgs, rmsprop, sgd


class Transitions(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        return 0

    def log_transition_matrices(self, data, input, mask, tag):
        raise NotImplementedError

    def transition_matrices(self, data, input, mask, tag):
        return np.exp(self.log_transition_matrices(data, input, mask, tag))

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="lbfgs", num_iters=1000, **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to BFGS.
        """
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[optimizer]

        # Maximize the expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # Normalize and negate for minimization
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            optimizer(_objective, self.params, num_iters=num_iters,
                      state=optimizer_state, full_output=True, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        warn("Analytical Hessian is not implemented for this transition class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        obj = lambda x, E_zzp1: np.sum(E_zzp1 * self.log_transition_matrices(x, input, mask, tag))
        hess = hessian(obj)
        terms = np.array([-1 * hess(x[None,:], Ezzp1) for x, Ezzp1 in zip(data, expected_joints)])
        return terms

class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M=M)
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K = self.K
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-32
        P = np.nan_to_num(P / P.sum(axis=-1, keepdims=True))

        # Set rows that are all zero to uniform
        P = np.where(P.sum(axis=-1, keepdims=True) == 0, 1.0 / K, P)
        log_P = np.log(P)
        self.log_Ps = log_P - logsumexp(log_P, axis=-1, keepdims=True)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))


class ConstrainedStationaryTransitions(StationaryTransitions):
    """
    Standard Hidden Markov Model with fixed transition matrix.
    Allows the user to specify some entries of the transition matrix to be zeros,
    in order to prohibit certain transitions.

    The user passes an array, `mask`, which must be the same size
    as the transition matrix. Entries of the mask which are zero
    correspond to entries in the transition matrix which will be
    fixed at zero.
    """
    def __init__(self, K, D, transition_mask=None, M=0):
        super(ConstrainedStationaryTransitions, self).__init__(K, D, M=M)
        Ps = self.transition_matrix
        if transition_mask is None:
            transition_mask = np.ones_like(Ps, dtype=bool)
        else:
            transition_mask = transition_mask.astype(bool)

        # Validate the transition mask. A valid mask must have be the same shape
        # as the transition matrix, contain only ones and zeros, and contain at
        # least one non-zero entry per row.
        assert transition_mask.shape == Ps.shape, "Mask must be the same size " \
            "as the transition matrix. Found mask of shape {}".format(transition_mask.shape)
        assert np.isin(transition_mask,[1,0]).all(), "Mask must contain only 1s and zeros."
        for i in range(transition_mask.shape[0]):
            assert transition_mask[i].any(), "Mask must contain at least one " \
                "nonzero entry per row."

        self.transition_mask = transition_mask
        Ps = Ps * transition_mask
        Ps /= Ps.sum(axis=-1, keepdims=True)
        self.log_Ps = np.log(Ps)
        self.log_Ps[~transition_mask] = -np.inf

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        super(ConstrainedStationaryTransitions, self).m_step(
            expectations,
            datas,
            inputs,
            masks,
            tags,
            **kwargs
        )
        assert np.allclose(self.transition_matrix[~self.transition_mask], 0,
                           atol=2 * LOG_EPS)
        self.log_Ps[~self.transition_mask] = -np.inf


class StickyTransitions(StationaryTransitions):
    """
    Upweight the self transition prior.

    pi_k ~ Dir(alpha + kappa * e_k)
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=100):
        super(StickyTransitions, self).__init__(K, D, M=M)
        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        K = self.K
        log_P = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)

        lp = 0
        for k in range(K):
            alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
            lp += np.dot((alpha - 1), log_P[k])
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        expected_joints = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        expected_joints += self.kappa * np.eye(self.K) + (self.alpha-1) * np.ones((self.K, self.K))
        P = (expected_joints / expected_joints.sum(axis=1, keepdims=True)) + 1e-16
        assert np.all(P >= 0), "mode is well defined only when transition matrix entries are non-negative! Check alpha >= 1"
        self.log_Ps = np.log(P)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

class InputDrivenTransitions(StickyTransitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a generalized linear model applied to the
    exogenous input.
    """
    def __init__(self, K, D, M, alpha=1, kappa=0, l2_penalty=0.0):
        super(InputDrivenTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(K, M)

        # Regularization of Ws
        self.l2_penalty = l2_penalty

    @property
    def params(self):
        return self.log_Ps, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.Ws = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_prior(self):
        lp = super(InputDrivenTransitions, self).log_prior()
        lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ws**2)
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

class RecurrentTransitions(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0):
        super(RecurrentTransitions, self).__init__(K, D, M, alpha=alpha, kappa=kappa)

        # Parameters linking past observations to state distribution
        self.Rs = np.zeros((K, D))

    @property
    def params(self):
        return super(RecurrentTransitions, self).params + (self.Rs,)

    @params.setter
    def params(self, value):
        self.Rs = value[-1]
        super(RecurrentTransitions, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(RecurrentTransitions, self).permute(perm)
        self.Rs = self.Rs[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, _ = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        for k in range(self.K):
            vtilde = vtildes[:,k,:] # normalized probabilities given state k
            Rv = vtilde @ self.Rs
            hess += Ez[:,k][:,None,None] * \
                    ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
                    + np.einsum('ti, tj -> tij', Rv, Rv))
        return -1 * hess

class RecurrentOnlyTransitions(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0):
        super(RecurrentOnlyTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M)
        self.Rs = npr.randn(K, D)
        self.r = npr.randn(K)

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        v = np.dot(input[1:], self.Ws.T) + np.dot(data[:-1], self.Rs.T) + self.r
        shifted_exp = np.exp(v - np.max(v,axis=1,keepdims=True))
        vtilde = shifted_exp / np.sum(shifted_exp,axis=1,keepdims=True) # normalized probabilities
        Rv = vtilde@self.Rs
        hess = np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
               + np.einsum('ti, tj -> tij', Rv, Rv)
        return -1 * hess


class RBFRecurrentTransitions(InputDrivenTransitions):
    """
    Recurrent transitions with radial basis functions for parameterizing
    the next state probability given current continuous data. We have,

    p(z_{t+1} = k | z_t, x_t)
        \propto N(x_t | \mu_k, \Sigma_k) \times \pi_{z_t, z_{t+1})

    where {\mu_k, \Sigma_k, \pi_k}_{k=1}^K are learned parameters.
    Equivalently,

    log p(z_{t+1} = k | z_t, x_t)
        = log N(x_t | \mu_k, \Sigma_k) + log \pi_{z_t, z_{t+1}) + const
        = -D/2 log(2\pi) -1/2 log |Sigma_k|
          -1/2 (x - \mu_k)^T \Sigma_k^{-1} (x-\mu_k)
          + log \pi{z_t, z_{t+1}}

    The difference between this and the recurrent model above is that the
    log transition matrices are quadratic functions of x rather than linear.

    While we're at it, there's no harm in adding a linear term to the log
    transition matrices to capture input dependencies.
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0):
        super(RBFRecurrentTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # RBF parameters
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)

    @property
    def params(self):
        return self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws = value

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Fit a GMM to the data to set the means and covariances
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(self.K, covariance_type="full")
        gmm.fit(np.vstack(datas))
        self.mus = gmm.means_
        self._sqrt_Sigmas = np.linalg.cholesky(gmm.covariances_)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.mus = self.mus[perm]
        self.sqrt_Sigmas = self.sqrt_Sigmas[perm]
        self.Ws = self.Ws[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        assert np.all(mask), "Recurrent models require that all data are present."

        T = data.shape[0]
        assert input.shape[0] == T
        K, D = self.K, self.D

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))

        # RBF recurrent function
        rbf = multivariate_normal_logpdf(data[:-1, None, :], self.mus, self.Sigmas)
        log_Ps = log_Ps + rbf[:, None, :]

        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


# Allow general nonlinear emission models with neural networks
class NeuralNetworkRecurrentTransitions(Transitions):
    def __init__(self, K, D, M=0, hidden_layer_sizes=(50,), nonlinearity="relu"):
        super(NeuralNetworkRecurrentTransitions, self).__init__(K, D, M=M)

        # Baseline transition probabilities
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Initialize the NN weights
        layer_sizes = (D + M,) + hidden_layer_sizes + (K,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(
            relu=relu,
            tanh=np.tanh,
            sigmoid=logistic)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.log_Ps, self.weights, self.biases

    @params.setter
    def params(self, value):
        self.log_Ps, self.weights, self.biases = value

    def permute(self, perm):
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:,perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_transition_matrices(self, data, input, mask, tag):
        # Pass the data and inputs through the neural network
        x = np.hstack((data[:-1], input[1:]))
        for W, b in zip(self.weights, self.biases):
            y = np.dot(x, W) + b
            x = self.nonlinearity(y)

        # Add the baseline transition biases
        log_Ps = self.log_Ps[None, :, :] + y[:, None, :]

        # Normalize
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=100, **kwargs):
        # Default to adam instead of bfgs for the neural network model.
        Transitions.m_step(self, expectations, datas, inputs, masks, tags,
            optimizer=optimizer, num_iters=num_iters, **kwargs)


class NegativeBinomialSemiMarkovTransitions(Transitions):
    """
    Semi-Markov transition model with negative binomial (NB) distributed
    state durations, as compared to the geometric state durations in the
    standard Markov model.  The negative binomial has higher variance than
    the geometric, but its mode can be greater than 1.

    The NB(r, p) distribution, with r a positive integer and p a probability
    in [0, 1], is this distribution over number of heads before seeing
    r tails where the probability of heads is p. The number of heads
    between each tails is an independent geometric random variable.  Thus,
    the total number of heads is the sum of r independent and identically
    distributed geometric random variables.

    We can "embed" the semi-Markov model with negative binomial durations
    in the standard Markov model by expanding the state space.  Map each
    discrete state k to r new states: (k,1), (k,2), ..., (k,r_k),
    for k in 1, ..., K. The total number of states is \sum_k r_k,
    where state k has a NB(r_k, p_k) duration distribution.

    The transition probabilities are as follows. The probability of staying
    within the same "super state" are:

    p(z_{t+1} = (k,i) | z_t = (k,i)) = p_k

    and for 0 <= j <= r_k - i

    p(z_{t+1} = (k,i+j) | z_t = (k,i)) = (1-p_k)^{j-i} p_k

    The probability of flipping (r_k - i + 1) tails in a row in state k;
    i.e. the probability of exiting super state k, is (1-p_k)^{r_k-i+1}.
    Thus, the probability of transitioning to a new super state is:

    p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * P[k, j]

    where P[k, j] is a transition matrix with zero diagonal.

    As a sanity check, note that the sum of probabilities is indeed 1:

    \sum_{j=i}^{r_k} p(z_{t+1} = (k,j) | z_t = (k,i))
        + \sum_{m \neq k}  p(z_{t+1} = (m, 1) | z_t = (k, i))

    = \sum_{j=0}^{r_k-i} (1-p_k)^j p_k + \sum_{m \neq k} (1-p_k)^{r_k-i+1} * P[k, j]

    = p_k (1-(1-p_k)^{r_k-i+1}) / (1-(1-p_k)) + (1-p_k)^{r_k-i+1}

    = 1 - (1-p_k)^{r_k-i+1} + (1 - p_k)^{r_k-i+1}

    = 1.

    where we used the geometric series and the fact that \sum_{j != k} P[k, j] = 1.
    """
    def __init__(self, K, D, M=0, r_min=1, r_max=20):
        assert K > 1, "Explicit duration models only work if num states > 1."
        super(NegativeBinomialSemiMarkovTransitions, self).__init__(K, D, M=M)

        # Initialize the super state transition probabilities
        self.Ps = npr.rand(K, K)
        np.fill_diagonal(self.Ps, 0)
        self.Ps /= self.Ps.sum(axis=1, keepdims=True)

        # Initialize the negative binomial duration probabilities
        self.r_min, self.r_max = r_min, r_max
        self.rs = npr.randint(r_min, r_max + 1, size=K)
        # self.rs = np.ones(K, dtype=int)
        # self.ps = npr.rand(K)
        self.ps = 0.5 * np.ones(K)

        # Initialize the transition matrix
        self._transition_matrix = None

    @property
    def params(self):
        return (self.Ps, self.rs, self.ps)

    @params.setter
    def params(self, value):
        Ps, rs, ps = value
        assert Ps.shape == (self.K, self.K)
        assert np.allclose(np.diag(Ps), 0)
        assert np.allclose(Ps.sum(1), 1)
        assert rs.shape == (self.K)
        assert rs.dtype == int
        assert np.all(rs > 0)
        assert ps.shape == (self.K)
        assert np.all(ps > 0)
        assert np.all(ps < 1)
        self.Ps, self.rs, self.ps = Ps, rs, ps

        # Reset the transition matrix
        self._transition_matrix = None

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ps = self.Ps[np.ix_(perm, perm)]
        self.rs = self.rs[perm]
        self.ps = self.ps[perm]

        # Reset the transition matrix
        self._transition_matrix = None

    @property
    def total_num_states(self):
        return np.sum(self.rs)

    @property
    def state_map(self):
        return np.repeat(np.arange(self.K), self.rs)

    @property
    def transition_matrix(self):
        if self._transition_matrix is not None:
            return self._transition_matrix

        As, rs, ps = self.Ps, self.rs, self.ps

        # Fill in the transition matrix one block at a time
        K_total = self.total_num_states
        P = np.zeros((K_total, K_total))
        starts = np.concatenate(([0], np.cumsum(rs)[:-1]))
        ends = np.cumsum(rs)
        for (i, j), Aij in np.ndenumerate(As):
            block = P[starts[i]:ends[i], starts[j]:ends[j]]

            # Diagonal blocks (stay in sub-state or advance to next sub-state)
            if i == j:
                for k in range(rs[i]):
                    # p(z_{t+1} = (.,i+k) | z_t = (.,i)) = (1-p)^k p
                    # for 0 <= k <= r - i
                    block += (1 - ps[i])**k * ps[i] * np.diag(np.ones(rs[i]-k), k=k)

            # Off-diagonal blocks (exit to a new super state)
            else:
                # p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * A[k, j]
                block[:,0] = (1-ps[i]) ** np.arange(rs[i], 0, -1) * Aij

        assert np.allclose(P.sum(1),1)
        assert (0 <= P).all() and (P <= 1.).all()

        # Cache the transition matrix
        self._transition_matrix = P

        return P

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        P = self.transition_matrix
        return np.tile(np.log(P)[None, :, :], (T-1, 1, 1))

    def m_step(self, expectations, datas, inputs, masks, tags, samples, **kwargs):
        # Update the transition matrix between super states
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        np.fill_diagonal(P, 0)
        P /= P.sum(axis=-1, keepdims=True)
        self.Ps = P

        # Fit negative binomial models for each duration based on sampled states
        states, durations = map(np.concatenate, zip(*[rle(z_smpl) for z_smpl in samples]))
        for k in range(self.K):
            self.rs[k], self.ps[k] = \
                fit_negative_binomial_integer_r(durations[states == k], self.r_min, self.r_max)

        # Reset the transition matrix
        self._transition_matrix = None
