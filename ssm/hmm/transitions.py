import autograd.numpy as np
import autograd.numpy.random as npr

import ssm.distributions as dists
from ssm.optimizers import bfgs, convex_combination
from ssm.util import format_dataset


def make_transitions(num_states, transitions, **transition_kwargs):
    """Helper function to construct transitions of the desired type.
    """
    transition_names = dict(
        standard=StationaryTransitions,
        stationary=StationaryTransitions,
    )
    if isinstance(transitions, np.ndarray):
        # Assume this is a transition matrix
        return StationaryTransitions(num_states,
                                     transition_matrix=transitions)

    elif isinstance(transitions, str):
        # String specifies class of transitions
        assert transitions.lower() in transition_names, \
            "`transitions` must be one of {}".format(transition_names.keys())
        transition_class = transition_names[transitions.lower()]
        return transition_class(num_states, **transition_kwargs)
    else:
        # Otherwise, we need a Transitions object
        assert isinstance(transitions, Transitions)
        return transitions


class Transitions(object):
    def __init__(self, num_states):
        self.num_states = num_states

    @format_dataset
    def initialize(self, dataset):
        pass

    def permute(self, perm):
        raise NotImplementedError

    def log_prior(self):
        return 0

    def get_transition_matrix(self, **kwargs):
        """
        Compute the transition matrix for a single timestep.
        Optional kwargs include a single timestep's data and
        covariates.
        """
        raise NotImplementedError

    def log_transition_matrices(self, data, **kwargs):
        raise NotImplementedError

    @format_dataset
    def expected_log_prob(self, dataset, posteriors):
        elp = self.log_prior()
        for data_dict, posterior in zip(dataset, posteriors):
            log_P = self.log_transition_matrices(**data_dict)
            elp += np.sum(posterior.expected_transitions * log_P)
        return elp

    @format_dataset
    def m_step(self, dataset, posteriors,
               num_iters=100,
               **kwargs):
        """
        By default, maximize the expected log likelihood with BFGS.
        """
        # Normalize and negate for minimization
        T = sum([data_dict["data"].shape[0] for data_dict in dataset])
        def _objective(params, itr):
            self.unconstrained_params = params
            return -self.expected_log_prob(dataset, posteriors) / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls
        # to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            bfgs(_objective, self.params, num_iters=num_iters,
                 state=optimizer_state, full_output=True, **kwargs)


class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and
    transition matrix.
    """
    def __init__(self, num_states, transition_matrix=None):
        super(StationaryTransitions, self).__init__(num_states)
        if transition_matrix is None:
            # default to a weakly sticky transition matrix
            self.set_transition_matrix(
                0.8 * np.eye(num_states) + \
                0.2 / (num_states - 1) * (1 - np.eye(num_states)))
        else:
            self.set_transition_matrix(transition_matrix)

        # Initialize state for stochastic EM
        self._stochastic_m_step_state = None

    def get_transition_matrix(self, **kwargs):
        return np.row_stack([d.probs for d in self.conditional_dists])

    def set_transition_matrix(self, value):
        assert value.shape == (self.num_states, self.num_states)
        assert np.allclose(np.sum(value, axis=1), 1.0)
        self.conditional_dists = [
            dists.Categorical(row) for row in value
        ]

    def log_prior(self):
        return np.sum([d.log_prior() for d in self.conditional_dists])

    def log_transition_matrices(self, data, **kwargs):
        with np.errstate(divide='ignore'):
            return np.log(self.get_transition_matrix())

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.set_transition_matrix(
            self.get_transition_matrix()[np.ix_(perm, perm)]
        )

    @format_dataset
    def m_step(self, dataset, posteriors):
        """Exact M-step is possible for the standard transition model.
        """
        expected_transitions = sum([posterior.expected_transitions()
                                    for posterior in posteriors])

        for row, dist in zip(expected_transitions, self.conditional_dists):
            dist.fit_expfam_with_stats((row,), row.sum())

    @format_dataset
    def stochastic_m_step(self, dataset, posteriors, step_size,
                          scale_factor=1.0,
                          **kwargs):
        # Compute expected sufficient statistics (transition counts)
        # for this dataset
        expected_transitions = \
            scale_factor * sum([posterior.expected_transitions()
                                for posterior in posteriors])

        # Take a convex combination with past sufficient statistics
        if self._stochastic_m_step_state is not None:
            # Take a convex combination of sufficient statistics from
            # this batch and those accumulated thus far.
            expected_transitions = convex_combination(
                self._stochastic_m_step_state["expected_transitions"],
                expected_transitions,
                step_size)

        # Update the transition matrix using effective suff stats
        for row, dist in zip(expected_transitions, self.conditional_dists):
            dist.fit_expfam_with_stats((row,), row.sum())

        # Save the updated sufficient statistics
        self._stochastic_m_step_state = dict(
            expected_transitions=expected_transitions
        )



# class ConstrainedStationaryTransitions(StationaryTransitions):
#     """
#     Standard Hidden Markov Model with fixed transition matrix.
#     Allows the user to specify some entries of the transition matrix to be zeros,
#     in order to prohibit certain transitions.

#     The user passes an array, `mask`, which must be the same size
#     as the transition matrix. Entries of the mask which are zero
#     correspond to entries in the transition matrix which will be
#     fixed at zero.
#     """
#     def __init__(self, K, D, transition_mask=None, M=0):
#         super(ConstrainedStationaryTransitions, self).__init__(K, D, M=M)
#         Ps = self.transition_matrix
#         if transition_mask is None:
#             transition_mask = np.ones_like(Ps, dtype=bool)
#         else:
#             transition_mask = transition_mask.astype(bool)

#         # Validate the transition mask. A valid mask must have be the same shape
#         # as the transition matrix, contain only ones and zeros, and contain at
#         # least one non-zero entry per row.
#         assert transition_mask.shape == Ps.shape, "Mask must be the same size " \
#             "as the transition matrix. Found mask of shape {}".format(transition_mask.shape)
#         assert np.isin(transition_mask,[1,0]).all(), "Mask must contain only 1s and zeros."
#         for i in range(transition_mask.shape[0]):
#             assert transition_mask[i].any(), "Mask must contain at least one " \
#                 "nonzero entry per row."

#         self.transition_mask = transition_mask
#         Ps = Ps * transition_mask
#         Ps /= Ps.sum(axis=-1, keepdims=True)
#         self.log_Ps = np.log(Ps)
#         self.log_Ps[~transition_mask] = -np.inf

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         super(ConstrainedStationaryTransitions, self).m_step(
#             expectations,
#             datas,
#             inputs,
#             masks,
#             tags,
#             **kwargs
#         )
#         assert np.allclose(self.transition_matrix[~self.transition_mask], 0,
#                            atol=2 * LOG_EPS)
#         self.log_Ps[~self.transition_mask] = -np.inf


# class StickyTransitions(StationaryTransitions):
#     """
#     Upweight the self transition prior.

#     pi_k ~ Dir(alpha + kappa * e_k)
#     """
#     def __init__(self, K, D, M=0, alpha=1, kappa=100):
#         super(StickyTransitions, self).__init__(K, D, M=M)
#         self.alpha = alpha
#         self.kappa = kappa

#     def log_prior(self):
#         K = self.K
#         log_P = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)

#         lp = 0
#         for k in range(K):
#             alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
#             lp += np.dot((alpha - 1), log_P[k])
#         return lp

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         expected_joints = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
#         expected_joints += self.kappa * np.eye(self.K) + (self.alpha-1) * np.ones((self.K, self.K))
#         P = (expected_joints / expected_joints.sum(axis=1, keepdims=True)) + 1e-16
#         assert np.all(P >= 0), "mode is well defined only when transition matrix entries are non-negative! Check alpha >= 1"
#         self.log_Ps = np.log(P)


# class InputDrivenTransitions(StickyTransitions):
#     """
#     Hidden Markov Model whose transition probabilities are
#     determined by a generalized linear model applied to the
#     exogenous input.
#     """
#     def __init__(self, K, D, M, alpha=1, kappa=0, l2_penalty=0.0):
#         super(InputDrivenTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

#         # Parameters linking input to state distribution
#         self.Ws = npr.randn(K, M)

#         # Regularization of Ws
#         self.l2_penalty = l2_penalty

#     @property
#     def params(self):
#         return self.log_Ps, self.Ws

#     @params.setter
#     def params(self, value):
#         self.log_Ps, self.Ws = value

#     def permute(self, perm):
#         """
#         Permute the discrete latent states.
#         """
#         self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
#         self.Ws = self.Ws[perm]

#     def log_prior(self):
#         lp = super(InputDrivenTransitions, self).log_prior()
#         lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ws**2)
#         return lp

#     def log_transition_matrices(self, data, input, mask, tag):
#         T = data.shape[0]
#         assert input.shape[0] == T
#         # Previous state effect
#         log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
#         # Input effect
#         log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
#         return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

#     def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
#         # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
#         T, D = data.shape
#         return np.zeros((T-1, D, D))

# class RecurrentTransitions(InputDrivenTransitions):
#     """
#     Generalization of the input driven HMM in which the observations serve as future inputs
#     """
#     def __init__(self, K, D, M=0, alpha=1, kappa=0):
#         super(RecurrentTransitions, self).__init__(K, D, M, alpha=alpha, kappa=kappa)

#         # Parameters linking past observations to state distribution
#         self.Rs = np.zeros((K, D))

#     @property
#     def params(self):
#         return super(RecurrentTransitions, self).params + (self.Rs,)

#     @params.setter
#     def params(self, value):
#         self.Rs = value[-1]
#         super(RecurrentTransitions, self.__class__).params.fset(self, value[:-1])

#     def permute(self, perm):
#         """
#         Permute the discrete latent states.
#         """
#         super(RecurrentTransitions, self).permute(perm)
#         self.Rs = self.Rs[perm]

#     def log_transition_matrices(self, data, input, mask, tag):
#         T, D = data.shape
#         # Previous state effect
#         log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
#         # Input effect
#         log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
#         # Past observations effect
#         log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
#         return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

#     def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
#         # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
#         T, D = data.shape
#         hess = np.zeros((T-1,D,D))
#         vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
#         Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
#         for k in range(self.K):
#             vtilde = vtildes[:,k,:] # normalized probabilities given state k
#             Rv = vtilde @ self.Rs
#             hess += Ez[:,k][:,None,None] * \
#                     ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
#                     + np.einsum('ti, tj -> tij', Rv, Rv))
#         return -1 * hess

# class RecurrentOnlyTransitions(Transitions):
#     """
#     Only allow the past observations and inputs to influence the
#     next state.  Get rid of the transition matrix and replace it
#     with a constant bias r.
#     """
#     def __init__(self, K, D, M=0):
#         super(RecurrentOnlyTransitions, self).__init__(K, D, M)

#         # Parameters linking past observations to state distribution
#         self.Ws = npr.randn(K, M)
#         self.Rs = npr.randn(K, D)
#         self.r = npr.randn(K)

#     @property
#     def params(self):
#         return self.Ws, self.Rs, self.r

#     @params.setter
#     def params(self, value):
#         self.Ws, self.Rs, self.r = value

#     def permute(self, perm):
#         """
#         Permute the discrete latent states.
#         """
#         self.Ws = self.Ws[perm]
#         self.Rs = self.Rs[perm]
#         self.r = self.r[perm]

#     def log_transition_matrices(self, data, input, mask, tag):
#         T, D = data.shape
#         log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
#         log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
#         log_Ps = log_Ps + self.r                                       # bias
#         log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
#         return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

#     def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
#         # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
#         v = np.dot(input[1:], self.Ws.T) + np.dot(data[:-1], self.Rs.T) + self.r
#         shifted_exp = np.exp(v - np.max(v,axis=1,keepdims=True))
#         vtilde = shifted_exp / np.sum(shifted_exp,axis=1,keepdims=True) # normalized probabilities
#         Rv = vtilde@self.Rs
#         hess = np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
#                + np.einsum('ti, tj -> tij', Rv, Rv)
#         return -1 * hess


# class NegativeBinomialSemiMarkovTransitions(Transitions):
#     """
#     Semi-Markov transition model with negative binomial (NB) distributed
#     state durations, as compared to the geometric state durations in the
#     standard Markov model.  The negative binomial has higher variance than
#     the geometric, but its mode can be greater than 1.

#     The NB(r, p) distribution, with r a positive integer and p a probability
#     in [0, 1], is this distribution over number of heads before seeing
#     r tails where the probability of heads is p. The number of heads
#     between each tails is an independent geometric random variable.  Thus,
#     the total number of heads is the sum of r independent and identically
#     distributed geometric random variables.

#     We can "embed" the semi-Markov model with negative binomial durations
#     in the standard Markov model by expanding the state space.  Map each
#     discrete state k to r new states: (k,1), (k,2), ..., (k,r_k),
#     for k in 1, ..., K. The total number of states is \sum_k r_k,
#     where state k has a NB(r_k, p_k) duration distribution.

#     The transition probabilities are as follows. The probability of staying
#     within the same "super state" are:

#     p(z_{t+1} = (k,i) | z_t = (k,i)) = p_k

#     and for 0 <= j <= r_k - i

#     p(z_{t+1} = (k,i+j) | z_t = (k,i)) = (1-p_k)^{j-i} p_k

#     The probability of flipping (r_k - i + 1) tails in a row in state k;
#     i.e. the probability of exiting super state k, is (1-p_k)^{r_k-i+1}.
#     Thus, the probability of transitioning to a new super state is:

#     p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * P[k, j]

#     where P[k, j] is a transition matrix with zero diagonal.

#     As a sanity check, note that the sum of probabilities is indeed 1:

#     \sum_{j=i}^{r_k} p(z_{t+1} = (k,j) | z_t = (k,i))
#         + \sum_{m \neq k}  p(z_{t+1} = (m, 1) | z_t = (k, i))

#     = \sum_{j=0}^{r_k-i} (1-p_k)^j p_k + \sum_{m \neq k} (1-p_k)^{r_k-i+1} * P[k, j]

#     = p_k (1-(1-p_k)^{r_k-i+1}) / (1-(1-p_k)) + (1-p_k)^{r_k-i+1}

#     = 1 - (1-p_k)^{r_k-i+1} + (1 - p_k)^{r_k-i+1}

#     = 1.

#     where we used the geometric series and the fact that \sum_{j != k} P[k, j] = 1.
#     """
#     def __init__(self, K, D, M=0, r_min=1, r_max=20):
#         assert K > 1, "Explicit duration models only work if num states > 1."
#         super(NegativeBinomialSemiMarkovTransitions, self).__init__(K, D, M=M)

#         # Initialize the super state transition probabilities
#         self.Ps = npr.rand(K, K)
#         np.fill_diagonal(self.Ps, 0)
#         self.Ps /= self.Ps.sum(axis=1, keepdims=True)

#         # Initialize the negative binomial duration probabilities
#         self.r_min, self.r_max = r_min, r_max
#         self.rs = npr.randint(r_min, r_max + 1, size=K)
#         # self.rs = np.ones(K, dtype=int)
#         # self.ps = npr.rand(K)
#         self.ps = 0.5 * np.ones(K)

#         # Initialize the transition matrix
#         self._transition_matrix = None

#     @property
#     def params(self):
#         return (self.Ps, self.rs, self.ps)

#     @params.setter
#     def params(self, value):
#         Ps, rs, ps = value
#         assert Ps.shape == (self.K, self.K)
#         assert np.allclose(np.diag(Ps), 0)
#         assert np.allclose(Ps.sum(1), 1)
#         assert rs.shape == (self.K)
#         assert rs.dtype == int
#         assert np.all(rs > 0)
#         assert ps.shape == (self.K)
#         assert np.all(ps > 0)
#         assert np.all(ps < 1)
#         self.Ps, self.rs, self.ps = Ps, rs, ps

#         # Reset the transition matrix
#         self._transition_matrix = None

#     def permute(self, perm):
#         """
#         Permute the discrete latent states.
#         """
#         self.Ps = self.Ps[np.ix_(perm, perm)]
#         self.rs = self.rs[perm]
#         self.ps = self.ps[perm]

#         # Reset the transition matrix
#         self._transition_matrix = None

#     @property
#     def total_num_states(self):
#         return np.sum(self.rs)

#     @property
#     def state_map(self):
#         return np.repeat(np.arange(self.K), self.rs)

#     @property
#     def transition_matrix(self):
#         if self._transition_matrix is not None:
#             return self._transition_matrix

#         As, rs, ps = self.Ps, self.rs, self.ps

#         # Fill in the transition matrix one block at a time
#         K_total = self.total_num_states
#         P = np.zeros((K_total, K_total))
#         starts = np.concatenate(([0], np.cumsum(rs)[:-1]))
#         ends = np.cumsum(rs)
#         for (i, j), Aij in np.ndenumerate(As):
#             block = P[starts[i]:ends[i], starts[j]:ends[j]]

#             # Diagonal blocks (stay in sub-state or advance to next sub-state)
#             if i == j:
#                 for k in range(rs[i]):
#                     # p(z_{t+1} = (.,i+k) | z_t = (.,i)) = (1-p)^k p
#                     # for 0 <= k <= r - i
#                     block += (1 - ps[i])**k * ps[i] * np.diag(np.ones(rs[i]-k), k=k)

#             # Off-diagonal blocks (exit to a new super state)
#             else:
#                 # p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * A[k, j]
#                 block[:,0] = (1-ps[i]) ** np.arange(rs[i], 0, -1) * Aij

#         assert np.allclose(P.sum(1),1)
#         assert (0 <= P).all() and (P <= 1.).all()

#         # Cache the transition matrix
#         self._transition_matrix = P

#         return P

#     def log_transition_matrices(self, data, input, mask, tag):
#         T = data.shape[0]
#         P = self.transition_matrix
#         return np.tile(np.log(P)[None, :, :], (T-1, 1, 1))

#     def m_step(self, expectations, datas, inputs, masks, tags, samples, **kwargs):
#         # Update the transition matrix between super states
#         P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
#         np.fill_diagonal(P, 0)
#         P /= P.sum(axis=-1, keepdims=True)
#         self.Ps = P

#         # Fit negative binomial models for each duration based on sampled states
#         states, durations = map(np.concatenate, zip(*[rle(z_smpl) for z_smpl in samples]))
#         for k in range(self.K):
#             self.rs[k], self.ps[k] = \
#                 fit_negative_binomial_integer_r(durations[states == k], self.r_min, self.r_max)

#         # Reset the transition matrix
#         self._transition_matrix = None
