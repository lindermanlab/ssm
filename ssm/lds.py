import copy
import warnings
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad, grad

from .optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, bfgs, convex_combination
from .optimizers import adam, sgd, rmsprop
from .primitives import hmm_normalizer, hmm_expected_states, hmm_filter, \
    hmm_sample, viterbi, symm_block_tridiag_matmul
from .util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, \
    replicate, collapse, newtons_method_block_tridiag_hessian

from . import observations as obs
from . import transitions as trans
from . import init_state_distns as isd
from . import hierarchical as hier
from . import emissions as emssn
from . import hmm
from . import variational as varinf

from scipy.optimize import minimize

__all__ = ['SLDS', 'LDS']

class SLDS(object):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, K, D, *, M=0,
                 init_state_distn=None,
                 transitions="standard",
                 transition_kwargs=None,
                 hierarchical_transition_tags=None,
                 dynamics="gaussian",
                 dynamics_kwargs=None,
                 hierarchical_dynamics_tags=None,
                 emissions="gaussian_orthog",
                 emission_kwargs=None,
                 hierarchical_emission_tags=None,
                 single_subspace=True,
                 **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        assert isinstance(init_state_distn, isd.InitialStateDistribution)

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            sticky=trans.StickyTransitions,
            inputdriven=trans.InputDrivenTransitions,
            recurrent=trans.RecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            rbf_recurrent=trans.RBFRecurrentTransitions,
            nn_recurrent=trans.NeuralNetworkRecurrentTransitions
            )

        if isinstance(transitions, str):
            transitions = transitions.lower()
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            gaussian=obs.AutoRegressiveObservations,
            diagonal_gaussian=obs.AutoRegressiveDiagonalNoiseObservations,
            t=obs.RobustAutoRegressiveObservations,
            studentst=obs.RobustAutoRegressiveObservations,
            diagonal_t=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_studentst=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](K, D, M=M, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        # Make the emission distn
        emission_classes = dict(
            gaussian=emssn.GaussianEmissions,
            gaussian_orthog=emssn.GaussianOrthogonalEmissions,
            gaussian_id=emssn.GaussianIdentityEmissions,
            gaussian_nn=emssn.GaussianNeuralNetworkEmissions,
            studentst=emssn.StudentsTEmissions,
            studentst_orthog=emssn.StudentsTOrthogonalEmissions,
            studentst_id=emssn.StudentsTIdentityEmissions,
            studentst_nn=emssn.StudentsTNeuralNetworkEmissions,
            t=emssn.StudentsTEmissions,
            t_orthog=emssn.StudentsTOrthogonalEmissions,
            t_id=emssn.StudentsTIdentityEmissions,
            t_nn=emssn.StudentsTNeuralNetworkEmissions,
            poisson=emssn.PoissonEmissions,
            poisson_orthog=emssn.PoissonOrthogonalEmissions,
            poisson_id=emssn.PoissonIdentityEmissions,
            poisson_nn=emssn.PoissonNeuralNetworkEmissions,
            bernoulli=emssn.BernoulliEmissions,
            bernoulli_orthog=emssn.BernoulliOrthogonalEmissions,
            bernoulli_id=emssn.BernoulliIdentityEmissions,
            bernoulli_nn=emssn.BernoulliNeuralNetworkEmissions,
            ar=emssn.AutoRegressiveEmissions,
            ar_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            ar_id=emssn.AutoRegressiveIdentityEmissions,
            ar_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            autoregressive=emssn.AutoRegressiveEmissions,
            autoregressive_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            autoregressive_id=emssn.AutoRegressiveIdentityEmissions,
            autoregressive_nn=emssn.AutoRegressiveNeuralNetworkEmissions
            )

        if isinstance(emissions, str):
            emissions = emissions.lower()
            if emissions not in emission_classes:
                raise Exception("Invalid emission model: {}. Must be one of {}".
                    format(emissions, list(emission_classes.keys())))

            emission_kwargs = emission_kwargs or {}
            emissions = emission_classes[emissions](N, K, D, M=M,
                single_subspace=single_subspace, **emission_kwargs)
        if not isinstance(emissions, emssn.Emissions):
            raise TypeError("'emissions' must be a subclass of"
                            " ssm.emissions.Emissions")

        self.N, self.K, self.D, self.M = N, K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params, \
               self.emissions.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.dynamics.params = value[2]
        self.emissions.params = value[3]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # First initialize the observation model
        self.emissions.initialize(datas, inputs, masks, tags)

        # Get the initialized variational mean for the data
        xs = [self.emissions.invert(data, input, mask, tag)
              for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        xmasks = [np.ones_like(x, dtype=bool) for x in xs]

        # Now run a few iterations of EM on a ARHMM with the variational mean
        print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
        arhmm = hmm.HMM(self.K, self.D, M=self.M,
                        init_state_distn=copy.deepcopy(self.init_state_distn),
                        transitions=copy.deepcopy(self.transitions),
                        observations=copy.deepcopy(self.dynamics))

        arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags,
                  method="em", num_em_iters=num_em_iters)

        self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        self.transitions = copy.deepcopy(arhmm.transitions)
        self.dynamics = copy.deepcopy(arhmm.observations)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.dynamics.permute(perm)
        self.emissions.permute(perm)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.dynamics.log_prior() + \
               self.emissions.log_prior()

    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)

        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            x = np.zeros((T+1,) + D)
            data = np.zeros((T+1,) + D)
            # input = np.zeros((T+1,) + M) if input is None else input
            input = np.zeros((T+1,) + M) if input is None else np.concatenate((np.zeros((1,) + M), input))
            xmask = np.ones((T+1,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data, input, xmask, tag))
            z[0] = npr.choice(self.K, p=pi0)
            x[0] = self.dynamics.sample_x(z[0], x[:0], tag=tag, with_noise=with_noise)

        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            assert xhist.shape == (pad, D)
            assert yhist.shape == (pad, N)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            x = np.concatenate((xhist, np.zeros((T,) + D)))
            # input = np.zeros((T+pad,) + M) if input is None else input
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            xmask = np.ones((T+pad,) + D, dtype=bool)

        # Sample z and x
        for t in range(pad, T+pad):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x(z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Sample observations given latent states
        # TODO: sample in the loop above?
        y = self.emissions.sample(z, x, input=input, tag=tag)
        return z[pad:], x[pad:], y[pad:]

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        x_mask = np.ones_like(variational_mean, dtype=bool)
        log_pi0 = self.init_state_distn.log_initial_state_distn(variational_mean, input, x_mask, tag)
        log_Ps = self.transitions.log_transition_matrices(variational_mean, input, x_mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, x_mask, tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        log_pi0 = self.init_state_distn.log_initial_state_distn(variational_mean, input, mask, tag)
        log_Ps = self.transitions.log_transition_matrices(variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return viterbi(log_pi0, log_Ps, log_likes)

    @ensure_slds_args_not_none
    def smooth(self, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(variational_mean, data, input, mask, tag)
        return self.emissions.smooth(Ez, variational_mean, data, input, tag)

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Cannot compute exact marginal log probability for the SLDS. "
                      "the ELBO instead.")
        return np.nan

    @ensure_variational_args_are_lists
    def elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta)
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            xs = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # log p(x, y | theta) = log \sum_z p(x, y, z | theta)
            for x, data, input, mask, tag in zip(xs, datas, inputs, masks, tags):

                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
                log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)
                elbo += hmm_normalizer(log_pi0, log_Ps, log_likes)

            # -log q(x)
            elbo -= variational_posterior.log_density(xs)
            assert np.isfinite(elbo)

        return elbo / n_samples

    @ensure_variational_args_are_lists
    def _surrogate_elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None,
        alpha=0.75, **kwargs):
        """
        Lower bound on the marginal likelihood p(y | gamma)
        using variational posterior q(x; phi) where phi = variational_params
        and gamma = emission parameters.  As part of computing this objective,
        we optimize q(z | x) and take a natural gradient step wrt theta, the
        parameters of the dynamics model.

        Note that the surrogate ELBO is a lower bound on the ELBO above.
           E_p(z | x, y)[log p(z, x, y)]
           = E_p(z | x, y)[log p(z, x, y) - log p(z | x, y) + log p(z | x, y)]
           = E_p(z | x, y)[log p(x, y) + log p(z | x, y)]
           = log p(x, y) + E_p(z | x, y)[log p(z | x, y)]
           = log p(x, y) -H[p(z | x, y)]
          <= log p(x, y)
        with equality only when p(z | x, y) is atomic.  The gap equals the
        entropy of the posterior on z.
        """
        # log p(theta)
        elbo = self.log_prior()

        # Sample x from the variational posterior
        xs = variational_posterior.sample()

        # Inner optimization: find the true posterior p(z | x, y; theta).
        # Then maximize the inner ELBO wrt theta,
        #
        #    E_p(z | x, y; theta_fixed)[log p(z, x, y; theta).
        #
        # This can be seen as a natural gradient step in theta
        # space.  Note: we do not want to compute gradients wrt x or the
        # emissions parameters backward throgh this optimization step,
        # so we unbox them first.
        xs_unboxed = [getval(x) for x in xs]
        emission_params_boxed = self.emissions.params
        flat_emission_params_boxed, unflatten = flatten(emission_params_boxed)
        self.emissions.params = unflatten(getval(flat_emission_params_boxed))

        # E step: compute the true posterior p(z | x, y, theta_fixed) and
        # the necessary expectations under this posterior.
        expectations = [self.expected_states(x, data, input, mask, tag)
                        for x, data, input, mask, tag
                        in zip(xs_unboxed, datas, inputs, masks, tags)]

        # M step: maximize expected log joint wrt parameters
        # Note: Only do a partial update toward the M step for this sample of xs
        x_masks = [np.ones_like(x, dtype=bool) for x in xs_unboxed]
        for distn in [self.init_state_distn, self.transitions, self.dynamics]:
            curr_prms = copy.deepcopy(distn.params)
            distn.m_step(expectations, xs_unboxed, inputs, x_masks, tags, **kwargs)
            distn.params = convex_combination(curr_prms, distn.params, alpha)

        # Box up the emission parameters again before computing the ELBO
        self.emissions.params = emission_params_boxed

        # Compute expected log likelihood E_q(z | x, y) [log p(z, x, y; theta)]
        for (Ez, Ezzp1, _), x, x_mask, data, mask, input, tag in \
            zip(expectations, xs, x_masks, datas, masks, inputs, tags):

            # Compute expected log likelihood (inner ELBO)
            log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
            log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
            log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
            log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

            elbo += np.sum(Ez[0] * log_pi0)
            elbo += np.sum(Ezzp1 * log_Ps)
            elbo += np.sum(Ez * log_likes)

        # -log q(x)
        elbo -= variational_posterior.log_density(xs)
        assert np.isfinite(elbo)

        return elbo

    def _fit_svi(self, variational_posterior, datas, inputs, masks, tags,
                 learning=True, optimizer="adam", num_iters=100, **kwargs):
        """
        Fit with stochastic variational inference using a
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        # Define the objective (negative ELBO)
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            if learning:
                self.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self.elbo(variational_posterior, datas, inputs, masks, tags)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.params, variational_posterior.params)
        else:
            params = variational_posterior.params

        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("ELBO: {:.1f}".format(elbos[0]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # TODO: Check for convergence -- early stopping

            # Update progress bar
            pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
            pbar.update()

        # Save the final parameters
        if learning:
            self.params, variational_posterior.params = params
        else:
            variational_posterior.params = params

        return elbos

    def _fit_variational_em(self, variational_posterior, datas, inputs, masks, tags,
                 learning=True, alpha=.75, optimizer="adam", num_iters=100, **kwargs):
        """
        Let gamma denote the emission parameters and theta denote the transition
        and initial discrete state parameters. This is a mix of EM and SVI:
            1. Sample x ~ q(x; phi)
            2. Compute L(x, theta') = E_p(z | x, theta)[log p(x, z; theta')]
            3. Set theta = (1 - alpha) theta + alpha * argmax L(x, theta')
            4. Set gamma = gamma + eps * nabla log p(y | x; gamma)
            5. Set phi = phi + eps * dx/dphi * d/dx [L(x, theta) + log p(y | x; gamma) - log q(x; phi)]
        """
        # Optimize the standard ELBO when updating gamma (emissions params)
        # and phi (variational params)
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            if learning:
                self.emissions.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self._surrogate_elbo(variational_posterior, datas, inputs, masks, tags, **kwargs)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.emissions.params, variational_posterior.params)
        else:
            params = variational_posterior.params

        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("Surrogate ELBO: {:.1f}".format(elbos[0]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            # Update the emission and variational posterior parameters
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # Update progress bar
            pbar.set_description("Surrogate ELBO: {:.1f}".format(elbos[-1]))
            pbar.update()

        # Save the final emission and variational parameters
        if learning:
            self.emissions.params, variational_posterior.params = params
        else:
            variational_posterior.params = params

        return elbos

    def _fit_variational_em_with_conjugate_updates(\
            self, variational_posterior, datas, inputs, masks, tags,
            learning=True, alpha=.75, optimizer="adam", num_iters=100, **kwargs):
        """
        In the special case where the dynamics and observations are both linear
        Gaussian, we can perform mean field coordinate ascent in a posterior
        approximation of the form,

            p(x, z | y) \approx q(x) q(z)

        where q(x) is a linear Gaussian dynamical system and q(z) is a hidden
        Markov model.
        """
        raise NotImplementedError

    def _fit_laplace_em_discrete_state_update(
        self, variational_posterior, datas,
        inputs, masks, tags,
        num_samples):

        # 0. Draw samples of q(x) for Monte Carlo approximating expectations
        x_sampless = [variational_posterior.sample_continuous_states() for _ in range(num_samples)]
        # Convert this to a list of length len(datas) where each entry
        # is a tuple of length num_samples
        x_sampless = list(zip(*x_sampless))

        # 1. Update the variational posterior on q(z) for fixed q(x)
        #    - Monte Carlo approximate the log transition matrices
        #    - Compute the expected log likelihoods (i.e. log dynamics probs)
        #    - If emissions depend on z, compute expected emission likelihoods
        for prms, x_samples, data, input, mask, tag in \
            zip(variational_posterior.params, x_sampless, datas, inputs, masks, tags):

            # Make a mask for the continuous states
            x_mask = np.ones_like(x_samples[0], dtype=bool)

            # Compute expected log initial distribution, transition matrices, and likelihoods
            prms["log_pi0"] = np.mean(
                [self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
                 for x in x_samples], axis=0)

            prms["log_Ps"] = np.mean(
                [self.transitions.log_transition_matrices(x, input, x_mask, tag)
                 for x in x_samples], axis=0)

            prms["log_likes"] = np.mean(
                [self.dynamics.log_likelihoods(x, input, x_mask, tag)
                 for x in x_samples], axis=0)

            if not self.emissions.single_subspace:
                prms["log_likes"] += np.mean(
                    [self.emissions.log_likelihoods(data, input, mask, tag, x)
                     for x in x_samples], axis=0)

    def _fit_laplace_em_continuous_state_update(
        self, discrete_expectations, variational_posterior,
        datas, inputs, masks, tags,
        continuous_optimizer, continuous_tolerance, continuous_maxiter):

        # 2. Update the variational posterior q(x) for fixed q(z)
        #    - Use Newton's method or LBFGS to find the argmax of the expected log joint
        #       - Compute the gradient g(x) and block tridiagonal Hessian J(x)
        #       - Newton update: x' = x + J(x)^{-1} g(x)
        #       - Check for convergence of x'
        #    - Evaluate the J(x*) at the optimal x*

        # allow for using Newton's method or LBFGS to find x*
        optimizer = dict(newton="newton", lbfgs="lbfgs")[continuous_optimizer]

        # Compute the expected log joint
        def neg_expected_log_joint(x, Ez, Ezzp1, scale=1):
            # The "mask" for x is all ones
            x_mask = np.ones_like(x, dtype=bool)
            log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
            log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
            log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
            log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

            # Compute the expected log probability
            elp = np.sum(Ez[0] * log_pi0)
            elp += np.sum(Ezzp1 * log_Ps)
            elp += np.sum(Ez * log_likes)
            # assert np.all(np.isfinite(elp))

            return -1 * elp / scale

        # We'll need the gradient of the expected log joint wrt x
        grad_neg_expected_log_joint = grad(neg_expected_log_joint)

        # We also need the hessian of the of the expected log joint
        def hessian_neg_expected_log_joint(x, Ez, Ezzp1, scale=1):
            T, D = np.shape(x)
            x_mask = np.ones((T, D), dtype=bool)
            hessian_diag, hessian_lower_diag = self.dynamics.hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
            hessian_diag[:-1] += self.transitions.hessian_expected_log_trans_prob(x, input, x_mask, tag, Ezzp1)
            hessian_diag += self.emissions.hessian_log_emissions_prob(data, input, mask, tag, x, Ez)

            # The Hessian of the log probability should be *negative* definite since we are *maximizing* it.
            hessian_diag -= 1e-8 * np.eye(D)

            # Return the scaled negative hessian, which is positive definite
            return -1 * hessian_diag / scale, -1 * hessian_lower_diag / scale

        # Run Newton's method for each data array to find a
        # Laplace approximation for q(x)
        x0s = variational_posterior.mean_continuous_states
        for prms, (Ez, Ezzp1, _), x0, data, input, mask, tag in \
            zip(variational_posterior.params, discrete_expectations, x0s,
                datas, inputs, masks, tags):

            # Use Newton's method or LBFGS to find the argmax of the expected log joint
            scale = x0.size
            obj = lambda x: neg_expected_log_joint(x, Ez, Ezzp1, scale=scale)
            if optimizer == "newton":
                # Run Newtons method
                grad_func = lambda x: grad_neg_expected_log_joint(x, Ez, Ezzp1, scale=scale)
                hess_func = lambda x: hessian_neg_expected_log_joint(x, Ez, Ezzp1, scale=scale)
                x = newtons_method_block_tridiag_hessian(
                    x0, obj, grad_func, hess_func,
                    tolerance=continuous_tolerance, maxiter=continuous_maxiter)
            elif optimizer == "lbfgs":
                # use LBFGS
                def _objective(params, itr):
                    x = params
                    return neg_expected_log_joint(x, Ez, Ezzp1, scale=scale)
                x = lbfgs(_objective, x0, num_iters=continuous_maxiter,
                          tol=continuous_tolerance)

            # Evaluate the Hessian at the mode
            assert np.all(np.isfinite(obj(x)))
            J_diag, J_lower_diag = hessian_neg_expected_log_joint(x, Ez, Ezzp1)

            # Compute the Hessian vector product h = J * x = -H * x
            # We can do this without instantiating the full matrix
            h = symm_block_tridiag_matmul(J_diag, J_lower_diag, x)

            # update params
            prms["J_diag"] = J_diag
            prms["J_lower_diag"] = J_lower_diag
            prms["h"] = h

    def _fit_laplace_em_params_update(
        self, discrete_expectations, continuous_expectations,
        datas, inputs, masks, tags,
        emission_optimizer, emission_optimizer_maxiter, alpha):

        # 3. Update the model parameters.  Replace the expectation wrt x with sample from q(x).
        # The parameter update is partial and depends on alpha.
        xmasks = [np.ones_like(x, dtype=bool) for x in continuous_expectations]
        for distn in [self.init_state_distn, self.transitions, self.dynamics]:
            curr_prms = copy.deepcopy(distn.params)
            if curr_prms == tuple(): continue
            distn.m_step(discrete_expectations, continuous_expectations, inputs, xmasks, tags)
            distn.params = convex_combination(curr_prms, distn.params, alpha)

        # update emissions params
        curr_prms = copy.deepcopy(self.emissions.params)
        self.emissions.m_step(discrete_expectations, continuous_expectations,
                              datas, inputs, masks, tags,
                              optimizer=emission_optimizer,
                              maxiter=emission_optimizer_maxiter)
        self.emissions.params = convex_combination(curr_prms, self.emissions.params, alpha)

    def _fit_laplace_em_params_update_sgd(
        self, variational_posterior, datas, inputs, masks, tags,
        emission_optimizer="adam", emission_optimizer_maxiter=20):

        # 3. Update the model parameters.
        continuous_expectations = variational_posterior.mean_continuous_states
        xmasks = [np.ones_like(x, dtype=bool) for x in continuous_expectations]

        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = self._laplace_em_elbo(variational_posterior, datas, inputs, masks, tags)
            return -obj / T

        # Optimize parameters
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
        optimizer_state = self.p_optimizer_state if hasattr(self, "p_optimizer_state") else None
        self.params, self.p_optimizer_state = \
            optimizer(_objective, self.params, num_iters=emission_optimizer_maxiter,
                      state=optimizer_state, full_output=True, step_size=0.01)


    def _laplace_em_elbo(self, variational_posterior, datas, inputs, masks, tags, n_samples=1):

        elbo = 0.0
        for sample in range(n_samples):

            # sample continuous states
            continuous_samples = variational_posterior.sample_continuous_states()
            discrete_expectations = variational_posterior.mean_discrete_states

            # log p(theta)
            elbo += self.log_prior()

            for x, (Ez, Ezzp1, _), data, input, mask, tag in \
                zip(continuous_samples, discrete_expectations, datas, inputs, masks, tags):

                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
                log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

                # Compute the expected log probability
                elbo += np.sum(Ez[0] * log_pi0)
                elbo += np.sum(Ezzp1 * log_Ps)
                elbo += np.sum(Ez * log_likes)

            # add entropy of variational posterior
            elbo += variational_posterior.entropy(continuous_samples)

        return elbo / n_samples

    def _fit_laplace_em(self, variational_posterior, datas,
                        inputs=None, masks=None, tags=None,
                        num_iters=100,
                        num_samples=1,
                        continuous_optimizer="newton",
                        continuous_tolerance=1e-4,
                        continuous_maxiter=100,
                        emission_optimizer="lbfgs",
                        emission_optimizer_maxiter=100,
                        parameters_update="mstep",
                        alpha=0.5,
                        learning=True):
        """
        Fit an approximate posterior p(z, x | y) \approx q(z) q(x).
        Perform block coordinate ascent on q(z) followed by q(x).
        Assume q(x) is a Gaussian with a block tridiagonal precision matrix,
        and that we update q(x) via Laplace approximation.
        Assume q(z) is a chain-structured discrete graphical model.
        """
        elbos = [self._laplace_em_elbo(variational_posterior, datas, inputs, masks, tags)]
        pbar = trange(num_iters)
        pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
        for itr in pbar:
            # 1. Update the discrete state posterior q(z) if K>1
            if self.K > 1:
                self._fit_laplace_em_discrete_state_update(
                    variational_posterior, datas, inputs, masks, tags, num_samples)
            discrete_expectations = variational_posterior.mean_discrete_states

            # 2. Update the continuous state posterior q(x)
            self._fit_laplace_em_continuous_state_update(
                discrete_expectations, variational_posterior, datas, inputs, masks, tags,
                continuous_optimizer, continuous_tolerance, continuous_maxiter)
            continuous_expectations = variational_posterior.sample_continuous_states()

            # 3. Update parameters
            # Default is partial M-step given a sample from q(x)
            if learning and parameters_update=="mstep":
                self._fit_laplace_em_params_update(
                    discrete_expectations, continuous_expectations, datas, inputs, masks, tags,
                    emission_optimizer, emission_optimizer_maxiter, alpha)
            # Alternative is SGD on all parameters with samples from q(x)
            elif learning and parameters_update=="sgd":
                self._fit_laplace_em_params_update_sgd(
                    variational_posterior, datas, inputs, masks, tags,
                    emission_optimizer="adam",
                    emission_optimizer_maxiter=emission_optimizer_maxiter)

            # 4. Compute ELBO
            elbo = self._laplace_em_elbo(variational_posterior, datas, inputs, masks, tags)
            elbos.append(elbo)
            pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))

        return elbos

    def _make_variational_posterior(self, variational_posterior, datas, inputs, masks, tags, method, **variational_posterior_kwargs):
        # Initialize the variational posterior
        if isinstance(variational_posterior, str):
            # Make a posterior of the specified type
            _var_posteriors = dict(
                meanfield=varinf.SLDSMeanFieldVariationalPosterior,
                mf=varinf.SLDSMeanFieldVariationalPosterior,
                lds=varinf.SLDSTriDiagVariationalPosterior,
                tridiag=varinf.SLDSTriDiagVariationalPosterior,
                structured_meanfield=varinf.SLDSStructuredMeanFieldVariationalPosterior
                )

            if variational_posterior not in _var_posteriors:
                raise Exception("Invalid posterior: {}. Options are {}.".\
                                format(variational_posterior, _var_posteriors.keys()))
            posterior = _var_posteriors[variational_posterior](self, datas, inputs, masks, tags, **variational_posterior_kwargs)

        else:
            # Check validity of given posterior
            posterior = variational_posterior
            assert isinstance(posterior, varinf.VariationalPosterior), \
            "Given posterior must be an instance of ssm.variational.VariationalPosterior"

        # Check that the posterior type works with the fitting method
        if method in ["svi", "bbvi"]:
            assert isinstance(posterior,
                (varinf.SLDSMeanFieldVariationalPosterior, varinf.SLDSTriDiagVariationalPosterior)),\
            "BBVI only supports 'meanfield' or 'lds' posteriors."

        elif method in ["laplace_em"]:
            assert isinstance(posterior, varinf.SLDSStructuredMeanFieldVariationalPosterior),\
            "Laplace EM only supports 'structured' posterior."

        return posterior

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None,
            method="laplace_em", variational_posterior="structured_meanfield",
            variational_posterior_kwargs=None,
            initialize=True, **kwargs):

        """
        Fitting methods for an arbitrary switching LDS:

        1. Black box variational inference (bbvi/svi): stochastic gradient ascent
           on the evidence lower bound, collapsing out the discrete states and
           maintaining a variational posterior over the continuous states only.

           Pros: simple and broadly applicable.  easy to implement.
           Cons: doesn't leverage model structure.  slow to converge.

        2. Structured mean field: Maintain variational factors q(z) and q(x).
           Update them using block mean field coordinate ascent, if we have a
           Gaussian emission model and linear Gaussian dynamics, or using
           a Laplace approximation for nonconjugate models.

        In the future, we could also consider some other possibilities, like:

        3. Particle EM: run a (Rao-Blackwellized) particle filter targeting
           the posterior distribution of the continuous latent states and
           use its weighted trajectories to get the discrete states and perform
           a Monte Carlo M-step.

        4. Gibbs sampling: As above, if we have a conjugate emission and
           dynamics model we can do block Gibbs sampling of the discrete and
           continuous states.
        """

        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                bbvi=self._fit_svi,
                                laplace_em=self._fit_laplace_em)

        # Deprecate "svi" as a method
        if method == "svi":
            warnings.warn("SLDS fitting method 'svi' will be renamed 'bbvi' in future releases.",
                          category=DeprecationWarning)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        # Initialize the model parameters
        if initialize:
            self.initialize(datas, inputs, masks, tags)

        # Initialize the variational posterior
        variational_posterior_kwargs = variational_posterior_kwargs or {}
        posterior = self._make_variational_posterior(variational_posterior, datas, inputs, masks, tags, method, **variational_posterior_kwargs)
        elbos = _fitting_methods[method](posterior, datas, inputs, masks, tags, learning=True, **kwargs)
        return elbos, posterior

    @ensure_args_are_lists
    def approximate_posterior(self, datas, inputs=None, masks=None, tags=None,
                              method="laplace_em", variational_posterior="structured_meanfield",
                              **kwargs):
        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                bbvi=self._fit_svi,
                                vem=self._fit_variational_em,
                                laplace_em=self._fit_laplace_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        # Initialize the variational posterior
        posterior = self._make_variational_posterior(variational_posterior, datas, inputs, masks, tags, method)
        return _fitting_methods[method](posterior, datas, inputs, masks, tags, learning=False, **kwargs)


class LDS(SLDS):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, D, *, M=0,
            dynamics="gaussian",
            dynamics_kwargs=None,
            hierarchical_dynamics_tags=None,
            emissions="gaussian_orthog",
            emission_kwargs=None,
            hierarchical_emission_tags=None,
            **kwargs):

        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            gaussian=obs.AutoRegressiveObservations,
            diagonal_gaussian=obs.AutoRegressiveDiagonalNoiseObservations,
            t=obs.RobustAutoRegressiveObservations,
            studentst=obs.RobustAutoRegressiveObservations,
            diagonal_t=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_studentst=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](1, D, M=M, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        # Make the emission distn
        emission_classes = dict(
            gaussian=emssn.GaussianEmissions,
            gaussian_orthog=emssn.GaussianOrthogonalEmissions,
            gaussian_id=emssn.GaussianIdentityEmissions,
            gaussian_nn=emssn.GaussianNeuralNetworkEmissions,
            studentst=emssn.StudentsTEmissions,
            studentst_orthog=emssn.StudentsTOrthogonalEmissions,
            studentst_id=emssn.StudentsTIdentityEmissions,
            studentst_nn=emssn.StudentsTNeuralNetworkEmissions,
            t=emssn.StudentsTEmissions,
            t_orthog=emssn.StudentsTOrthogonalEmissions,
            t_id=emssn.StudentsTIdentityEmissions,
            t_nn=emssn.StudentsTNeuralNetworkEmissions,
            poisson=emssn.PoissonEmissions,
            poisson_orthog=emssn.PoissonOrthogonalEmissions,
            poisson_id=emssn.PoissonIdentityEmissions,
            poisson_nn=emssn.PoissonNeuralNetworkEmissions,
            bernoulli=emssn.BernoulliEmissions,
            bernoulli_orthog=emssn.BernoulliOrthogonalEmissions,
            bernoulli_id=emssn.BernoulliIdentityEmissions,
            bernoulli_nn=emssn.BernoulliNeuralNetworkEmissions,
            ar=emssn.AutoRegressiveEmissions,
            ar_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            ar_id=emssn.AutoRegressiveIdentityEmissions,
            ar_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            autoregressive=emssn.AutoRegressiveEmissions,
            autoregressive_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            autoregressive_id=emssn.AutoRegressiveIdentityEmissions,
            autoregressive_nn=emssn.AutoRegressiveNeuralNetworkEmissions
            )

        if isinstance(emissions, str):
            emissions = emissions.lower()
            if emissions not in emission_classes:
                raise Exception("Invalid emission model: {}. Must be one of {}".
                    format(emissions, list(emission_classes.keys())))

            emission_kwargs = emission_kwargs or {}
            emissions = emission_classes[emissions](N, 1, D, M=M,
                single_subspace=True, **emission_kwargs)
        if not isinstance(emissions, emssn.Emissions):
            raise TypeError("'emissions' must be a subclass of"
                            " ssm.emissions.Emissions")

        init_state_distn = isd.InitialStateDistribution(1, D, M)
        transitions = trans.StationaryTransitions(1, D, M)
        super().__init__(N, 1, D, M=M,
                         init_state_distn=init_state_distn,
                         transitions=transitions,
                         dynamics=dynamics,
                         emissions=emissions)

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        return np.ones((variational_mean.shape[0], 1)), \
               np.ones((variational_mean.shape[0], 1, 1)), \
               0

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def log_prior(self):
        return self.dynamics.log_prior() + self.emissions.log_prior()

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Log probability of LDS is not yet implemented.")
        return np.nan

    @ensure_variational_args_are_lists
    def elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta)
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            xs = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # Compute log p(y, x | theta)
            for x, data, input, mask, tag in zip(xs, datas, inputs, masks, tags):
                x_mask = np.ones_like(x, dtype=bool)
                elbo += np.sum(self.dynamics.log_likelihoods(x, input, x_mask, tag))
                elbo += np.sum(self.emissions.log_likelihoods(data, input, mask, tag, x))

            # -log q(x)
            elbo -= variational_posterior.log_density(xs)
            assert np.isfinite(elbo)

        return elbo / n_samples
