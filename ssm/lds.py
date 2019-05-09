import copy
import warnings
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

from .optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from .primitives import hmm_normalizer, hmm_expected_states, hmm_filter, hmm_sample, viterbi
from .util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, \
    replicate, collapse

from . import observations as obs
from . import transitions as trans
from . import init_state_distns as isd
from . import hierarchical as hier
from . import emissions as emssn
from . import hmm

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
            input = np.zeros((T+1,) + M) if input is None else input
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
            input = np.zeros((T+pad,) + M) if input is None else input
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

    def _fit_structured_meanfield(self, variational_posterior, datas,
                                  inputs=None, masks=None, tags=None,
                                  num_samples=1):
        """
        Fit an approximate posterior p(z, x | y) \approx q(z) q(x).
        Perform block coordinate ascent on q(z) followed by q(x).
        Assume q(x) is a Gaussian with a block tridiagonal precision matrix,
        and that we update q(x) via Laplace approximation.
        Assume q(z) is a chain-structured discrete graphical model.
        """
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

            x_mask = np.ones_like(x_sample, dtype=bool)

            # Compute expected log initial distribution, transition matrices, and likelihoods
            prms["log_pi0"] = np.mean(
                [self.transitions.log_initial_state_distn(x, input, mask, tag)
                 for x in x_samples], axis=0)

            prms["log_Ps"] = np.mean(
                [self.transitions.log_transition_matrices(x, input, mask, tag)
                 for x in x_samples], axis=0)

            prms["log_likes"] = np.mean(
                [self.dynamics.log_likelihoods(x, input, mask, tag)
                 for x in x_samples], axis=0)

            if not self.single_subspace:
                prms["log_likes"] += np.mean(
                    [self.emissions.log_likelihoods(data, input, mask, tag, x)
                     for x in x_samples], axis=0)

        # 2. Update the variational posterior q(x) for fixed q(z)
        #    - Use Newton's method to find the argmax of the expected log joint
        #       - Compute the gradient g(x) and block tridiagonal Hessian J(x)
        #       - Newton update: x' = x + J(x)^{-1} g(x)
        #       - Check for convergence of x'
        #    - Evaluate the J(x*) at the optimal x*
        discrete_expectations = variational_posterior.mean_discrete_states
        continuous_expectations = variational_posterior.mean_continuous_states # use these to initialize newton steps

        for prms, (Ez, Ezzp1, _), Ex, data, input, mask, tag in \
            zip(variational_posterior.params, discrete_expectations, continuous_expectations,
                datas, inputs, masks, tags):

            # Run Newton's method
            # Compute the expected log joint
            def expected_log_joint(x):
                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                log_pi0 = self.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
                log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

                # Compute the expected log probability
                elp += np.sum(Ez[0] * log_pi0)
                elp += np.sum(Ezzp1 * log_Ps)
                elp += np.sum(Ez * log_likes)

                return elp

            # Compute the gradient of the expected log joint at a point x
            grad_expected_log_joint = grad(expected_log_joint)

            # Compute the expected Hessian, represented in blocks
            def hessian_blocks(x):

                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)
                hessian_diag, hessian_lower_diag = self.dynamics.hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
                hessian_diag += self.transitions.hessian_expected_log_trans_prob(x, input, x_mask, tag, Ezzp1)
                #TODO include Ez
                hessian_diag += self.emissions.hessian_log_emissions_prob(data, input, mask, tag, x)

                return hessian_diag, hessian_lower_diag

            def newtons_method(x0, grad, hessian_blocks):
                # TODO: damping, etc
                x = x0
                while not is_converged:
                    hessian_diag, hessian_lower_diag = hessian_blocks(x)
                    J_banded = blocks_to_bands(hessian_diag, hessian_lower_diag, lower=True)
                    dx = np.reshape(solveh_banded(J_banded, np.ravel(grad(x)), lower=True), x.shape)
                    x = x + dx
                    is_converged = np.mean(np.abs(dx)) < 1e-8

                return x

            xstar = newtons_method(Ez, grad, hessian_blocks)
            Jstar_diag, Jstar_lower_diag = hessian_blocks(xstar)

            # Solve linear system in the Hessian to get h = J * xstar
            hstar = ...

        # 3. Update the model parameters
        xstar_mask = np.ones_like(xstar, dtype=bool)
        self.transitions.m_step(xstars, inputs, xstar_masks, tags)
        self.dynamics.m_step(xstars, inputs, xstar_masks, tags)
        self.emissions.m_step(xstars, inputs, xstar_masks, tags)

    @ensure_variational_args_are_lists
    def fit(self, variational_posterior, datas,
            inputs=None, masks=None, tags=None, method="svi",
            initialize=True, **kwargs):

        """
        Fitting methods for an arbitrary switching LDS:

        1. Black box variational inference (bbvi/svi): stochastic gradient ascent
           on the evidence lower bound, collapsing out the discrete states and
           maintaining a variational posterior over the continuous states only.

           Pros: simple and broadly applicable.  easy to implement.
           Cons: doesn't leverage model structure.  slow to converge.

        2. Variational expectation maximization (vem): variational posterior
           on the continuous states q(x) and a discrete Markov chain
           posterior on the discrete states q(z). We use samples of q(x)
           to approximate the log transition matrix (pairwise potentials)
           and the log transition bias (unary potentials) for q(z).  From
           these we can derive the necessary expectations wrt q(z) for
           updating the model parameters theta.

        In the future, we could also consider some other possibilities, like:

        3. Particle EM: run a (Rao-Blackwellized) particle filter targeting
           the posterior distribution of the continuous latent states and
           use its weighted trajectories to get the discrete states and perform
           a Monte Carlo M-step.

        4. Structured mean field: Maintain variational factors q(z) and q(x).
           Update them using block mean field coordinate ascent, if we have a
           Gaussian emission model and linear Gaussian dynamics, or using an
           approximate update (e.g. a Laplace approximation) if we have a
           nonconjugate model.

        5. Gibbs sampling: As above, if we have a conjugate emission and dynamics
           model we can do block Gibbs sampling of the discrete and continuous
           states.
        """

        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                bbvi=self._fit_svi,
                                vem=self._fit_variational_em)

        # Deprecate "svi" as a method
        warnings.warn("SLDS fitting method 'svi' will be renamed 'bbvi' in future releases.",
                      category=DeprecationWarning)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs, masks, tags)

        return _fitting_methods[method](variational_posterior, datas, inputs, masks, tags,
            learning=True, **kwargs)

    @ensure_variational_args_are_lists
    def approximate_posterior(self, variational_posterior, datas, inputs=None, masks=None, tags=None,
                              method="svi", **kwargs):
        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                vem=self._fit_variational_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        return _fitting_methods[method](variational_posterior, datas, inputs, masks, tags,
            learning=False, **kwargs)


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
