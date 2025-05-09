from functools import partial
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from ssm.primitives import hmm_normalizer
from ssm.messages import hmm_expected_states, hmm_filter, hmm_sample, viterbi
from ssm.util import ensure_args_are_lists, ensure_args_not_none_modified, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, \
    replicate, collapse, ssm_pbar, ensure_args_are_lists_modified

import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.hierarchical as hier
import ssm.emissions as emssn

__all__ = ['HMM_TO']


class HMM_TO(object):
    """
    Base class for hidden Markov models with observation and transition inputs.

    Notation:
    K: number of discrete latent states
    D: dimensionality of observations
    M_obs: dimensionality of observation inputs
    M_trans: dimensionality of transition inputs

    In the code we will sometimes refer to the discrete
    latent state sequence as z and the data as x.
    """

    def __init__(self, K, D, M_trans=0, M_obs=0, init_state_distn=None,
                 transitions='standard',
                 transition_kwargs=None,
                 hierarchical_transition_tags=None,
                 observations="gaussian", observation_kwargs=None,
                 hierarchical_observation_tags=None, **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M_trans)
        if not isinstance(init_state_distn, isd.InitialStateDistribution):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.InitialStateDistribution.")

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            constrained=trans.ConstrainedStationaryTransitions,
            sticky=trans.StickyTransitions,
            inputdriven=trans.InputDrivenTransitions,
            inputdrivenalt=trans.InputDrivenTransitionsAlternativeFormulation,
            recurrent=trans.RecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            rbf_recurrent=trans.RBFRecurrentTransitions,
            nn_recurrent=trans.NeuralNetworkRecurrentTransitions
        )

        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                                format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = \
                hier.HierarchicalTransitions(transition_classes[transitions], K, D, M=M_trans,
                                             tags=hierarchical_transition_tags,
                                             **transition_kwargs) \
                    if hierarchical_transition_tags is not None \
                    else transition_classes[transitions](K, D, M=M_trans, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(
            gaussian=obs.GaussianObservations,
            diagonal_gaussian=obs.DiagonalGaussianObservations,
            studentst=obs.MultivariateStudentsTObservations,
            t=obs.MultivariateStudentsTObservations,
            diagonal_t=obs.StudentsTObservations,
            diagonal_studentst=obs.StudentsTObservations,
            exponential=obs.ExponentialObservations,
            bernoulli=obs.BernoulliObservations,
            categorical=obs.CategoricalObservations,
            input_driven_obs=obs.InputDrivenObservations,
            input_driven_obs_diff_inputs=obs.InputDrivenObservationsDiffInputs,
            poisson=obs.PoissonObservations,
            vonmises=obs.VonMisesObservations,
            ar=obs.AutoRegressiveObservations,
            autoregressive=obs.AutoRegressiveObservations,
            no_input_ar=obs.AutoRegressiveObservationsNoInput,
            diagonal_ar=obs.AutoRegressiveDiagonalNoiseObservations,
            diagonal_autoregressive=obs.AutoRegressiveDiagonalNoiseObservations,
            independent_ar=obs.IndependentAutoRegressiveObservations,
            robust_ar=obs.RobustAutoRegressiveObservations,
            no_input_robust_ar=obs.RobustAutoRegressiveObservationsNoInput,
            robust_autoregressive=obs.RobustAutoRegressiveObservations,
            diagonal_robust_ar=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_robust_autoregressive=obs.RobustAutoRegressiveDiagonalNoiseObservations,
        )

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                                format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = \
                hier.HierarchicalObservations(observation_classes[observations], K, D, M_obs=M_obs,
                                              tags=hierarchical_observation_tags,
                                              **observation_kwargs) \
                    if hierarchical_observation_tags is not None \
                    else observation_classes[observations](K, D, M_obs=M_obs, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        self.K, self.D, self.M_trans, self.M_obs = K, D, M_trans, M_obs
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.observations = observations

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.observations.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    @ensure_args_are_lists_modified
    def initialize(self, datas, transition_input=None, observation_input=None, masks=None, tags=None,
                   init_method="random"):
        """
        Initialize parameters given data.
        """
        self.init_state_distn.initialize(datas, inputs=observation_input, masks=masks, tags=tags)
        self.transitions.initialize(datas, inputs=transition_input, masks=masks, tags=tags)
        self.observations.initialize(datas, inputs=observation_input, masks=masks, tags=tags, init_method=init_method)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def sample(self, T, prefix=None, transition_input=None, observation_input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        T : int
            number of time steps to sample

        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.

        transition_input : (T, transition_input_dim) array_like
            Optional transition inputs to specify for sampling

        observation_input : (T, observation_input_dim) array_like
            Optional observation inputs to specify for sampling

        tag : object
            Optional tag indicating which "type" of sampled data

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M_trans = (self.M_trans,) if isinstance(self.M_trans, int) else self.M_trans
        M_obs = (self.M_obs,) if isinstance(self.M_obs, int) else self.M_obs

        assert isinstance(D, tuple)
        assert isinstance(M_trans, tuple)
        assert isinstance(M_obs, tuple)
        assert T > 0

        # Check the transition_input
        if transition_input is not None:
            assert transition_input.shape == (T,) + M_trans

        # Check the observation_input
        if observation_input is not None:
            assert observation_input.shape == (T,) + M_obs

        # Get the type of the observations
        if isinstance(self.observations, obs.InputDrivenObservationsDiffInputs):
            dtype = int
        else:
            dummy_data = self.observations.sample_x(0, np.empty(0, ) + D)
            dtype = dummy_data.dtype

        # Fit the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            transition_input = np.zeros((T,) + M_trans) if transition_input is None else transition_input
            observation_input = np.zeros((T,) + M_obs) if observation_input is None else observation_input

            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observations.sample_x(z[0], data[:0], observation_input=observation_input[0],
                                                 with_noise=with_noise)

            # We only need to sample T-1 data points now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, transition_input, observation_input and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            transition_input = np.zeros((T + pad,) + M_trans) if transition_input is None else np.concatenate(
                (np.zeros((pad,) + M_trans), transition_input))
            observation_input = np.zeros((T + pad,) + M_obs) if observation_input is None else np.concatenate(
                (np.zeros((pad,) + M_obs), observation_input))
            mask = np.ones((T + pad,) + D, dtype=bool)

        # Fill in the rest of the data
        for t in range(pad, pad + T):
            Pt = self.transitions.transition_matrices(data[t - 1:t + 1], transition_input[t - 1:t + 1],
                                                      mask=mask[t - 1:t + 1], tag=tag)[0]
            z[t] = npr.choice(self.K, p=Pt[z[t - 1]])
            data[t] = self.observations.sample_x(z[t], data[:t], observation_input=observation_input[t], tag=tag,
                                                 with_noise=with_noise)

        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    @ensure_args_not_none_modified
    def expected_states(self, data, transition_input=None, observation_input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, transition_input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, observation_input, mask, tag)
        return hmm_expected_states(pi0, Ps, log_likes)

    def Ps_matrix(self, data, transition_input=None, observation_input=None, mask=None, tag=None):
        Ps = self.transitions.transition_matrices(data, transition_input, mask, tag)
        return Ps

    @ensure_args_not_none_modified
    def most_likely_states(self, data, transition_input=None, observation_input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, transition_input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, observation_input, mask, tag)
        return viterbi(pi0, Ps, log_likes)

    @ensure_args_not_none_modified
    def filter(self, data, transition_input=None, observation_input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, transition_input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, observation_input, mask, tag)
        return hmm_filter(pi0, Ps, log_likes)

    @ensure_args_not_none_modified
    def smooth(self, data, transition_input=None, observation_input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(data, transition_input, observation_input, mask)
        return self.observations.smooth(Ez, data, transition_input, observation_input, tag)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @ensure_args_are_lists_modified
    def log_likelihood(self, datas, transition_input=None, observation_input=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, transition_input, observation_input, mask, tag in zip(datas, transition_input, observation_input,
                                                                        masks, tags):
            pi0 = self.init_state_distn.initial_state_distn
            Ps = self.transitions.transition_matrices(data, transition_input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, observation_input, mask, tag)
            ll += hmm_normalizer(pi0, Ps, log_likes)
            assert np.isfinite(ll)
        return ll

    @ensure_args_are_lists_modified
    def log_probability(self, datas, transition_input=None, observation_input=None, masks=None, tags=None):
        return self.log_likelihood(datas, transition_input, observation_input, masks, tags) + self.log_prior()

    def expected_log_likelihood(self, expectations, datas, transition_inputs=None, observation_inputs=None, masks=None,
                                tags=None):
        """
        Compute log-likelihood given current model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ell = 0.0
        for (Ez, Ezzp1, _), data, transition_input, observation_input, mask, tag in \
                zip(expectations, datas, transition_inputs, observation_inputs, masks, tags):
            pi0 = self.init_state_distn.initial_state_distn
            log_Ps = self.transitions.log_transition_matrices(data, transition_input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, observation_input, mask, tag)

            ell += np.sum(Ez[0] * np.log(pi0))
            ell += np.sum(Ezzp1 * log_Ps)
            ell += np.sum(Ez * log_likes)
            assert np.isfinite(ell)

        return ell

    def expected_log_probability(self, expectations, datas, transition_inputs=None, observation_inputs=None, masks=None,
                                 tags=None):
        """
        Compute the log-probability of the data given current
        model parameters.
        """
        ell = self.expected_log_likelihood(expectations, datas, transition_inputs=transition_inputs,
                                           observation_inputs=observation_inputs, masks=masks, tags=tags)
        return ell + self.log_prior()

    def trans_weights_K(self, hmm_params, K):
        """
        Standardize the GLM transition weights.
        """
        trans_weight_append_zero = np.vstack((hmm_params[1][1], np.zeros((1, hmm_params[1][1].shape[1]))))
        permutation = range(K)
        trans_weight_append_zero_standard = trans_weight_append_zero
        v1 = - np.mean(trans_weight_append_zero, axis=0)
        trans_weight_append_zero_standard[-1, :] = v1
        for i in range(K - 1):
            trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
        weight_vectors_trans = trans_weight_append_zero_standard[permutation]
        return weight_vectors_trans

    # Model fitting
    def _fit_sgd(self, optimizer, datas, transition_input, observation_input, masks, tags, verbose=2, num_iters=1000,
                 **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data.shape[0] for data in datas])

        def _objective(params, itr):
            self.params = params
            obj = self.log_probability(datas, transition_input, observation_input, masks, tags)
            return -obj / T

        # Set up the progress bar
        lls = [-_objective(self.params, 0) * T]
        pbar = ssm_pbar(num_iters, verbose, "Epoch {} Itr {} LP: {:.1f}", [0, 0, lls[-1]])

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            lls.append(-val * T)
            if verbose == 2:
                pbar.set_description("LP: {:.1f}".format(lls[-1]))
                pbar.update(1)
        return lls

    def _fit_stochastic_em(self, optimizer, datas, transition_input, observation_input, masks, tags, verbose=2,
                           num_epochs=100, **kwargs):
        """
        Replace the M-step of EM with a stochastic gradient update using the ELBO computed
        on a minibatch of data.
        """
        M = len(datas)
        T = sum([data.shape[0] for data in datas])

        # A helper to grab a minibatch of data
        perm = [np.random.permutation(M) for _ in range(num_epochs)]

        def _get_minibatch(itr):
            epoch = itr // M
            m = itr % M
            i = perm[epoch][m]
            return datas[i], transition_input[i], observation_input[i], masks[i], tags[i][i]

        # Define the objective (negative ELBO)
        def _objective(params, itr):
            # Grab a minibatch of data
            data, transition_input, observation_input, mask, tag = _get_minibatch(itr)
            Ti = data.shape[0]

            # E step: compute expected latent states with current parameters
            Ez, Ezzp1, _ = self.expected_states(data, transition_input, observation_input, mask, tag)

            # M step: set the parameter and compute the (normalized) objective function
            self.params = params
            pi0 = self.init_state_distn.initial_state_distn
            log_Ps = self.transitions.log_transition_matrices(data, transition_input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, observation_input, mask, tag)

            # Compute the expected log probability
            # (Scale by number of length of this minibatch.)
            obj = self.log_prior()
            obj += np.sum(Ez[0] * np.log(pi0)) * M
            obj += np.sum(Ezzp1 * log_Ps) * (T - M) / (Ti - 1)
            obj += np.sum(Ez * log_likes) * T / Ti
            assert np.isfinite(obj)

            return -obj / T

        # Set up the progress bar
        lls = [-_objective(self.params, 0) * T]
        pbar = ssm_pbar(num_epochs * M, verbose, "Epoch {} Itr {} LP: {:.1f}", [0, 0, lls[-1]])

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, _, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            epoch = itr // M
            m = itr % M
            lls.append(-val * T)
            if verbose == 2:
                pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(epoch, m, lls[-1]))
                pbar.update(1)
        return lls

    def _fit_em(self, datas, transition_input, observation_input, masks, tags, verbose=2, num_iters=100, tolerance=0,
                init_state_mstep_kwargs={},
                transitions_mstep_kwargs={},
                observations_mstep_kwargs={},
                **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = [self.log_probability(datas, transition_input, observation_input, masks, tags)]
        pbar = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [lls[-1]])

        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, transition_input, observation_input, mask, tag)
                            for data, transition_input, observation_input, mask, tag,
                            in zip(datas, transition_input, observation_input, masks, tags)]

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step_modified(expectations, datas, transition_input, observation_input, masks, tags,
                                                  **init_state_mstep_kwargs)
            self.transitions.m_step(expectations, datas, transition_input, masks, tags, **transitions_mstep_kwargs)
            self.observations.m_step(expectations, datas, observation_input, masks, tags, **observations_mstep_kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))

            if verbose == 2:
                pbar.set_description("LP: {:.1f}".format(lls[-1]))

            # Check for convergence
            if itr > 0 and abs(lls[-1] - lls[-2]) < tolerance:
                if verbose == 2:
                    pbar.set_description("Converged to LP: {:.1f}".format(lls[-1]))
                break

        return lls

    @ensure_args_are_lists_modified
    def fit(self, datas, transition_input=None, observation_input=None, masks=None, tags=None,
            verbose=2, method="em",
            initialize=True,
            init_method="random",
            **kwargs):

        _fitting_methods = \
            dict(sgd=partial(self._fit_sgd, "sgd"),
                 adam=partial(self._fit_sgd, "adam"),
                 em=self._fit_em,
                 stochastic_em=partial(self._fit_stochastic_em, "adam"),
                 stochastic_em_sgd=partial(self._fit_stochastic_em, "sgd"),
                 )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(datas,
                            transition_input=None, observation_input=None,
                            masks=masks,
                            tags=tags,
                            init_method=init_method)

        if isinstance(self.transitions,
                      trans.ConstrainedStationaryTransitions):
            if method != "em":
                raise Exception("Only EM is implemented for constrained transitions.")

        return _fitting_methods[method](datas,
                                        transition_input=transition_input,
                                        observation_input=observation_input,
                                        masks=masks,
                                        tags=tags,
                                        verbose=verbose,
                                        **kwargs)
