import copy
import warnings
from functools import partial

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.primitives import hmm_normalizer, hmm_expected_states
from ssm.util import ensure_args_are_lists, ensure_args_not_none, ensure_slds_args_not_none, ensure_elbo_args_are_lists

class _StackedSwitchingLDS(object):
    """
    Stack of SLDS.  Each continuous layer has conditionally linear Gaussian dynamics.
    Continuous states bias the transitions of the discrete state layer below. 

    for l = 1, ..., L

        z_t^l ~ pi(x_t^{l-1}, z_{t-1}^l)

        x_t^l ~ N(A_{z_t^l} x_{t-1}^l + b_{z_t^l})

    y_t ~ N(C x_t^L + d)

    """
    def __init__(self, N, K, D, M, list_of_init_state_distns, list_of_transitions, list_of_dynamics, emissions):
        self.N, self.K, self.D, self.M = N, K, D, M
        self.list_of_init_state_distns = list_of_init_state_distns
        self.list_of_transitions = list_of_transitions
        self.list_of_dynamics = list_of_dynamics
        self.emissions = emissions

        # Figure out the number of layers
        self.L = len(list_of_init_state_distns)
        assert len(list_of_transitions) == len(list_of_dynamics) == self.L
        
        # Only allow fitting by SVI
        self._fitting_methods = dict(svi=self._fit_svi)

    @property
    def params(self):
        init_state_distn_params = []
        for init_state_distn in self.list_of_init_state_distns:
            init_state_distn_params.append(init_state_distn.params)

        transition_params = []
        for transition in self.list_of_transitions:
            transition_params.append(transition.params)

        dynamics_params = []
        for dynamics in self.list_of_dynamics:
            dynamics_params.append(dynamics.params)

        return init_state_distn_params, transition_params, dynamics_params, self.emissions.params
    
    @params.setter
    def params(self, value):
        for d, prms in zip(self.list_of_init_state_distns, value[0]):
            d.params = prms
        for d, prms in zip(self.list_of_transitions, value[1]):
            d.params = prms
        for d, prms in zip(self.list_of_dynamics, value[2]):
            d.params = prms
        self.emissions.params = value[3]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # # First initialize the observation model
        # self.emissions.initialize(datas, inputs, masks, tags)

        # # Get the initialized variational mean for the data
        # xs = [self.emissions.initialize_variational_params(data, input, mask, tag) [0]
        #       for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        # xmasks = [np.ones_like(x, dtype=bool) for x in xs]
        
        # # Now run a few iterations of EM on a ARHMM with the variational mean
        # print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
        # arhmm = _HMM(self.K, self.D, self.M, 
        #              copy.deepcopy(self.init_state_distn),
        #              copy.deepcopy(self.transitions),
        #              copy.deepcopy(self.dynamics))

        # arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags, 
        #           method="em", num_em_iters=num_em_iters, num_iters=10, verbose=False)

        # self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        # self.transitions = copy.deepcopy(arhmm.transitions)
        # self.dynamics = copy.deepcopy(arhmm.observations)
        # print("Done")
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        raise NotImplementedError

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return 0

    def sample(self, T, input=None, tag=None):
        L, K, D = self.L, self.K, self.D
        input = np.zeros((T, self.M)) if input is None else input
        mask = np.ones((T, D), dtype=bool)
        
        # Initialize outputs
        z = np.zeros((L, T), dtype=int)
        x = np.zeros((L, T, D))
        
        # Sample discrete and continuous latent states
        for l, (init_state_distn, transitions, dynamics) in \
            enumerate(zip(self.list_of_init_state_distns, 
                          self.list_of_transitions, 
                          self.list_of_dynamics)):

            pi0 = np.exp(init_state_distn.log_initial_state_distn(x, input, mask, tag))
            z[l, 0] = npr.choice(self.K, p=pi0)
            x[l, 0] = dynamics.sample_x(z[l, 0], x[l, :0], input=input[0], tag=tag)

            for t in range(1, T):
                # Inputs go into the first layer; previous layer's continuous go into subsequent layers
                ut = input[t-1:t+1] if l == 0 else x[l-1, t-1:t+1] 
                Pt = np.exp(transitions.log_transition_matrices(x[l, t-1:t+1], input=ut, mask=None, tag=tag))[0]

                z[l, t] = npr.choice(self.K, p=Pt[z[l, t-1]])
                x[l, t] = dynamics.sample_x(z[l, t], x[l, :t], input=None, tag=tag)

        # Sample observations given latent states
        y = self.emissions.sample_y(z[-1], x[-1], input=None, tag=tag)
        return z, x, y

    # @ensure_slds_args_not_none
    # def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
    #     log_pi0 = self.init_state_distn.log_initial_state_distn(variational_mean, input, mask, tag)
    #     log_Ps = self.transitions.log_transition_matrices(variational_mean, input, mask, tag)
    #     log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
    #     log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
    #     return hmm_expected_states(log_pi0, log_Ps, log_likes)

    # @ensure_slds_args_not_none
    # def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
    #     Ez, _ = self.expected_states(variational_mean, data, input, mask, tag)
    #     return np.argmax(Ez, axis=1)

    # @ensure_slds_args_not_none
    # def smooth(self, variational_mean, data, input=None, mask=None, tag=None):
    #     """
    #     Compute the mean observation under the posterior distribution
    #     of latent discrete states.
    #     """
    #     Ez, _ = self.expected_states(variational_mean, data, input, mask)
    #     return self.emissions.smooth(Ez, variational_mean, data, input, tag)

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Cannot compute exact marginal log probability for the SLDS. "
                      "the ELBO instead.")
        return np.nan

    @ensure_elbo_args_are_lists
    def elbo(self, variational_params, datas, inputs=None, masks=None, tags=None, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta) 
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for data, input, mask, tag, (q_mus, q_sigma_invs) in \
            zip(datas, inputs, masks, tags, variational_params):

            for sample in range(n_samples):
                # log p(theta)
                elbo += self.log_prior()

                # Sample x from the variational posterior for each layer
                xs = [q_mu + np.sqrt(np.exp(q_sigma_inv)) * npr.randn(data.shape[0], self.D) \
                      for q_mu, q_sigma_inv in zip(q_mus, q_sigma_invs)]
                x_masks = [np.ones_like(x, dtype=bool) for x in xs]
                    
                for l, (x, x_mask, q_mu, q_sigma_inv) in enumerate(zip(xs, x_masks, q_mus, q_sigma_invs)):
                    q_sigma = np.exp(q_sigma_inv)
                    u = input if l == 0 else xs[l-1]

                    # Compute log p(x | theta) = log \sum_z p(x, z | theta)
                    log_pi0 = self.list_of_init_state_distns[l].log_initial_state_distn(x, u, x_mask, tag)
                    log_Ps = self.list_of_transitions[l].log_transition_matrices(x, u, x_mask, tag)
                    log_likes = self.list_of_dynamics[l].log_likelihoods(x, input, x_mask, tag)

                    if l == self.L - 1:
                        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

                    elbo += hmm_normalizer(log_pi0, log_Ps, log_likes)

                    # -log q(x)
                    elbo -= np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
                    elbo -= np.sum(-0.5 * (x - q_mu)**2 / q_sigma)
                    
                assert np.isfinite(elbo)
        
        return elbo / n_samples

    def _fit_svi(self, datas, inputs, masks, tags, learning=True, optimizer="adam", print_intvl=1, **kwargs):
        """
        Fit with stochastic variational inference using a 
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        T = sum([data.shape[0] for data in datas])

        # Initialize the variational posterior parameters
        variational_params = [([npr.randn(data.shape[0], self.D) for l in range(self.L)],
                               [0.1 * npr.randn(data.shape[0], self.D) for l in range(self.L)])
                              for data in datas]

        def _objective(params, itr):
            if learning:
                self.params, variational_params = params
            else:
                variational_params = params

            obj = self.elbo(variational_params, datas, inputs, masks, tags)
            return -obj / T

        elbos = []
        def _print_progress(params, itr, g):
            elbos.append(-_objective(params, itr) * T)
            if itr % print_intvl == 0:
                print("Iteration {}.  ELBO: {}".format(itr, elbos[-1]))
        
        optimizers = dict(sgd=sgd, adam=adam)
        initial_params = (self.params, variational_params) if learning else variational_params
        results = \
            optimizers[optimizer](grad(_objective), 
                initial_params,
                callback=_print_progress,
                **kwargs)

        if learning:
            self.params, variational_params = results
        else:
            variational_params = results

        # unpack outputs as necessary
        variational_params = variational_params[0] if \
            len(variational_params) == 1 else variational_params
        return elbos, variational_params

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None, method="svi", initialize=True, **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs, masks, tags)

        return self._fitting_methods[method](datas, inputs, masks, tags, learning=True, **kwargs)

    @ensure_args_are_lists
    def approximate_posterior(self, datas, inputs=None, masks=None, tags=None, method="svi", **kwargs):
        if method not in self._fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, self._fitting_methods.keys()))

        return self._fitting_methods[method](datas, inputs, masks, tags, learning=False, **kwargs)



def StackedSLDS(L, N, K, D, M=0):
    """
    Construct an SLDS object with the appropriate observations, latent states, and dynamics. 

    :param N: observation dimension
    :param K: number of discrete latent states
    :param D: latent dimension
    :param M: input dimension
    :param observations: conditional distribution of the data 
    :param robust_dynamics: if true, continuous latent states have Student's t noise.
    :param recurrent: whether or not past observations influence transitions probabilities.
    :param recurrent_only: if true, _only_ the past observations influence transitions. 
    :param single_subspace: if true, all discrete states share the same mapping from 
        continuous latent states to observations.
    """
    # Make the initial state distribution
    
    # Make the transition model
    from ssm.init_state_distns import InitialStateDistribution
    from ssm.transitions import InputDrivenTransitions
    from ssm.observations import AutoRegressiveObservations
    from ssm.emissions import GaussianEmissions
    list_of_init_state_distns = [InitialStateDistribution(K, D, M=M if l == 0 else D) for l in range(L)]
    list_of_transition_distns = [InputDrivenTransitions(K, D, M=M if l == 0 else D) for l in range(L)]
    list_of_dynamics_distns = [AutoRegressiveObservations(K, D, M=M if l == 0 else 0) for l in range(L)]
    emission_distn = GaussianEmissions(N, K, D, M=0, single_subspace=True)

    # Make the Stacked SLDS
    return _StackedSwitchingLDS(N, K, D, M, list_of_init_state_distns, list_of_transition_distns, list_of_dynamics_distns, emission_distn)

