import autograd.numpy as np
from ssm.hmm.messages import hmm_expected_states, hmm_filter, hmm_sample, viterbi


class HMMPosterior(object):
    """
    Posterior distribution on the latent states of an HMM.
    """
    def __init__(self, model, data_dict):
        """
        Initialize the posterior distribution on the data given the model.
        """
        self.model = model
        self.data_dict = data_dict
        self._posterior = None

        # Perform inference to compute posterior expectations
        self.update()

    def update(self):
        """
        Run the exact message passing algorithm to infer the posterior distribution.
        """
        # f = jit(grad(hmm_log_normalizer, has_aux=True))
        # (_, Ezzp1, Ez), log_normalizer = f(*self._natural_params)
        Ez, Ezzp1, log_normalizer = hmm_expected_states(*self._natural_params)
        assert np.all(np.isfinite(Ez))
        self._posterior = dict(Ez=Ez,
                               Ezzp1=Ezzp1,
                               log_normalizer=log_normalizer)
        return self

    @property
    def _natural_params(self):
        model, data_dict = self.model, self.data_dict
        log_initial_distn = model.initial_state.log_initial_prob(**data_dict)
        log_transition_matrices = model.transitions.log_transition_matrices(**data_dict)
        log_likelihoods = model.observations.log_likelihoods(**data_dict)

        # TODO: We return the initial distribution and transition matrices
        #       rather than their logs for consistency with the current
        #       message passing interface.  We should probably change that!
        return np.exp(log_initial_distn), \
               np.exp(log_transition_matrices), \
               log_likelihoods

    def marginal_likelihood(self):
        """Compute the marginal likelihood of the data under the model.

        Returns:
            ``\log p(x_{1:T})`` the marginal likelihood of the data
            summing over discrete latent state sequences.
        """
        if self._posterior is None:
            self.update()
        return self._posterior["log_normalizer"]

    def expected_states(self):
        """Compute the expected values of the latent states under the
        posterior distribution.

        Returns:
            ``E[z_t | x_{1:T}]`` the expected value of the latent state
                at time ``t`` given the sequence of data.
        """
        if self._posterior is None:
            self.update()
        return self._posterior["Ez"]

    def expected_transitions(self):
        """Compute the expected transitions of the latent states under the
        posterior distribution.

        Returns:
            ``E[z_t z_{t+1} | x_{1:T}]`` the expected value of
                adjacent latent states given the sequence of data.
        """
        if self._posterior is None:
            self.update()
        return self._posterior["Ezzp1"]

    def sample(self):
        return hmm_sample(*self._natural_params)

    def most_likely_states(self):
        return viterbi(*self._natural_params)

    def reconstruct(self):
        """Reconstruct the data from the latent states
        """
        if self._posterior is None:
            self.update()
        states = self._posterior["Ez"]
        # TODO: The conditional mean might need the data_dict as well
        means = np.array([
            o.mean for o in self.model.observation_distributions])
        return np.dot(states, means)
