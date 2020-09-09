from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.optimizers import convex_combination
from ssm.util import ensure_args_are_lists


class InitialStateDistribution(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M
        self.log_pi0 = -np.log(K) * np.ones(K)

    @property
    def params(self):
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        self.log_pi0 = value[0]

    @property
    def initial_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    @property
    def log_initial_state_distn(self):
        return self.log_pi0 - logsumexp(self.log_pi0)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_pi0 = self.log_pi0[perm]

    def log_prior(self):
        return 0

    def compute_sample_size(self, datas, inputs, masks, tags):
        return len(datas)

    def expected_sufficient_stats(self, expectations, datas, inputs, masks, tags):
        """
        Sufficient statistics are

            Ez0 = E[z_0 = k]

        """
        # Return early if no data is given
        if len(expectations) == 0:
            return np.zeros(self.K)
        Ez0 = sum([Ez[0] for Ez, _, _ in expectations]) + 1e-8
        return Ez0

    def m_step(self, expectations, datas, inputs, masks, tags, sufficient_stats=None, **kwargs):
        """Compute M-step for initial state distribution."""
        # Collect sufficient statistics
        if sufficient_stats is None:
            Ez0 = self.expected_sufficient_stats(expectations, datas, inputs, masks, tags)
        else:
            Ez0 = sufficient_stats
        self.log_pi0 = np.log(Ez0 / Ez0.sum())

    def stochastic_m_step(self, 
                          optimizer_state, 
                          total_sample_size,
                          expectations, 
                          datas,
                          inputs,
                          masks, 
                          tags,
                          step_size=0.5):

        # Get the expected sufficient statistics for this minibatch
        stats = self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)

        # Scale the stats by the fraction of the sample size
        this_sample_size = self.compute_sample_size(datas, inputs, masks, tags)
        stats *= total_sample_size /  this_sample_size

        # Combine them with the running average sufficient stats
        if optimizer_state is not None:
            stats = convex_combination(optimizer_state, stats, step_size)

        # Call the regular m-step with these sufficient statistics
        self.m_step(None, None, None, None, None, sufficient_stats=stats)

        # Return the update state (i.e. the new stats)
        return stats


class FixedInitialStateDistribution(InitialStateDistribution):
    def __init__(self, K, D, pi0=None, M=0):
        super(FixedInitialStateDistribution, self).__init__(K, D, M=M)
        if pi0 is not None:
            # Handle the case where user passes a numpy array of (K, 1) instead of (K,)
            pi0 = np.squeeze(np.array(pi0))
            assert len(pi0) == K, "Array passed as pi0 is of the wrong length"
            self.log_pi0 = np.log(pi0 + 1e-16)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        # Don't change the distribution
        pass