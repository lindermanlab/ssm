import autograd.numpy as np
from scipy.stats import norm, invwishart

from ssm.observations import Observations, AutoRegressiveObservations
from ssm.optimizers import convex_combination


class HierarchicalAutoRegressiveObservations(Observations):
    """
        Hierarchical ARobservation model with Gaussian noise.

            (x_t | z_t = k, u_t, tag=i)
                ~ N(A_{i,k} x_{t-1} + b_{i,k} + V_{i,k} u_t, S_{i,k})

        where S_k is a positive definite covariance matrix and tag
        specifies which "group" this observation comes from.

        The parameters are fit via maximum likelihood estimation with
        the hierarchical prior,

        A_{i,k} ~ N(A_k, sigma^2 I)

        where A_k is the group-average dynamics matrix for state k, and
        sigma^2 specifies how much the per-group parameters vary around
        the group-average.

        The same model applies to the b's and V's and S's, but the S's
        are a bit different because they are covariance matrices.
        Instead, update S_{ik} under a inverse Wishart prior,

         S_{i,k} ~ IW(S_k, nu)

        where nu specifies the degrees of freedom.
        """
    def __init__(self, K, D, M=0, lags=1,
                 cond_variance_A=0.001,
                 cond_variance_V=0.001,
                 cond_variance_b=0.001,
                 cond_dof_Sigma=10,
                 tags=(None,)):

        super().__init__(K, D, M)
        self.lags = lags

        # First figure out how many tags/groups
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0

        # Set the hierarchical prior hyperparameters
        self.cond_variance_A = cond_variance_A
        self.cond_variance_V = cond_variance_V
        self.cond_variance_b = cond_variance_b
        self.cond_dof_Sigma = cond_dof_Sigma

        # Create a group-level AR model
        self.global_ar_model = \
            AutoRegressiveObservations(K, D, M=M, lags=lags,
                                       initialize="random_rotation")

        # Create AR objects for each tag
        self.per_group_ar_models = [
            AutoRegressiveObservations(K, D, M=M, lags=lags,
                                       mean_A=self.global_ar_model.As,
                                       variance_A=cond_variance_A,
                                       mean_V=self.global_ar_model.Vs,
                                       variance_V=cond_variance_V,
                                       mean_b=self.global_ar_model.bs,
                                       variance_b=cond_variance_b,
                                       mean_Sigma=self.global_ar_model.Sigmas,
                                       extra_dof_Sigma=cond_dof_Sigma,
                                       initialize="prior"
                                       )
            for _ in range(self.G)
        ]

    def get_As(self, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]].As

    def get_Vs(self, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]].Vs

    def get_bs(self, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]].bs

    def get_Sigmas_init(self, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]].Sigmas_init

    def get_Sigmas(self, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]].Sigmas

    @property
    def params(self):
        raise Exception("Don't try to get these parameters")

    @params.setter
    def params(self, value):
        raise Exception("Don't try to set these parameters")

    def permute(self, perm):
        self.global_ar_model.permute(perm)
        for ar in self.per_group_ar_models:
            ar.permute(perm)

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with linear regressions
        self.global_ar_model.initialize(datas, inputs, masks, tags)
        self._update_hierarchical_prior()

        # Copy global parameters to per-group models
        for ar in self.per_group_ar_models:
            ar.As = self.global_ar_model.As.copy()
            ar.Vs = self.global_ar_model.Vs.copy()
            ar.bs = self.global_ar_model.bs.copy()
            ar.Sigmas = self.global_ar_model.Sigmas.copy()

            ar.As = norm.rvs(self.global_ar_model.As, np.sqrt(self.cond_variance_A))
            ar.Vs = norm.rvs(self.global_ar_model.Vs, np.sqrt(self.cond_variance_V))
            ar.bs = norm.rvs(self.global_ar_model.bs, np.sqrt(self.cond_variance_b))
            ar.Sigmas = self.global_ar_model.Sigmas.copy()

    def log_prior(self):
        lp = 0
        for ar in self.per_group_ar_models:
            lp += ar.log_prior()
        return lp

    def log_likelihoods(self, data, input, mask, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]].\
            log_likelihoods(data, input, mask, tag)

    def compute_sample_size(self, datas, inputs, masks, tags):
        sample_sizes = np.zeros(self.G)
        for g, (ar_model, tag) in enumerate(zip(self.per_group_ar_models, self.tags)):
            if any([t == tag for t in tags]):
                # Pull out the subset of data that corresponds to this tag
                tdatas = [d for d, t in zip(datas, tags)        if t == tag]
                tinpts = [i for i, t in zip(inputs, tags)       if t == tag]
                tmasks = [m for m, t in zip(masks, tags)        if t == tag]
                ttags  = [t for t    in tags                    if t == tag]
            
                # Compute total sample size for this tag
                sample_sizes[g] = ar_model.compute_sample_size(tdatas, tinpts, tmasks, ttags)

        return sample_sizes

    def expected_sufficient_stats(self, expectations, datas, inputs, masks, tags):
        # assumes that each input is a list of length 1
        stats = []
        for ar_model, tag in zip(self.per_group_ar_models, self.tags):

            if any([t == tag for t in tags]):
                # Pull out the subset of data that corresponds to this tag
                texpts = [e for e, t in zip(expectations, tags) if t == tag]
                tdatas = [d for d, t in zip(datas, tags)        if t == tag]
                tinpts = [i for i, t in zip(inputs, tags)       if t == tag]
                tmasks = [m for m, t in zip(masks, tags)        if t == tag]
                ttags  = [t for t    in tags                    if t == tag]
            
                # Compute expected sufficient stats for this subset of data
                these_stats = ar_model.expected_sufficient_stats(texpts, 
                                                                 tdatas, 
                                                                 tinpts, 
                                                                 tmasks, 
                                                                 ttags)

                stats.append(these_stats)
            else:
                stats.append(None)

        return stats

    def _m_step_global(self):
        # Note: we could explore smarter averaging techniques for estimating
        #       the global parameters.  E.g. using uncertainty estimages for
        #       the per-group parameters in a hierarchical Bayesian fashion.
        self.global_ar_model.As = np.mean([ar.As for ar in self.per_group_ar_models], axis=0)
        self.global_ar_model.Vs = np.mean([ar.Vs for ar in self.per_group_ar_models], axis=0)
        self.global_ar_model.bs = np.mean([ar.bs for ar in self.per_group_ar_models], axis=0)

        # TODO: Determine the correct MLE for Sigma given Sigma_i ~ IW(Sigma, nu)
        self.global_ar_model.Sigmas = np.mean([ar.Sigmas for ar in self.per_group_ar_models], axis=0)

    def _update_hierarchical_prior(self):
        # Update the per-group AR objects to have the global AR model
        # parameters as their prior.
        for ar in self.per_group_ar_models:
            ar.set_prior(self.global_ar_model.As, self.cond_variance_A,
                         self.global_ar_model.Vs, self.cond_variance_V,
                         self.global_ar_model.bs, self.cond_variance_b,
                         self.global_ar_model.Sigmas, self.cond_dof_Sigma)

    def m_step(self, expectations, datas, inputs, masks, tags,
               sufficient_stats=None, 
               **kwargs):

        # Collect sufficient statistics for each group
        if sufficient_stats is None:
            sufficient_stats = \
                self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)
        else:
            assert isinstance(sufficient_stats, list) and \
                   len(sufficient_stats) == self.G

        # Update the per-group weights
        for ar_model, stats in zip(self.per_group_ar_models, sufficient_stats):
            # Note: this is going to perform M-steps even for groups that 
            #       are not present in this minibatch.  Hopefully this isn't
            #       too much extra overhead.
            if stats is not None:
                ar_model.m_step(None, None, None, None, None, 
                                sufficient_stats=stats)

        # Update the shared weights
        self._m_step_global()
        self._update_hierarchical_prior()

    def stochastic_m_step(self, 
                          optimizer_state,
                          total_sample_size,
                          expectations,
                          datas,
                          inputs,
                          masks,
                          tags,
                          step_size=0.5):
        """
        """
        # Get the expected sufficient statistics for this minibatch
        # Note: this is an array of length num_groups (self.G)
        #       and entries in the array are None if there is no
        #       data with the corresponding tag in this minibatch.
        stats = self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)

        # Scale the statistics by the total sample size on a per-group basis
        this_sample_size = self.compute_sample_size(datas, inputs, masks, tags)
        for g in range(self.G):
            if stats[g] is not None:
                stats[g] = tuple(map(lambda x: x * total_sample_size[g] / this_sample_size[g], stats[g]))

        # Combine them with the running average sufficient stats
        if optimizer_state is not None:
            # we've seen some data before, but not necessarily from all groups
            for g in range(self.G):
                if optimizer_state[g] is not None and stats[g] is not None:
                    # we've seen this group before and we have data for it. 
                    # update its stats.
                    stats[g] = convex_combination(optimizer_state[g], stats[g], step_size)
                elif optimizer_state[g] is not None and stats[g] is None:
                    # we've seen this group before but not in this minibatch.
                    # pass existing stats through.
                    stats[g] = optimizer_state[g]
        else:
            # first time we're seeing any data.  return this minibatch's stats.
            pass

        # Call the regular m-step with these sufficient statistics
        self.m_step(None, None, None, None, None, sufficient_stats=stats)

        # Return the update state (i.e. the new stats)
        return stats

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.per_group_ar_models[self.tags_to_indices[tag]]. \
            sample_x(z, xhist, input, tag, with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]]. \
            smooth(expectations, data, input, tag)
