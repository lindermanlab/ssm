import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import norm
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.util import ensure_args_are_lists


class _Hierarchical(object):
    """
    Base class for hierarchical models.  Maintains a parent class and a
    bunch of children with their own perturbed parameters.
    """
    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda

        # Top-level parameters (parent)
        self.parent = base_class(*args, **kwargs)

        # Make models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            ch = self.children[tag] = base_class(*args, **kwargs)
            ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

    @property
    def params(self):
        prms = (self.parent.params,)
        for tag in self.tags:
            prms += (self.children[tag].params,)
        return prms

    @params.setter
    def params(self, value):
        self.parent.params = value[0]
        for tag, prms in zip(self.tags, value[1:]):
            self.children[tag].params = prms

    def permute(self, perm):
        self.parent.permute(perm)
        for tag in self.tags:
            self.children[tag].permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        for tag in self.tags:
            self.children[tag].params = copy.deepcopy(self.parent.params)

    def log_prior(self):
        lp = self.parent.log_prior()

        # Gaussian likelihood on each child param given parent param
        for tag in self.tags:
            for pprm, cprm in zip(self.parent.params, self.children[tag].params):
                lp += np.sum(norm.logpdf(cprm, pprm, self.lmbda))
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # Optimize parent and child parameters at the same time with SGD
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = \
            optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)


class HierarchicalInitialStateDistribution(_Hierarchical):
    def log_initial_state_distn(self, data, input, mask, tag):
        return self.log_pi0 - logsumexp(self.log_pi0)


class HierarchicalTransitions(_Hierarchical):
    def log_transition_matrices(self, data, input, mask, tag):
        return self.children[tag].log_transition_matrices(data, input, mask, tag)


class HierarchicalObservations(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag, with_noise=with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(expectations, data, input, tag)


class HierarchicalEmissions(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_y(self, z, x, input=None, tag=None):
        return self.children[tag].sample_y(z, x, input=input, tag=tag)

    def initialize_variational_params(self, data, input, mask, tag):
        return self.children[tag].initialize_variational_params(data, input, mask, tag)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        return self.children[tag].smooth(expected_states, variational_mean, data, input, mask, tag)

