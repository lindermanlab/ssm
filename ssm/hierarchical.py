import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import norm

from ssm.util import ensure_args_are_lists


class _Hierarchical(object):
    """
    Base class for hierarchical models.  Maintains a parent class and a 
    bunch of children with their own perturbed parameters. 
    """
    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, **kwargs):
        # How similar should parent and child params be
        self.lmbda = lmbda

        # Top-level AR parameters (parent mean)
        self.parent = base_class(*args, **kwargs)

        # Make AR models for each tag
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
            cprm = copy.deepcopy(self.parent.params)
            for prm in cprm:
                prm += np.sqrt(self.lmbda) + npr.randn(*prm.shape)
            self.children[tag].params = cprm

    def log_prior(self):
        lp = self.parent.log_prior()

        # # Gaussian likelihood on each child param given parent param
        for tag in self.tags:
            for pprm, cprm in zip(self.parent.params, self.children[tag].params):
                lp = lp + np.sum(norm.logpdf(cprm, pprm, self.lmbda))
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        warnings.warn("m_step for does not include the global prior. "
                      "We still need to implement this feature.")
        
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        for tag in self.tags:
            self.children[tag].m_step(
                [e for e,t in zip(expectations, tags) if t == tag],
                [d for d,t in zip(datas, tags) if t == tag],
                [i for i,t in zip(inputs, tags) if t == tag],
                [m for m,t in zip(masks, tags) if t == tag],
                [t for t in tags if t == tag],
                **kwargs)

        # Set the parent params to the average of the child params
        avg_params = ()
        for i in range(len(self.parent.params)):
            avg_params += (np.mean([self.children[tag].params[i] for tag in self.tags], axis=0),)
        self.parent.params = avg_params


class HierarchicalInitialStateDistribution(_Hierarchical):
    def log_initial_state_distn(self, data, input, mask, tag):
        return self.log_pi0 - logsumexp(self.log_pi0)


class HierarchicalTransitions(_Hierarchical):
    def log_transition_matrices(self, data, input, mask, tag):
        return self.children[tag].log_transition_matrices(data, input, mask, tag)


class HierarchicalObservations(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag)
    
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

