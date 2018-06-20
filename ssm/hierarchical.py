import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.observations import _Observations
from ssm.util import ensure_args_are_lists


class HierarchicalObservations(_Observations):

    def __init__(self, base_class, K, D, M=0, tags=(None,), lmbda=0.01):
        super(_HierarchicalObservations, self).__init__(K, D, M)

        # How similar should parent and child params be
        self.lmbda = lmbda

        # Top-level AR parameters (parent mean)
        self.parent = base_class(K, D, M=M)

        # Make AR models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            ch = self.children[tag] = base_class(K, D, M=M)
            ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

    @property
    def params(self):
        return self.parent.params, (self.children[tag].params for tag in self.tags)

    @params.setter
    def params(self, value):
        self.parent.params, children_params = value
        for tag, prms in zip(self.tags, children_params):
            self.children[tag].params = prms

    def permute(self, perm):
        self.parent.permute(perm)
        for ch in self.children:
            ch.permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        for ch in self.children:
            cprm = copy.deepcopy(self.parent.params)
            for prm in cprm:
                prm += np.sqrt(lmbda) + npr.randn(*prm.shape)
            ch.params = cprm

    def log_prior(self):
        lp = 0

        # Gaussian likelihood on each AR param given global AR param
        for ch in self.children:
            for pprm, cprm in zip(self.parent.params, ch.params):
                lp += -0.5 * np.sum(np.log(2 * np.pi * self.lmbda) + (cprm - pprm)**2 / self.lmbda, axis=2)
        return lp

    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag]._log_likelihoods(data, input, mask, tag)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        warnings.warn("_m_step_observations for _HierarchicalAutoRegressiveHMMObservations "
                      "does not include the global prior. We still need to implement this feature.")
        
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

    def sample_x(self, z, xhist, input=None, tag=None):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(data, input=input, mask=mask, tag=tag)

