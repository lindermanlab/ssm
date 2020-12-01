from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet
from autograd import hessian

from ssm.util import one_hot, logistic, relu, rle, ensure_args_are_lists, LOG_EPS, DIV_EPS
from ssm.regression import fit_multiclass_logistic_regression, fit_negative_binomial_integer_r
from ssm.stats import multivariate_normal_logpdf
from ssm.optimizers import adam, bfgs, lbfgs, rmsprop, sgd


from ssm.transitions import Transitions, InputDrivenTransitions


class StickyRecurrentTransitions(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0, l2_penalty=1e-8, l1_penalty=1e-8):
        super(StickyRecurrentTransitions, self).__init__(K, D, M, alpha=alpha, kappa=kappa)

        # Parameters linking past observations to state distribution
        self.Rs = np.zeros((K, D))
        self.Ss = np.zeros((K, D))
        self.l2_penalty=l2_penalty
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return super(StickyRecurrentTransitions, self).params + (self.Rs,self.Ss)

    @params.setter
    def params(self, value):
        self.Rs = value[-2]
        self.Ss = value[-1]
        super(StickyRecurrentTransitions, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(StickyRecurrentTransitions, self).permute(perm)
        self.Rs = self.Rs[perm]
        self.Ss = self.Ss[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]

        # Past observations effect

        #Off diagonal elements of transition matrix (state switches), from past observations
        log_Ps_offdiag = np.tile(np.dot(data[:-1], self.Rs.T)[:,None,:],(1, self.K, 1))
        mult_offdiag=1-np.tile(np.identity(self.K)[None,:,:],(log_Ps_offdiag.shape[0],1,1))

        #Diagonal elements of transition matrix (stickiness), from past observations
        log_Ps_diag = np.tile(np.dot(data[:-1], self.Ss.T)[:,None,:],(1, self.K, 1))
        mult_diag=np.tile(np.identity(self.K)[None,:,:],(log_Ps_diag.shape[0],1,1))

        log_Ps = log_Ps + log_Ps_diag*mult_diag #Diagonal elements (stickness) from past observations
        log_Ps = log_Ps + log_Ps_offdiag*mult_offdiag #Off diagonal elements (state switching) from past observations

        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True) #Normalize

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # SWAP?
                #Sticky terms
                if k1==k2:
                    Rv = vtilde@self.Ss[k2:k2+1,:]
                    hess += Ez[k1,k2] * \
                            ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Ss[k2:k2+1,:], self.Ss[k2:k2+1,:]) \
                            + np.einsum('ti, tj -> tij', Rv, Rv))
                #Switching terms
                else:
                    Rv = vtilde@self.Rs[k2:k2+1,:]
                    hess += Ez[k1,k2] * \
                            ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                            + np.einsum('ti, tj -> tij', Rv, Rv))
        return -1 * hess


    def log_prior(self):
        #L2 penalty
        lp = np.sum(-0.5 * self.l2_penalty * self.Rs**2)
        lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ss**2)

        #L1 penalty
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        lp = lp + np.sum(-1* self.l1_penalty * np.abs(self.Ss))

        return lp



class StickyRecurrentOnlyTransitions(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0, l2_penalty_similarity=1e-8, l1_penalty=1e-8):
        super(StickyRecurrentOnlyTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M) ## NOTE - W (REFLECTING INPUTS) HAS NOT BEEN USED HERE
        self.Rs = np.zeros([K, D])
        self.Ss = np.zeros([K, D])
        self.r = np.zeros(K)
        self.s = np.zeros(K)

        #Regularization parameters
        self.l2_penalty_similarity=l2_penalty_similarity
        self.l1_penalty=l1_penalty

    @property
    def params(self):
        return self.Ws, self.Rs, self.Ss, self.r, self.s

    @params.setter
    def params(self, value):
        self.Ws, self.Rs, self.Ss, self.r, self.s = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.Ss = self.Ss[perm]
        self.r = self.r[perm]
        self.s = self.s[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape

        #Off diagonal elements of transition matrix (state switches), from past observations
        log_Ps_offdiag = np.tile(np.dot(data[:-1], self.Rs.T)[:,None,:],(1, self.K, 1))
        mult_offdiag=1-np.tile(np.identity(self.K)[None,:,:],(log_Ps_offdiag.shape[0],1,1))

        #Diagonal elements of transition matrix (stickiness), from past observations
        log_Ps_diag = np.tile(np.dot(data[:-1], self.Ss.T)[:,None,:],(1, self.K, 1))
        mult_diag=np.tile(np.identity(self.K)[None,:,:],(log_Ps_diag.shape[0],1,1))

        log_Ps = log_Ps_diag*mult_diag #Diagonal elements (stickness) from past observations
        log_Ps = log_Ps + np.identity(self.K)*self.s #Diagonal elements (stickness) bias
        log_Ps = log_Ps + log_Ps_offdiag*mult_offdiag #Off diagonal elements (state switching) from past observations
        log_Ps = log_Ps + (1-np.identity(self.K))*self.r #Off diagonal elements (state switching) bias

        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", num_iters=1000, **kwargs)


    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian

        T, D = data.shape

        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        #Loop through transitions between K states
        for k1 in range(self.K):
            for k2 in range(self.K):
                vtilde = vtildes[:,k1,k2][:,None] # There's a chance the order of k1 and k2 is flipped
                if k1==k2:
                    Rv = vtilde@self.Ss[k2:k2+1,:]
                    hess += Ez[k1,k2] * \
                            ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Ss[k2:k2+1,:], self.Ss[k2:k2+1,:]) \
                            + np.einsum('ti, tj -> tij', Rv, Rv))
                else:
                    Rv = vtilde@self.Rs[k2:k2+1,:]
                    hess += Ez[k1,k2] * \
                            ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs[k2:k2+1,:], self.Rs[k2:k2+1,:]) \
                            + np.einsum('ti, tj -> tij', Rv, Rv))
        return -1 * hess
        # return np.zeros([T-1,D,D])

    def log_prior(self):

        #L1 penalty on the parameters R and S
        lp = np.sum(-1 * self.l1_penalty * np.abs(self.Rs))
        lp = lp + np.sum(-1* self.l1_penalty * np.abs(self.Ss))

        #L2 penalty on the similarity between the sticky and switching Parameters
        #In the limit of an infinite penalty, the solution will become R=S and r=s,
        # which is the "recurrent only" model
        # This could be a reasonable prior as X might be similar when switching into a discrete state and staying in a discrete state
        lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.Rs-self.Ss)**2)
        # lp = lp + np.sum(-0.5 * self.l2_penalty_similarity * (self.r-self.s)**2)  #Consider not including this term??

        return lp
