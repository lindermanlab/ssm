
import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import gammaln, digamma, logsumexp
from autograd.scipy.special import logsumexp

from ssm.util import random_rotation, ensure_args_are_lists, \
    logistic, logit, one_hot
from ssm.regression import fit_linear_regression, generalized_newton_studentst_dof
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm.stats as stats
from ssm.observations import Observations, AutoRegressiveObservations

class SparseAutoRegressiveObservations(AutoRegressiveObservations):
    """
    AutoRegressive observation model with sparse weights.
        (x_t | z_t = k, u_t) ~ N((A_k * Amask_k) x_{t-1} + b_k + V_k u_t, sigma_k^2 I)
    where
        A_k is a set of regression weights
        Amask_k is a binary mask that zeros out some weights
        sigma_k^2 is the noise variance
    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, K, D, M=0, lags=1,
                 prior_precision_A=100,
                 prior_precision_b=1e-8,
                 prior_precision_V=1e-8,
                 block_size=(1,1),
                 sparsity=0.5,
                 start_iter=0):
        assert lags == 1, "Sparse AR model is only implemented for lags==1"

        super(SparseAutoRegressiveObservations, self).\
            __init__(K, D, M, lags=lags,
                     l2_penalty_A=prior_precision_A,
                     l2_penalty_b=prior_precision_b,
                     l2_penalty_V=prior_precision_V)

        # Initialize the dynamics and the noise covariances
        self._As = .80 * np.array([
                np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))])
            for _ in range(K)])

        # Initialize the dynamics mask
        assert isinstance(block_size, (tuple, list)) and len(block_size) == 2
        assert D % block_size[0] == 0, "block size must perfectly divide D"
        assert D % block_size[1] == 0, "block size must perfectly divide D"
        self.block_size = block_size
        self.As_mask = np.ones((K, D // block_size[0], D // block_size[1]), dtype=bool)
        self.As_mask_posteriors = [None] * K
        assert 0 < sparsity < 1
        self.sparsity = sparsity

        # Our current code assumes isotropic variance for efficiency.
        # Get rid of the square root parameterization and replace with sigmasq
        del self._sqrt_Sigmas_init
        del self._sqrt_Sigmas
        self.sigmasq_inits = np.ones((K, ))
        self.sigmasqs = np.ones((K, D))

        self.l2_penalty_A = prior_precision_A
        self.l2_penalty_b = prior_precision_b
        self.l2_penalty_V = prior_precision_V

        self.iter_count = -1
        self.start_iter = start_iter

    @property
    def As(self):
        return self._As * np.kron(self.As_mask, np.ones(self.block_size))

    @As.setter
    def As(self, value):
        self._As = value

    @property
    def Sigmas_init(self):
        return np.array([sigmasq * np.eye(self.D) for sigmasq in self.sigmasq_inits])

    @property
    def Sigmas(self):
        return np.array([sigmasq * np.eye(self.D) for sigmasq in self.sigmasqs])

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self.sigmasqs,)

    @params.setter
    def params(self, value):
        self._log_sigmasq = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])


    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self.As_mask = self.As_mask[perm]
        self.sigmasq_inits = self.sigmasq_inits[perm]
        self.sigmasqs = self.sigmasqs[perm]


    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        D, L = self.D, self.lags
        mus = self._compute_mus(data, input, mask, tag)

        ll_init = np.column_stack([stats.diagonal_gaussian_logpdf(data[:L], mu[:L], sigmasq * np.ones(D))
                               for mu, sigmasq in zip(mus, self.sigmasq_inits)])

        ll_ar = np.column_stack([stats.diagonal_gaussian_logpdf(data[L:], mu[L:], sigmasq * np.ones(D))
                               for mu, sigmasq in zip(mus, self.sigmasqs)])

        return np.row_stack((ll_init, ll_ar))

    def fit_sparse_linear_regression(self, ExxT, ExyT, EyyT, En, sigmasq, J0, h0, rho):
        from itertools import product
        from scipy.linalg import solve_triangular
        D, M = self.D, self.M
        D_in = D + M + 1
        S_out, S_in = self.block_size
        B_out = D // S_out
        B_in = D // S_in

        # Initialize the outputs
        W = np.zeros((D, D_in))
        Z = np.ones((B_out, B_in))
        Z_posteriors = []

        # Compute the posterior distribution for each row of Z
        for bo in range(B_out):
            # Find the output slice
            slc = slice(bo * S_out, (bo + 1) * S_out)
            # Enumerate all 2^{B_in} assignments of the binary mask
            assert B_in <= 16, "You probably don't want to explicitly enumerate 2^B_{in}" \
                               "assignments when B_{in} > 16."
            assignments = np.array(list(product([0, 1], repeat=B_in-1)))
            log_probs = np.zeros(2 ** (B_in-1))
            for ind, zk_temp in enumerate(assignments):
                # Always include the diagonal block in zk
                zk = np.ones(B_in)
                inc_idx = np.delete(np.arange(0, B_in), bo)
                zk[inc_idx] = zk_temp

                # Add the prior of the sparsity for this row of blocks
                log_probs[ind] = np.sum(zk * np.log(rho) + (1 - zk) * np.log(1 - rho))

                # Expand the binary mask and pad with extra ones for the input and intercept
                z = np.concatenate([np.kron(zk, np.ones(S_in)), np.ones(M + 1)])
                for i in range(bo * S_out, (bo + 1) * S_out):
                    J = J0 + 1 / sigmasq[i] * ExxT * np.outer(z, z)
                    # J = J0 + 1 / sigmasq * ExxT * np.outer(z, z)
                    L = np.linalg.cholesky(J)
                    h = h0[:, i] + 1 / sigmasq[i] * ExyT[:, i] * z
                    # h = h0[:, i] + 1 / sigmasq * ExyT[:, i] * z
                    tmp = solve_triangular(L, h, lower=True)
                    log_probs[ind] += 0.5 * np.sum(tmp ** 2) - np.sum(np.log(np.diag(L)))

            # Save the posterior
            Z_posteriors.append((assignments, np.exp(log_probs - logsumexp(log_probs))))
            # Find the most likely assignment for these outputs
            Z[bo, inc_idx] = assignments[np.argmax(log_probs)]
            z = np.concatenate([np.kron(Z[bo], np.ones(S_in)), np.ones(M+1)])
            for i in range(bo * S_out, (bo + 1) * S_out):
                J = J0 + 1 / sigmasq[i] * ExxT * np.outer(z, z)
                h = h0[:, i] + 1 / sigmasq[i] * ExyT[:, i] * z
                # J = J0 + 1 / sigmasq * ExxT * np.outer(z, z)
                # h = h0[:, i] + 1 / sigmasq * ExyT[:, i] * z
                W[i] = np.linalg.solve(J, h).T

        # Solve for the optimal variance
        EWxyT =  W @ ExyT
        sqerr = EyyT - EWxyT.T - EWxyT + W @ ExxT @ W.T
        sigmasq = np.diag(sqerr) / En ### MSIG
        # sigmasq = np.sum(np.diag(sqerr)) / (En * D)

        # Unpack the weights and intercept
        A, V, b = W[:, :D], W[:, D:D+M], W[:, -1]
        return A, V, b, Z, sigmasq, Z_posteriors

    def m_step(self, expectations, datas, inputs, masks, tags,
               J0=None, h0=None, continuous_expectations=None, **kwargs):
        """Compute M-step for Gaussian Auto Regressive Observations.
        If `continuous_expectations` is not None, this function will
        compute an exact M-step using the expected sufficient statistics for the
        continuous states. In this case, we ignore the prior provided by (J0, h0),
        because the calculation is exact. `continuous_expectations` should be a tuple of
        (Ex, Ey, ExxT, ExyT, EyyT).
        If `continuous_expectations` is None, we use `datas` and `expectations,
        and (optionally) the prior given by (J0, h0). In this case, we estimate the sufficient
        statistics using `datas,` which is typically a single sample of the continuous
        states from the posterior distribution.
        """
        K, D, M, lags = self.K, self.D, self.M, self.lags

        S_out, S_in = self.block_size
        B_out = D // S_out
        B_in = D // S_in

        self.iter_count=self.iter_count+1
        print(self.iter_count)



        #Here, we do sparse spike-and-slab regression
        #For our prior on the precision, we use the precision of a standard regression
        if self.iter_count>self.start_iter:

            #First, run the normal regression (to get the prior for the sparse regression)

            # Initialize the prior
            J0=np.zeros([K,D*lags+M+1,D*lags+M+1])
            h0=np.zeros([K,D*lags+M+1,D])

            # Collect the data and weights
            if continuous_expectations is None:
                ExuxuTs, ExuyTs, EyyTs, Ens = \
                    self._get_sufficient_statistics(expectations, datas, inputs)
            else:
                ExuxuTs, ExuyTs, EyyTs, Ens = \
                    self._extend_given_sufficient_statistics(expectations, continuous_expectations, inputs)

            # Set up the priors
            J0_diag_tmp = np.concatenate((1e-8 * np.ones(D * lags),
                                     self.l2_penalty_V * np.ones(M),
                                     self.l2_penalty_b * np.ones(1)))
            J0_tmp = np.tile(np.diag(J0_diag_tmp)[None, :, :], (K, 1, 1))
            h0_tmp = np.concatenate((1e-8 * np.eye(D),
                                 np.zeros((D * (lags - 1), D)),
                                 np.zeros((M + 1, D))))
            h0_tmp = np.tile(h0_tmp[None, :, :], (K, 1, 1))




            #Use the precision of the standard regression as a prior for the sparse regression
            for k in range(K):
                Wk = np.linalg.solve(ExuxuTs[k] + J0_tmp[k], ExuyTs[k] + h0_tmp[k]).T
                A = Wk[:, :D]
                mask=1-np.kron(np.identity(B_out), np.ones((S_out, S_out), dtype=bool)) #NEED TO UPDATE W/ EITHER IN OR OUT
                # prec=1/(np.var(A-np.identity(D)))*2
                prec=1/(np.var(A[mask.astype(bool)]))#*1.5
                print(prec)
                J0_diag = np.concatenate((prec * np.ones(D),
                                         self.l2_penalty_V * np.ones(M),
                                         self.l2_penalty_b * np.ones(1)))
                J0[k] = np.diag(J0_diag)[None, :, :]
                h0[k] = np.concatenate((prec * np.eye(D),
                                     np.zeros((D * (lags - 1), D)),
                                     np.zeros((M + 1, D))))


            # Solve the sprase linear regressions
            for k in range(K):
                self._As[k], self.Vs[k], self.bs[k], self.As_mask[k], self.sigmasqs[k], self.As_mask_posteriors[k] = \
                    self.fit_sparse_linear_regression(ExuxuTs[k], ExuyTs[k], EyyTs[k], Ens[k],
                                                      self.sigmasqs[k], J0[k], h0[k], self.sparsity)


        else:

            # Set up the prior
            if J0 is None:
                J0_diag = np.concatenate((1e-8 * np.ones(D * lags),
                                         self.l2_penalty_V * np.ones(M),
                                         self.l2_penalty_b * np.ones(1)))
                J0 = np.tile(np.diag(J0_diag)[None, :, :], (K, 1, 1))

            if h0 is None:
                h0 = np.concatenate((1e-8 * np.eye(D),
                                     np.zeros((D * (lags - 1), D)),
                                     np.zeros((M + 1, D))))
                h0 = np.tile(h0[None, :, :], (K, 1, 1))

            nu0 = 1e-4
            Psi0 = 1e-4*np.eye(D)

            # Collect sufficient statistics
            if continuous_expectations is None:
                ExuxuTs, ExuyTs, EyyTs, Ens = \
                    self._get_sufficient_statistics(expectations, datas, inputs)
            else:
                ExuxuTs, ExuyTs, EyyTs, Ens = \
                    self._extend_given_sufficient_statistics(expectations, continuous_expectations, inputs)

            # Solve the linear regressions
            As = np.zeros((K, D, D * lags))
            Vs = np.zeros((K, D, M))
            bs = np.zeros((K, D))
            # Sigmas = np.zeros((K, D, D))
            for k in range(K):
                Wk = np.linalg.solve(ExuxuTs[k] + J0[k], ExuyTs[k] + h0[k]).T
                self._As[k] = Wk[:, :D * lags]
                self.Vs[k] = Wk[:, D * lags:-1]
                self.bs[k] = Wk[:, -1]

                # sqerr = EyyTs[k] - 2 * Wk @ ExuyTs[k] + Wk @ ExuxuTs[k] @ Wk.T
                sqerr = EyyTs[k] - Wk @ ExuyTs[k] - ExuyTs[k].T @ Wk.T  + Wk @ ExuxuTs[k] @ Wk.T
                self.sigmasqs[k] = np.diag(sqerr) / Ens[k]
                # self.sigmasqs[k] = (np.diag(sqerr) + Psi0) / (Ens[k] + nu0 + D + 1)


        # If any states are unused, set their parameters to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                self._As[k] = self._As[i] + 0.01 * npr.randn(*self._As[i].shape)
                self.Vs[k] = self.Vs[i] + 0.01 * npr.randn(*self.Vs[i].shape)
                self.bs[k] = self.bs[i] + 0.01 * npr.randn(*self.bs[i].shape)
                self.As_mask[k] = self.As_mask[i]
                self.As_mask_posteriors[k] = self.As_mask_posteriors[i]
                self.sigmasqs[k] = self.sigmasqs[i]






class IdentityAutoRegressiveObservations(Observations):

    def __init__(self, K, D, M):
        super(IdentityAutoRegressiveObservations, self).__init__(K, D, M)
        self.log_sigmas = -2 + npr.randn(K, D) # log of the variance
        self.log_sigma_init= np.zeros(D) # log of the variance of the initial data point
        self.K=K
        self.D=D
        self.As=[np.identity(D)]

    @property
    def params(self):
        return self.log_sigmas

    @property
    def Sigmas_init(self):
        return np.array([np.diag(np.exp(self.log_sigma_init))])

    @property
    def Sigmas(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self.log_sigmas])

    @params.setter
    def params(self, value):
        self.log_sigmas = value

    def permute(self, perm):
        self.log_sigmas = self.log_sigmas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.log_sigmas = -2 + npr.randn(self.K, self.D)#np.log(sigmas + 1e-16)

    def log_likelihoods(self, data, input, mask, tag):
        sigmas = np.exp(self.log_sigmas) + 1e-16
        sigma_init=np.exp(self.log_sigma_init)+1e-16

        #Log likelihood of data (except for first point)
        ll1 = -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[1:, None, :] - data[:-1, None, :])**2 / sigmas)
            * mask[1:, None, :], axis=2)

        ll2 = np.zeros([1,self.K])

        ll = np.concatenate((ll1,ll2),axis=0)

        return ll

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D = self.D
        sigmas = np.exp(self.log_sigmas) if with_noise else np.zeros((self.K, self.D))
        sigma_init = np.exp(self.log_sigma_init) if with_noise else 0

        if xhist.shape[0] == 0:
            return np.sqrt(sigma_init[z]) * npr.randn(D)
        else:
            return xhist[-1] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            sqerr = (x[1:] - x[:-1])**2
            d2=np.average(sqerr, weights=weights[1:,k], axis=0)

            self.log_sigmas[k] = np.log(d2)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute negative Hessian of autoregressive obsevations with missing data."

        # initial distribution contributes a Gaussian term to first diagonal block
        J_ini = np.sum(Ez[0, :, None, None] * np.linalg.inv(self.Sigmas_init), axis=0)

        # first part is transition dynamics - goes to all terms except final one
        # E_q(z) x_{t} A_{z_t+1}.T Sigma_{z_t+1}^{-1} A_{z_t+1} x_{t}
        inv_Sigmas = np.linalg.inv(self.Sigmas)
        dynamics_terms = np.array([A.T@inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)]) # A^T Qinv A terms
        J_dyn_11 = np.sum(Ez[1:,:,None,None] * dynamics_terms[None,:], axis=1)

        # second part of diagonal blocks are inverse covariance matrices - goes to all but first time bin
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} x_{t+1}
        J_dyn_22 = np.sum(Ez[1:,:,None,None] * inv_Sigmas[None,:], axis=1)

        # lower diagonal blocks are (T-1,D,D):
        # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} A_{z_t+1} x_t
        off_diag_terms = np.array([inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)])
        J_dyn_21 = -1 * np.sum(Ez[1:,:,None,None] * off_diag_terms[None,:], axis=1)

        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22
