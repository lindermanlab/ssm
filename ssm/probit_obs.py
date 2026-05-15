"""Binary input-driven observations with a probit (Phi) link.

`InputDrivenObservations` in `ssm.observations` parameterises the binary
emission with a logit link.  For SDT-style readouts of the per-state
weights, a probit link is more natural: with probit,

    P(y = 1 | x, z = k) = Phi(w_k . x)

and the standard SDT identities

    d'_k = z(HR_k) - z(FAR_k)
    c_k  = -0.5 * (z(HR_k) + z(FAR_k))

reduce, under a +/-1 stimulus coding, to

    d'_k = 2 * w_stim,k
    c_k  =    -w_bias,k

with no logit->probit scale approximation (no lambda ~= 1.7).

This class is a minimal addition that mirrors the style of
`InputDrivenObservations`: a Gaussian prior on the per-state weight
vectors, autograd-compatible log-likelihood, and an autograd-based
m-step via scipy.optimize.minimize.  It is hardwired to C = 2 (binary
output) because that is the only setting in which the probit-SDT
correspondence is well-defined.
"""
from __future__ import annotations
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.stats import norm
from scipy.optimize import minimize

from ssm.observations import Observations
import ssm.stats as stats


class ProbitInputDrivenObservations(Observations):
    """Bernoulli emissions with a probit link (binary; C = 2).

    Per-state weights `Wk` have shape (K, M).  P(y=1 | x, z=k) = Phi(Wk[k] . x).
    """

    def __init__(self, K, D, M=0, prior_mean: float = 0.0,
                 prior_sigma: float = 1000.0):
        super().__init__(K, D, M)
        assert D == 1, "ProbitInputDrivenObservations is binary (D = 1)."
        self.C = 2
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma
        # Per-state weight vector.
        self.Wk = npr.randn(K, M)

    @property
    def params(self):
        return self.Wk

    @params.setter
    def params(self, value):
        self.Wk = value

    def permute(self, perm):
        self.Wk = self.Wk[perm]

    def log_prior(self):
        lp = 0.0
        for k in range(self.K):
            w = self.Wk[k]
            lp += stats.multivariate_normal_logpdf(
                w, mus=np.repeat(self.prior_mean, self.M),
                Sigmas=(self.prior_sigma ** 2) * np.identity(self.M))
        return lp

    def _eta(self, input):
        """Linear predictor eta_{t,k} = Wk[k] . input[t], shape (T, K)."""
        return input @ self.Wk.T  # (T, M) @ (M, K) = (T, K)

    def log_likelihoods(self, data, input, mask, tag):
        """Log P(y_t | x_t, z_t = k) for every (t, k).  Shape (T, K)."""
        assert self.D == 1, "ProbitInputDrivenObservations is D = 1."
        eta = self._eta(input)                 # (T, K)
        y = data[:, 0]                         # (T,)  0/1
        # log Phi(eta) when y=1, log Phi(-eta) when y=0  (= log(1 - Phi(eta))).
        signed_eta = np.where(y[:, None] == 1, eta, -eta)
        ll = norm.logcdf(signed_eta)           # (T, K)
        if mask is not None:
            m = mask[:, 0:1] if mask.ndim == 2 else mask
            ll = ll * np.asarray(m)
        return ll

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        assert self.D == 1
        if input.ndim == 1 and input.shape == (self.M,):
            input = np.expand_dims(input, axis=0)
        eta = self._eta(input)                 # (T, K)
        T = eta.shape[0]
        if T == 1:
            p1 = float(norm.cdf(eta[0, z]))
            y = 1 if (not with_noise and p1 >= 0.5) else int(npr.rand() < p1)
            return np.array([y])
        out = np.zeros(T, dtype=int)
        for t in range(T):
            p1 = float(norm.cdf(eta[t, z[t]]))
            out[t] = 1 if (not with_noise and p1 >= 0.5) else int(npr.rand() < p1)
        return out

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """One BFGS step per state on the negative expected complete LL.

        Objective for state k:
            - sum_t  E[z_t = k]  * log P(y_t | x_t, z_t = k)
            + 0.5 / sigma^2 * ||w_k - mu||^2
        normalised by T_total for numerical stability.
        """
        T_total = sum(d.shape[0] for d in datas)
        sig2 = self.prior_sigma ** 2 if self.prior_sigma != 0 else None
        mu = self.prior_mean

        def _objective(wk, k):
            obj = 0.0
            for data, input, mask, (Ez, _, _) in zip(datas, inputs, masks,
                                                     expectations):
                eta = input @ wk                          # (T,)
                y = data[:, 0]
                signed = np.where(y == 1, eta, -eta)
                ll = norm.logcdf(signed)                  # (T,)
                obj = obj - np.sum(Ez[:, k] * ll)
            if sig2 is not None:
                obj = obj + 0.5 / sig2 * np.sum((wk - mu) ** 2)
            return obj / T_total

        grad_fn = grad(lambda wk, k: _objective(wk, k))

        for k in range(self.K):
            x0 = np.array(self.Wk[k])
            sol = minimize(
                _objective, x0, args=(k,), jac=grad_fn,
                method="L-BFGS-B",
            )
            self.Wk[k] = sol.x

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError
