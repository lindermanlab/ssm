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
from autograd.scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import norm as snorm

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
        """Trust-region Newton (trust-ncg) per state on the negative expected
        complete LL with analytic gradient and Hessian.

        Objective for state k (normalised by total trial count):

            F_k(w) = -sum_t E[z_t=k] * log P(y_t | x_t; w)
                    + 0.5 / sigma^2 * ||w - mu||^2

        Letting s_t = 2 y_t - 1, eta_t = w . x_t, and z_t = s_t eta_t,

            log P(y_t | x_t; w) = log Phi(z_t)
            d/dw  = -s_t * lambda(z_t) * x_t       (lambda = phi/Phi)
            d^2/dw^2 = lambda(z_t) * (z_t + lambda(z_t)) * x_t x_t^T

        The Hessian factor lambda*(z+lambda) is the truncated-normal
        variance and is non-negative, so the objective is convex.
        """
        T_total = sum(d.shape[0] for d in datas)
        sig2 = (self.prior_sigma ** 2) if self.prior_sigma != 0 else None
        mu = self.prior_mean

        # Precompute per-dataset quantities that don't depend on w.
        s_arrs, X_arrs, Ez_arrs = [], [], []
        for data, input, (Ez, _, _) in zip(datas, inputs, expectations):
            y = data[:, 0]
            s_arrs.append(2.0 * y - 1.0)            # (T,)  ±1
            X_arrs.append(np.asarray(input))         # (T, M)
            Ez_arrs.append(np.asarray(Ez))           # (T, K)

        def _f_g_h(wk, k):
            f = 0.0
            g = np.zeros_like(wk)
            H = np.zeros((wk.size, wk.size))
            for s, X, Ez in zip(s_arrs, X_arrs, Ez_arrs):
                eta = X @ wk                         # (T,)
                z = s * eta                          # (T,)
                logPhi = snorm.logcdf(z)             # (T,)
                # inverse Mills ratio phi/Phi, stable for z << 0.
                lam = np.exp(snorm.logpdf(z) - logPhi)
                we = Ez[:, k]                        # (T,)
                f -= np.sum(we * logPhi)
                g += X.T @ (-we * s * lam)
                hf = we * lam * (z + lam)            # (T,)  >= 0
                H += (X * hf[:, None]).T @ X
            if sig2 is not None:
                f += 0.5 / sig2 * np.sum((wk - mu) ** 2)
                g += (wk - mu) / sig2
                H += np.eye(wk.size) / sig2
            return f / T_total, g / T_total, H / T_total

        for k in range(self.K):
            x0 = np.asarray(self.Wk[k], dtype=float)

            def _obj(w, k=k):
                return _f_g_h(w, k)[:2]

            def _hess(w, k=k):
                return _f_g_h(w, k)[2]

            sol = minimize(_obj, x0, jac=True, hess=_hess,
                           method="trust-ncg",
                           options=dict(maxiter=100))
            self.Wk[k] = sol.x

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError
