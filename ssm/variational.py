import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.emissions import _LinearEmissions
from ssm.preprocessing import interpolate_data
from ssm.primitives import lds_log_probability, lds_sample, lds_mean, \
                           block_tridiagonal_sample, hmm_expected_states, \
                           hmm_sample, block_tridiagonal_mean, block_tridiagonal_log_probability

from ssm.util import ensure_variational_args_are_lists

from warnings import warn

class VariationalPosterior(object):
    """
    Base class for a variational posterior distribution.

        q(z; phi) \approx p(z | x, theta)

    where z is a latent variable and x is the observed data.

    ## Reparameterization Gradients
    We assume that the variational posterior is "reparameterizable"
    in the sense that,

    z ~ q(z; phi)  =d  eps ~ r(eps); z = f(eps; phi).

    where =d denotes equal in distirbution.  If this is the case,
    we can rewrite

    L(phi) = E_q(z; phi) [g(z)] = E_r(eps) [g(f(eps; phi))]

    and

    dL/dphi = E_r(eps) [d/dphi g(f(eps; phi))]
            approx 1/S sum_s [d/dphi g(f(eps_s; phi))]

    where eps_s ~iid r(eps).  In practice, this Monte Carlo estimate
    of dL/dphi is lower variance than alternative approaches like
    the score function estimator.

    ## Amortization
    We also allow for "amortized variational inference," in which the
    variational posterior parameters are a function of the data.  We
    write the posterior as

        q(z; x, phi) approx p(z | x, theta).


    ## Requirements
    A variational posterior must support sampling and point-wise
    evaluation in order to be used for the reparameterization trick.
    """
    @ensure_variational_args_are_lists
    def __init__(self, model, datas, inputs=None, masks=None, tags=None):
        """
        Initialize the posterior with a ref to the model and datas,
        where datas is a list of data arrays.
        """
        self.model = model
        self.datas = datas

    @property
    def params(self):
        """
        Return phi.
        """
        raise NotImplemented

    def sample(self):
        """
        Return a sample from q(z; x, phi)
        """
        raise NotImplemented

    def log_density(self, sample):
        """
        Return log q(z; x, phi)
        """
        raise NotImplemented


class SLDSMeanFieldVariationalPosterior(VariationalPosterior):
    """
    Mean field variational posterior for the continuous latent
    states of an SLDS.
    """
    @ensure_variational_args_are_lists
    def __init__(self, model, datas,
                 inputs=None, masks=None, tags=None,
                 initial_variance=0.01):

        super(SLDSMeanFieldVariationalPosterior, self).\
            __init__(model, datas, masks, tags)

        # Initialize the parameters
        self.D = model.D
        self.Ts = [data.shape[0] for data in datas]
        self.initial_variance = initial_variance
        self._params = [self._initialize_variational_params(data, input, mask, tag)
                        for data, input, mask, tag in zip(datas, inputs, masks, tags)]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        assert len(value) == len(self.datas)
        for v, T in zip(value, self.Ts):
            assert len(v) == 2
            q_mu, q_sigma_inv = v
            assert q_mu.shape == q_sigma_inv.shape == (T, self.D)

        self._params = value

    @property
    def mean(self):
        return [mu for mu, _ in self.params]

    def _initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        q_mu = self.model.emissions.invert(data, input=input, mask=mask, tag=tag)
        q_sigma_inv = np.log(self.initial_variance) * np.ones((T, self.D))
        return q_mu, q_sigma_inv

    def sample(self):
        return [q_mu + np.sqrt(np.exp(q_sigma_inv)) * npr.randn(*q_mu.shape)
                for (q_mu, q_sigma_inv) in self.params]

    def log_density(self, sample):
        assert isinstance(sample, list) and len(sample) == len(self.datas)

        logq = 0
        for s, (q_mu, q_sigma_inv) in zip(sample, self.params):
            assert s.shape == q_mu.shape
            q_sigma = np.exp(q_sigma_inv)
            logq += np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
            logq += np.sum(-0.5 * (s - q_mu)**2 / q_sigma)

        return logq


class SLDSTriDiagVariationalPosterior(VariationalPosterior):
    """
    Gaussian variational posterior for the continuous latent
    states of an SLDS.  The Gaussian is constrained to have
    a block tri-diagonal inverse covariance matrix, as in a
    linear dynamical system.
    """
    @ensure_variational_args_are_lists
    def __init__(self, model, datas,
                 inputs=None, masks=None, tags=None,
                 initial_variance=0.01):

        super(SLDSTriDiagVariationalPosterior, self).\
            __init__(model, datas, masks, tags)

        # Initialize the parameters
        self.D = model.D
        self.Ts = [data.shape[0] for data in datas]
        self.initial_variance = initial_variance
        self._params = [self._initialize_variational_params(data, input, mask, tag)
                        for data, input, mask, tag in zip(datas, inputs, masks, tags)]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        D = self.D

        # Check the value for correct shape
        assert len(value) == len(self.datas)
        for v, T in zip(value, self.Ts):
            As, bs, Qi_sqrts, ms, Ri_sqrts = v
            assert As.shape == (T-1, D, D)
            assert bs.shape == (T-1, D)
            assert Qi_sqrts.shape == (T-1, D, D)
            assert ms.shape == (T, D)
            assert Ri_sqrts.shape == (T, D, D)

        self._params = value

    @property
    def mean(self):
        return [lds_mean(*prms) for prms in self.params]

    def _initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        D = self.D

        # Initialize the mean with the linear model, if applicable
        ms = self.model.emissions.invert(data, input=input, mask=mask, tag=tag)

        # Initialize with no covariance between adjacent time steps
        # NOTE: it's important to initialize A and Q to be nonzero,
        # otherwise the gradients wrt them are zero and they never
        # change during optimization!
        As = np.repeat(np.eye(D)[None, :, :], T-1, axis=0)
        bs = np.zeros((T-1, D))
        Qi_sqrts = np.repeat(np.eye(D)[None, :, :], T-1, axis=0)
        Ri_sqrts = 1./np.sqrt(self.initial_variance) * np.repeat(np.eye(D)[None, :, :], T, axis=0)
        return As, bs, Qi_sqrts, ms, Ri_sqrts

    def sample(self):
        return [lds_sample(*prms) for prms in self.params]

    def log_density(self, sample):
        assert isinstance(sample, list) and len(sample) == len(self.datas)

        logq = 0
        for s, prms in zip(sample, self.params):
            logq += lds_log_probability(s, *prms)
        return logq


class SLDSStructuredMeanFieldVariationalPosterior(VariationalPosterior):
    """
    p(z, x | y) \approx q(z) q(x).

    Assume q(x) is a Gaussian with a block tridiagonal precision matrix,
    and that we update q(x) via Laplace approximation.
    Assume q(z) is a chain-structured discrete graphical model.
    """
    @ensure_variational_args_are_lists
    def __init__(self, model, datas,
                 inputs=None, masks=None, tags=None,
                 initial_variance=1):

        super(SLDSStructuredMeanFieldVariationalPosterior, self).\
            __init__(model, datas, masks, tags)

        # Initialize the parameters
        self.D = model.D
        self.K = model.K
        self.Ts = [data.shape[0] for data in datas]
        self.initial_variance = initial_variance
        self._params = [self._initialize_variational_params(data, input, mask, tag)
                       for data, input, mask, tag in zip(datas, inputs, masks, tags)]

    def _initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        K = self.K
        D = self.D

        # Initialize q(z) parameters: log_pi0, log_likes, log_transition_matrices
        log_pi0 = -np.log(K) * np.ones(K)
        log_Ps = np.zeros((T-1, K, K))
        log_likes = np.zeros((T, K))

        # Initialize q(x) = = N(J, h) where J is block tridiagonal precision
        # and h is the linear potential.  The mapping to mean parameters is
        # mu = J^{-1} h and Sigma = J^{-1}.  Initialize J to inverse of
        # initial variance and scale h accordingly, so the mean is the output
        # of the emissions invert function.
        J_diag = np.tile(1.0 / self.initial_variance * np.eye(D)[None, :, :], (T, 1, 1))
        J_lower_diag = np.zeros((T-1, D, D))
        if self.model.emissions.single_subspace:
            h = (1.0 / self.initial_variance) \
                * self.model.emissions.invert(data, input=input, mask=mask, tag=tag)
        else:
            # TODO smarter inversion with multiple subspace!
            warn("Posterior initialization is not implemented for multiple subspaces. \
                  A random initialization is used.")
            h = (1.0 / self.initial_variance) * npr.randn(data.shape[0], self.D)

        return dict(log_pi0=log_pi0,
                    log_Ps=log_Ps,
                    log_likes=log_likes,
                    J_diag=J_diag,
                    J_lower_diag=J_lower_diag,
                    h=h)

    @property
    def params(self):
        return self._params

    def sample_discrete_states(self):
        return [hmm_sample(prms["log_pi0"], prms["log_Ps"], prms["log_likes"])
                for prms in self.params]

    def sample_continuous_states(self):
        return [block_tridiagonal_sample(prms["J_diag"], prms["J_lower_diag"], prms["h"])
                for prms in self.params]

    def sample(self):
        return list(zip(self.sample_discrete_states(), self.sample_continuous_states()))

    @property
    def mean_discrete_states(self):
        # Now compute the posterior expectations of z under q(z)
        return [hmm_expected_states(prms["log_pi0"], prms["log_Ps"], prms["log_likes"])
                for prms in self.params]

    @property
    def mean_continuous_states(self):
        # Now compute the posterior expectations of z under q(z)
        return [block_tridiagonal_mean(prms["J_diag"], prms["J_lower_diag"], prms["h"], lower=True).reshape(np.shape(prms["h"]))
                for prms in self.params]

    @property
    def mean(self):
        return list(zip(self.mean_discrete_states, self.mean_continuous_states))

    def entropy(self, sample=None):
        """
        Compute the entropy of the variational posterior distirbution.

        Recall that under the structured mean field approximation

        H[q(z)q(x)] = -E_{q(z)q(x)}[log q(z) + log q(x)]
                    = -E_q(z)[log q(z)] - E_q(x)[log q(x)]
                    = H[q(z)] + H[q(x)].

        That is, the entropy separates into the sum of entropies for the
        discrete and continuous states.

        For each one, we have

        E_q(u)[log q(u)] = E_q(u) [log q(u_1) + sum_t log q(u_t | u_{t-1}) + loq q(u_t) - log Z]
                         = E_q(u_1)[log q(u_1)] + sum_t E_{q(u_t, u_{t-1}[log q(u_t | u_{t-1})]
                             + E_q(u_t)[loq q(u_t)] - log Z

        where u \in {z, x} and log Z is the log normalizer.  This shows that we just need the
        posterior expectations and potentials, and the log normalizer of the distribution.

        Note
        ----
        We haven't implemented the exact calculations for the continuous states yet,
        so for now we're approximating the continuous state entropy via samples.
        """

        # Sample the continuous states
        if sample is None:
            sample = self.sample_continuous_states()
        else:
            assert isinstance(sample, list) and len(sample) == len(self.datas)

        negentropy = 0
        for s, prms in zip(sample, self.params):

            # 1. Compute log q(x) of samples of x
            negentropy += block_tridiagonal_log_probability(s, prms["J_diag"], prms["J_lower_diag"], prms["h"])

            # 2. Compute E_{q(z)}[ log q(z) ]
            (Ez, Ezzp1, normalizer) = hmm_expected_states(prms["log_pi0"], prms["log_Ps"], prms["log_likes"])
            negentropy -= normalizer # -log Z
            negentropy += np.sum(Ez[0] * prms["log_pi0"]) # initial factor
            negentropy += np.sum(Ez * prms["log_likes"]) # unitary factors
            negentropy += np.sum(Ezzp1 * prms["log_Ps"]) # pairwise factors

        return -negentropy
