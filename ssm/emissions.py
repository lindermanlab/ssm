import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.util import ensure_args_are_lists, ensure_args_not_none, interpolate_data


# Observation models for SLDS
class _Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace

    @property
    def params(self):
        raise NotImplementedError
        
    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        q_mu = np.zeros((T, self.D))
        q_sigma_inv = np.zeros((T, self.D))
        return q_mu, q_sigma_inv

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def sample_y(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


# Many emissions models start with a linear layer
class _LinearEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        self.Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return self.Cs, self.ds
        
    @params.setter
    def params(self, value):
        self.Cs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.ds = self.ds[perm]

    def compute_mus(self, x):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + self.ds

    def _initialize_with_pca(self, datas):
        from sklearn.decomposition import PCA
        pca = PCA(self.D).fit(np.concatenate(datas))
        self.Cs[:,...] = pca.components_.T
        self.ds[:,...] = pca.mean_
        return pca
        
    def _initialize_variational_params(self, data, input, mask, tag):
        # y = Cx + d + noise; C orthogonal.  xhat = (C^T C)^{-1} C^T (y-d)
        data = interpolate_data(data, mask)
        T = data.shape[0]
        C, d = self.Cs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T
        q_mu = (data-d).dot(C_pseudoinv)
        q_sigma_inv = -4 * np.ones((T, self.D))
        return q_mu, q_sigma_inv


# Observation models for SLDS
class GaussianEmissions(_LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(GaussianEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(GaussianEmissions, self).params + (self.inv_etas,)
        
    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(GaussianEmissions, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(GaussianEmissions, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def initialize_variational_params(self, data, input, mask, tag):
        return self._initialize_variational_params(data, input, mask, tag)
        
    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.compute_mus(x)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_y(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.compute_mus(x)
        etas = np.exp(self.inv_etas)
        return mus[:, z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)
        
    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.compute_mus(variational_mean)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)
        

class StudentsTEmissions(_LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(StudentsTEmissions, self).__init__(N, K, D, M, single_subspace=single_subspace)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones(1, N) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(StudentsTEmissions, self).params + (self.inv_etas, self.inv_nus)
        
    @params.setter
    def params(self, value):
        self.inv_etas, self.inv_nus = value[-2]
        super(StudentsTEmissions, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(StudentsTEmissions, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm] 
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def initialize_variational_params(self, data, input, mask, tag):
        return self._initialize_variational_params(data, input, mask, tag)
        
    def log_likelihoods(self, data, input, mask, tag, x):
        N, etas, nus = self.N, np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.compute_mus(x)
        
        resid = data[:, None, :] - mus
        z = resid / etas
        return -0.5 * (nus + N) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + N) / 2.0) - gammaln(nus / 2.0) - N / 2.0 * np.log(nus) \
            -N / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(etas), axis=1)
        
    def sample_y(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.compute_mus(x)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[:,z,:] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)
        
    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.compute_mus(variational_mean)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)
        

class BernoulliEmissions(_LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit"):
        super(BernoulliEmissions, self).__init__(N, K, D, M, single_subspace=single_subspace)

        mean_functions = dict(
            logit=lambda x: 1./(1+np.exp(-x))
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=lambda p: np.log(p / (1-p))
            )
        self.link = link_functions[link]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        logits = [self.link(np.clip(d, .25, .75)) for d in datas]
        self._initialize_with_pca(datas)

    def initialize_variational_params(self, data, input, mask, tag):
        logit = self.link(np.clip(data, .25, .75))
        return self._initialize_variational_params(logit, input, mask, tag)
        
    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int and data.min() >= 0 and data.max() <= 1
        ps = self.mean(self.compute_mus(x))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)
        
    def sample_y(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.compute_mus(x))
        return npr.rand(T, self.N) < ps[:,z,:]

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.compute_mus(variational_mean))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class PoissonEmissions(_LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="log"):
        super(PoissonEmissions, self).__init__(K, D, M)
        
        mean_functions = dict(
            log=lambda x: np.exp(x),
            softplus=lambda x: np.log(1 + np.exp(x))
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            log=lambda rate: np.log(rate),
            softplus=lambda rate: np.log(np.exp(rate) - 1)
            )
        self.link = link_functions[link]
    
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        logrates = [self.link(np.clip(d, .25, np.inf)) for d in datas]
        self._initialize_with_pca(datas)

    def initialize_variational_params(self, data, input, mask, tag):
        lograte = self.link(np.clip(data, .25, np.inf))
        return self._initialize_variational_params(lograte, input, mask, tag)
        
    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.compute_mus(x))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)
        
    def sample_y(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.compute_mus(x))
        return npr.poisson(lambdas[:,z,:])

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.compute_mus(variational_mean))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)


class AutoRegressiveEmissions(_LinearEmissions):
    """
    Include past observations as a covariate in the SLDS emissions.
    The AR model is restricted to be diagonal. 
    """
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(AutoRegressiveEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize AR component of the model
        self.As = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(AutoRegressiveEmissions, self).params + (self.As, self.inv_etas)
        
    @params.setter
    def params(self, value):
        self.As, self.inv_etas = value[-2]
        super(AutoRegressiveEmissions, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(AutoRegressiveEmissions, self).permute(perm)
        if not self.single_subspace:
            self.As = self.inv_nus[perm] 
            self.inv_etas = self.inv_etas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        
        # Solve a linear regression for the AR coefficients. 
        from sklearn.linear_model import LinearRegression
        for n in range(self.N):
            lr = LinearRegression()
            lr.fit(np.concatenate([d[:-1, n] for d in datas])[:,None],
                   np.concatenate([d[1:, n] for d in datas]))
            self.As[:,n] = lr.coef_[0]

        # Compute the residuals of the AR model
        mus = [np.concatenate([np.zeros(self.N), self.As[0] * d[:-1]]) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def initialize_variational_params(self, data, input, mask, tag):
        data = interpolate_data(data, mask)
        residual = np.concatenate((data[0], self.As[0] * data[:-1]))
        return self._initialize_variational_params(residual, input, mask, tag)
        
    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.compute_mus(x)
        mus[1:] = mus[1:] + self.As[None, :, :] * data[:-1, None, :]
        
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_y(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.compute_mus(x)
        etas = np.exp(self.inv_etas)

        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + np.sqrt(etas[z[0]]) * npr.randn(N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.compute_mus(variational_mean)
        mus[1:] += self.As[None, :, :] * data[:, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)
        
