import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.util import ensure_args_are_lists, ensure_args_not_none, logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation


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
        # Use the rational Cayley transform to parameterize an orthogonal emission matrix
        assert N > D
        self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def Cs(self):
        # See https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
        # for a derivation of the rational Cayley transform.
        D = self.D
        T = lambda X: np.swapaxes(X, -1, -2)
        
        Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
        Fs = np.matmul(T(self._As), self._As) - Bs
        trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
        trm2 = np.eye(D) + Fs
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(
            np.matmul(T(Cs), Cs), 
            np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
            )
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.N, self.D
        T = lambda X: np.swapaxes(X, -1, -2)
        
        # Make sure value is the right shape and orthogonal
        Keff = 1 if self.single_subspace else self.K
        assert value.shape == (Keff, N, D)
        assert np.allclose(
            np.matmul(T(value), value), 
            np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
            )

        Q1s, Q2s = value[:, :D, :], value[:, D:, :]
        Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._Ms = T(Fs)
        self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
        assert np.allclose(self.Cs, value)

    @property
    def params(self):
        return self._As, self._Ms, self.ds
        
    @params.setter
    def params(self, value):
        self._As, self._Ms, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self._As = self._As[perm]
            self._Ms = self._Bs[perm]
            self.ds = self.ds[perm]

    def compute_mus(self, x):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + self.ds

    def _initialize_with_pca(self, datas, masks, num_iters=25):
        pca, xs = pca_with_imputation(self.D, datas, masks, num_iters=num_iters)
        Keff = 1 if self.single_subspace else self.K
        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))
            
        return pca
        
    def _initialize_variational_params(self, data, input, mask, tag):
        # y = Cx + d + noise; C orthogonal.  xhat = (C^T C)^{-1} C^T (y-d)
        data = interpolate_data(data, mask)
        T = data.shape[0]
        
        C, d = self.Cs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T
        
        # We would like to find the PCA coordinates in the face of missing data
        # To do so, alternate between running PCA and imputing the missing entries
        for itr in range(25):
            q_mu = (data-d).dot(C_pseudoinv)
            data[:, ~mask[0]] = (q_mu.dot(C.T) + d)[:, ~mask[0]]

        # Set a low posterior variance
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
        pca = self._initialize_with_pca(datas, masks)
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
        return mus[np.arange(T), z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)
        
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
        super(StudentsTEmissions, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(StudentsTEmissions, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm] 
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, masks)
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
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]
        
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        logits = [self.link(np.clip(d, .25, .75)) for d in datas]
        self._initialize_with_pca(datas, masks)

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
            softplus=softplus
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            log=lambda rate: np.log(rate),
            softplus=inv_softplus
            )
        self.link = link_functions[link]
    
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        logrates = [self.link(np.clip(d, .25, np.inf)) for d in datas]
        self._initialize_with_pca(datas, masks)

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
        self.As, self.inv_etas = value[-2:]
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
        pad = np.zeros((1,self.N))
        mus = [np.concatenate((pad, self.As[0] * d[:-1])) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals, masks)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def initialize_variational_params(self, data, input, mask, tag):
        data = interpolate_data(data, mask)
        mu = np.concatenate((np.zeros((1, self.N)), self.As[0] * data[:-1]))
        residual = data - mu
        return self._initialize_variational_params(residual, input, mask, tag)
        
    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.compute_mus(x)
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        mus = mus + np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :])) 
        
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
        mus[1:] += self.As[None, :, :] * data[:-1, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)
        

# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,)):
        super(_NeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D,) + hidden_layer_sizes + (N,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases
        
    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def compute_mus(self, x):
        inputs = x
        for W, b in zip(self.weights, self.biases):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs
    
    def initialize_variational_params(self, data, input, mask, tag):
        T = data.shape[0]
        q_mu = npr.randn(T, self.D)
        q_sigma_inv = np.zeros((T, self.D))
        return q_mu, q_sigma_inv


# Observation models for SLDS
class GaussianNeuralNetworkEmissions(_NeuralNetworkEmissions):
    def __init__(self, N, K, D, M=0):
        super(GaussianNeuralNetworkEmissions, self).__init__(N, K, D, M=M)
        self.inv_etas = -4 + npr.randn(N)

    @property
    def params(self):
        return super(GaussianNeuralNetworkEmissions, self).params + (self.inv_etas,)
        
    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(GaussianNeuralNetworkEmissions, self.__class__).params.fset(self, value[:-1])

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.compute_mus(x)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data - mus)**2 / etas
        return np.sum(lls * mask, axis=1)[:, None]

    def sample_y(self, z, x, input=None, tag=None):
        mus = self.compute_mus(x)
        etas = np.exp(self.inv_etas)
        return mus + np.sqrt(etas) * npr.randn(*mus.shape)
        
    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.compute_mus(variational_mean)
        return mus
        