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

# Observation models for SLDS
class GaussianEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(GaussianEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize observation model
        self.Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return self.Cs, self.ds, self.inv_etas
        
    @params.setter
    def params(self, value):
        self.Cs, self.ds, self.inv_etas = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.ds = self.ds[perm]
            # self.Vs = self.Vs[perm]
            self.inv_eta = self.inv_etas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        data = np.concatenate(datas)
        
        pca = PCA(self.D)
        x = pca.fit_transform(data)
        resid = data - pca.inverse_transform(x)
        etas = np.var(resid, axis=0)

        self.Cs[:,...] = pca.components_.T
        self.ds[:,...] = pca.mean_
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def initialize_variational_params(self, data, input, mask, tag):
        # y = Cx + d + noise; C orthogonal.  xhat = (C^T C)^{-1} C^T (y-d)
        data = interpolate_data(data, mask)
        T = data.shape[0]
        C, d = self.Cs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T
        q_mu = (data-d).dot(C_pseudoinv)
        q_sigma_inv = -4 * np.ones((T, self.D))
        return q_mu, q_sigma_inv
        
    def log_likelihoods(self, data, input, mask, tag, x):
        Cs, ds = self.Cs, self.ds
        mus = np.matmul(Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + ds
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_y(self, z, x, input=None, tag=None):
        T = z.shape[0]
        Cs, ds = self.Cs, self.ds
        mus = np.matmul(Cs[None, ...], x[:, None, :, None])[:, :, :, 0] + ds
        etas = np.exp(self.inv_etas)

        mu = mus[:,0,:] if self.single_subspace else mus[:, z, :] 
        eta = etas[0] if self.single_subspace else etas[z]
        y = mu + np.sqrt(eta) * npr.randn(T, self.N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Cs, ds = self.Cs, self.ds
        mus = np.matmul(Cs[None, ...], variational_mean[:, None, :, None])[:, :, :, 0] + ds
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)
        
        
