import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd.scipy.linalg import block_diag

from sklearn.decomposition import PCA

from ssm.emissions import Emissions, _GaussianEmissionsMixin, _PoissonEmissionsMixin, \
    _LinearEmissions, _OrthogonalLinearEmissions, _NeuralNetworkEmissions, _BernoulliEmissionsMixin
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation


class _CompoundLinearEmissions(Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_LinearEmissions(n, K, d, M=M) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return tuple(em.params for em in self.emissions_models)

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca


class _CompoundOrthogonalLinearEmissions(Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundOrthogonalLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_OrthogonalLinearEmissions(n, K, d, M=M) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return tuple(em.params for em in self.emissions_models)
        # return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca


# Allow general nonlinear emission models with neural networks
class _CompoundNeuralNetworkEmissions(Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True, N_vec=None, D_vec=None, **kwargs):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_CompoundNeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)


        print(hidden_layer_sizes)
        #Make sure N_vec and D_vec are in correct form
        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_NeuralNetworkEmissions(n, K, d, hidden_layer_sizes=hidden_layer_sizes) for n, d in zip(N_vec, D_vec)]

    @property
    def params(self):
        return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def log_prior(self):
        alpha=1
        ssq_all=[]
        for em in self.emissions_models:
            ssq_w=[np.sum(i**2) for i in em.weights]
            ssq_all.append(-np.sum(alpha*ssq_w))
        return np.sum(ssq_all)


class GaussianCompoundEmissions(_GaussianEmissionsMixin, _CompoundLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = -4#np.log(pca.noise_variance_)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        assert self.single_subspace, "Only implemented for a single emission model"
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
        return -1 * np.tile(hess[None,:,:], (T, 1, 1))


class GaussianOrthogonalCompoundEmissions(_GaussianEmissionsMixin, _CompoundOrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        assert self.single_subspace, "Only implemented for a single emission model"
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
        return -1 * np.tile(hess[None,:,:], (T, 1, 1))

class GaussianCompoundNeuralNetworkEmissions(_GaussianEmissionsMixin, _CompoundNeuralNetworkEmissions):
    pass


class PoissonOrthogonalCompoundEmissions(_PoissonEmissionsMixin, _CompoundOrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.link_name == "log":
            assert self.single_subspace
            lambdas = self.mean(self.forward(x, input, tag))
            return -np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])


        elif self.link_name == "softplus":
            assert self.single_subspace
            lambdas = np.log1p(np.exp(np.dot(x,self.Cs[0].T)+np.dot(input,self.Fs[0].T)+self.ds[0]))
            expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
            diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms) / (1.0+expterms)**2
            return -np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])


class PoissonCompoundEmissions(_PoissonEmissionsMixin, _CompoundLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.link_name == "log":
            assert self.single_subspace
            lambdas = self.mean(self.forward(x, input, tag))
            return -np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])

        elif self.link_name == "softplus":
            assert self.single_subspace
            lambdas = np.log1p(np.exp(np.dot(x,self.Cs[0].T)+np.dot(input,self.Fs[0].T)+self.ds[0]))
            expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
            diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms) / (1.0+expterms)**2
            return -np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])


class PoissonCompoundNeuralNetworkEmissions(_PoissonEmissionsMixin, _CompoundNeuralNetworkEmissions):
    pass


class _NormalizedLinearEmissions(Emissions):
    """
    A simple linear mapping from continuous states x to data y.

        E[y | x] = Cx + d + Fu

    where C is an emission matrix, d is a bias, F an input matrix,
    and u is an input.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_NormalizedLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer.  Set _Cs to be private so that it can be
        # changed in subclasses.
        self._Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def Cs(self):
        ssq=np.sum(self._Cs**2,axis=1)
        # print(ssq)
        tmp = self._Cs/np.sqrt(ssq)
        # print(np.sum(tmp**2,axis=1))
        return tmp

    @Cs.setter
    def Cs(self, value):
        K, N, D = self.K, self.N, self.D
        assert value.shape == (1, N, D) if self.single_subspace else (K, N, D)
        self._Cs = value

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input, mask, tag):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        assert self.single_subspace, "Can only invert with a single emission model"

        T = data.shape[0]
        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
             + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
             + self.ds

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data
        pca, xs, ll = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)

        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))

        return pca


class _CompoundNormalizedLinearEmissions(Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundNormalizedLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_NormalizedLinearEmissions(n, K, d) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return tuple(em.params for em in self.emissions_models)
        # return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca


class GaussianNormalizedEmissions(_GaussianEmissionsMixin, _NormalizedLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass


class GaussianNormalizedCompoundEmissions(_GaussianEmissionsMixin, _CompoundNormalizedLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass


class BernoulliOrthogonalCompoundEmissions(_BernoulliEmissionsMixin, _CompoundOrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)


class BernoulliCompoundEmissions(_BernoulliEmissionsMixin, _CompoundLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)
