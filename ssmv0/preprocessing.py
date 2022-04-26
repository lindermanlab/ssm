from tqdm.auto import trange
import autograd.numpy as np
from sklearn.decomposition import PCA

def pca_with_imputation(D, datas, masks, num_iters=20):
    datas = [datas] if not isinstance(datas, (list, tuple)) else datas
    if masks is not None:
        masks = [masks] if not isinstance(masks, (list, tuple)) else masks
        assert np.all([m.shape == d.shape for d, m in zip(datas, masks)])
    else:
        masks = [np.ones_like(data, dtype=bool) for data in datas]

    # Flatten the data and masks
    data = np.concatenate(datas)
    mask = np.concatenate(masks)

    if np.any(~mask):
        # Fill in missing data with mean to start
        fulldata = data.copy()
        for n in range(fulldata.shape[1]):
            fulldata[~mask[:,n], n] = fulldata[mask[:,n], n].mean()

        for itr in range(num_iters):
            # Run PCA on imputed data
            pca = PCA(D)
            x = pca.fit_transform(fulldata)

            # Fill in missing data with PCA predictions
            pred = pca.inverse_transform(x)
            fulldata[~mask] = pred[~mask]

        ll = pca.score(fulldata)

    else:
        pca = PCA(D)
        x = pca.fit_transform(data)
        ll = pca.score(data)

    # Unpack xs
    xs = np.split(x, np.cumsum([len(data) for data in datas])[:-1])
    assert len(xs) == len(datas)
    assert all([x.shape[0] == data.shape[0] for x, data in zip(xs, datas)])

    return pca, xs, ll


def factor_analysis_with_imputation(D, datas, masks=None, num_iters=50):
    datas = [datas] if not isinstance(datas, (list, tuple)) else datas
    if masks is not None:
        masks = [masks] if not isinstance(masks, (list, tuple)) else masks
        assert np.all([m.shape == d.shape for d, m in zip(datas, masks)])
    else:
        masks = [np.ones_like(data, dtype=bool) for data in datas]
    N = datas[0].shape[1]

    # Make the factor analysis model
    from pybasicbayes.models import FactorAnalysis
    fa = FactorAnalysis(N, D, alpha_0=1e-3, beta_0=1e-3)
    fa.regression.sigmasq_flat = np.ones(N)
    for data, mask in zip(datas, masks):
        fa.add_data(data, mask=mask)
    fa.set_empirical_mean()

    # Fit with EM
    lls = [fa.log_likelihood()]
    pbar = trange(num_iters)
    pbar.set_description("Itr {} LP: {:.1f}".format(0, lls[-1]))
    for itr in pbar:
        fa.EM_step()
        lls.append(fa.log_likelihood())

        pbar.set_description("Itr {} LP: {:.1f}".format(itr, lls[-1]))
        pbar.update(1)
    lls = np.array(lls)

    # Get the continuous states and their covariances
    E_xs = [states.E_Z for states in fa.data_list]
    E_xxTs = [states.E_ZZT for states in fa.data_list]
    Cov_xs = [E_xxT - E_x[:, :, None] * E_x[:, None, :] for E_x, E_xxT in zip(E_xs, E_xxTs)]

    # Rotate the states with SVD so that the columns of the
    # emission matrix, C, are orthogonal and sorted in order
    # of decreasing explained variance.
    #
    # Note: the columns of C are not normalized like in PCA!
    # This is because factor analysis assumes the latents are
    # distributed according to a standard normal distribution.
    # The FA latents are only invariant to *orthogonal* transforms.
    # Thus, the scaling must be accounted for in C.
    C, S, VT = np.linalg.svd(fa.W, full_matrices=False)
    xhats = [x.dot(VT.T) for x in E_xs]
    Cov_xhats = [np.matmul(np.matmul(VT[None, :, :], Cov_x), VT.T[None, :, :]) for Cov_x in Cov_xs]

    # Test that we got this right
    for x, xhat in zip(E_xs, xhats):
        y = x.dot(fa.W.T) + fa.mean
        yhat = xhat.dot((C * S).T) + fa.mean
        assert np.allclose(y, yhat)

    # Strip out the data from the factor analysis model,
    # update the emission matrix
    fa.regression.A = C * S
    fa.data_list = []

    return fa, xhats, Cov_xhats, lls


def interpolate_data(data, mask):
    """
    Interpolate over missing entries
    """
    assert data.shape == mask.shape and mask.dtype == bool
    T, N = data.shape
    interp_data = data.copy()
    if np.any(~mask):
        for n in range(N):
            if np.sum(mask[:,n]) >= 2:
                t_missing = np.arange(T)[~mask[:,n]]
                t_given = np.arange(T)[mask[:,n]]
                y_given = data[mask[:,n], n]
                interp_data[~mask[:,n], n] = np.interp(t_missing, t_given, y_given)
            else:
                # Can't do much if we don't see anything... just set it to zero
                interp_data[~mask[:,n], n] = 0
    return interp_data


def trend_filter(data, npoly=1, nexp=0):
    """
    Subtract a linear trend from the data
    """
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression(fit_intercept=True)
    T = data.shape[0]
    t = np.arange(T)

    # Create feature matrix
    features = np.zeros((T, npoly + nexp))

    # Polynomial of given order (npoly)
    for k in range(npoly):
        features[:, k] = t**(k+1)

    # Exponential functions (logarithmically spaced)
    for k in range(nexp):
        tau = T / (k+1)
        features[:, npoly+k] = np.exp(-t / tau)

    lr.fit(features, data)
    trend = lr.predict(features)
    return data - trend


def standardize(data, mask):
    data2 = data.copy()
    data2[~mask] = np.nan
    m = np.nanmean(data2, axis=0)
    s = np.nanstd(data2, axis=0)
    s[~np.any(mask, axis=0)] = 1
    y = (data - m) / s
    y[~mask] = 0
    assert np.all(np.isfinite(y))
    return y

