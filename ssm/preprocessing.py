import autograd.numpy as np
from sklearn.decomposition import PCA

def pca_with_imputation(D, datas, masks, num_iters=20):
    if isinstance(datas, (list, tuple)) and isinstance(masks, (list, tuple)):
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
    else:
        pca = PCA(D)
        x = pca.fit_transform(data)
        
    # Unpack xs
    xs = np.split(x, np.cumsum([len(data) for data in datas])[:-1])
    assert len(xs) == len(datas)
    assert all([x.shape[0] == data.shape[0] for x, data in zip(xs, datas)])

    return pca, xs


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


def trend_filter(data):
    """
    Subtract a linear trend from the data
    """
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    T = data.shape[0]
    lr.fit(np.arange(T)[:, None], data)
    trend = lr.predict(np.arange(T)[:, None])
    return data - trend

def standardize(data, mask): 
    data2 = data.copy()
    data2[~mask] = np.nan
    m = np.nanmean(data2, axis=0)
    s = np.nanstd(data2, axis=0)
    s[~np.any(mask, axis=0)] = 1
    y = (data - m) / s
    assert np.all(np.isfinite(y))
    return y

