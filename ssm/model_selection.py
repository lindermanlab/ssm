"""
Cross-validation and related model diagnostics.
"""
import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.util import ensure_args_are_lists


@ensure_args_are_lists
def cross_val_scores(
        model, datas, inputs=None, masks=None, tags=None,
        heldout_frac=0.1, n_repeats=3, **fit_kw):
    """
    Evaluate HMM log-likelihood scores on heldout data using
    a speckled holdout pattern.

    Parameters
    ----------
    model : HMM
        Model instance.
    datas : ndarray or list of ndarray
        Data matrices with shape (n_obs, n_out).
    inputs : ndarray
        If applicable, matrix holding input variables with shape
        (n_obs, n_in).
    masks : ndarray
        If applicable, binary matrix specifying censored or
        unobserved entries in data. Entries of one correspond
        to observed data. Has shape (n_obs, n_out).
    tags : ???
        ???
    heldout_frac : float
        Number between zero and one specifying how much data
        to hold out on each cross-validation run.
    n_repeats : int
        Number of randomized cross-validation runs to perform.
    **fit_kw : dict
        Additional keyword arguments are passed to model.fit(...)

    Returns
    -------
    test_scores : ndarray
        Array holding normalized log-likelihood scores on test
        set. Has shape (n_repeats,).
    train_scores : ndarray
        Array holding log-likelihood scores on training set.
        Has shape (n_repeats,).
    """

    # Allocate space for train and test log-likelihoods.
    test_scores = np.empty(n_repeats)
    train_scores = np.empty(n_repeats)

    for r in range(n_repeats):

        # Create mask for training data.
        train_masks = []
        for k, m in enumerate(masks):

            # Determine number of heldout points.
            total_obs = np.sum(m)
            obs_inds = np.argwhere(m)
            heldout_num = int(total_obs * heldout_frac)

            # Randomly hold out speckled data pattern.
            heldout_flat_inds = npr.choice(
                total_obs, heldout_num, replace=False)

            # Create training mask.
            train_masks.append(m.copy())
            i, j = obs_inds[heldout_flat_inds].T
            train_masks[-1][i, j] = False

        # Fit model with training mask.
        model.fit(datas, inputs=inputs, tags=tags, masks=train_masks, **fit_kw)

        train_ll = model.log_likelihood(
            datas, inputs=inputs, tags=tags, masks=train_masks)

        # Compute log-likelihood without training mask.
        full_ll = model.log_likelihood(
            datas, inputs=inputs, tags=tags, masks=masks)

        # Total number of training and observed datapoints.
        n_observed = np.sum([m.sum() for m in masks])
        n_train = np.sum([m.sum() for m in train_masks])

        # Save normalized log-likelihood scores.
        test_scores[r] = (full_ll - train_ll) / (n_observed - n_train)
        train_scores[r] = train_ll / n_train

    return test_scores, train_scores
