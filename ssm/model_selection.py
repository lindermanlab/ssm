"""
Cross-validation and related model diagnostics.
"""
import autograd.numpy as np
import autograd.numpy.random as npr


def cross_val_scores(
        model, data, inputs=None, masks=None, heldout_frac=0.1,
        n_repeats=3, normalize_scores=True):
    """
    Evaluate HMM log-likelihood scores on heldout data.

    Parameters
    ----------
    model : HMM
        Model instance.
    data : ndarray
        Data matrix with shape (n_obs, n_out).
    inputs : ndarray
        If applicable, matrix holding input variables with shape
        (n_obs, n_in).
    masks : ndarray
        If applicable, binary matrix specifying censored or
        unobserved entries in data. Entries of one correspond
        to observed data. Has shape (n_obs, n_out).
    heldout_frac : float
        Number between zero and one specifying how much data
        to hold out on each cross-validation run.
    n_repeats : int
        Number of randomized cross-validation runs to perform.
    normalize_scores : bool
        If True, normalize log-likelihood scores per observation.
        Defaults to True.

    Returns
    -------
    test_scores : ndarray
        Array holding log-likelihood scores on test set.
        Has shape (n_repeats,).
    train_scores : ndarray
        Array holding log-likelihood scores on training set.
        Has shape (n_repeats,).
    """

    if masks is None:
        masks = np.ones_like(data, dtype=bool)
    else:
        masks = np.asarray(masks, dtype=bool)

    total_obs = np.sum(masks)
    obs_ind = np.argwhere(masks)

    heldout_num = int(total_obs * heldout_frac)

    test_scores = np.empty(n_repeats)
    train_scores = np.empty(n_repeats)

    for r in range(n_repeats):

        # Randomly choose indices to hold out.
        i = npr.choice(total_obs, heldout_num, replace=False)
        heldout_ind = obs_ind[i]

        # Create mask for training data.
        train_mask = np.copy(masks)
        train_mask[heldout_ind] = False

        # Fit model.
        model.fit(data, inputs=inputs, masks=train_mask)

        # Compute expectaations.
        expectations = [model.expected_states(x, inp, m, tg)
                        for x, inp, m, tg
                        in zip(data, inputs, train_mask, tags)]

        # Evaluate loss on training data.
        train_scores[r] = model.log_likelihood(
            data, inputs=inputs, masks=train_mask)

        # Compute log-likelihood on test data.
        test_mask = ~train_mask & masks
        test_scores[r] = model.log_likelihood(
            data, inputs=inputs, masks=test_mask)

    # Normalize log-likelihoods per observation.
    if normalize_scores:
        train_scores /= train_mask.sum()
        test_scores /= test_mask.sum()

    return test_scores, train_scores
