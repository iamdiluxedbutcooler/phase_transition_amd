"""
Statistical testing utilities for AMD experiments.
Includes vectorized permutation tests, partial correlation,
Pearson correlation with p-values, and bootstrap confidence intervals.
"""

import numpy as np
from scipy import stats


def pearson_with_pvalue(x, y):
    """
    Compute Pearson correlation coefficient and two-sided p-value.

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.

    Returns
    -------
    tuple
        (r, p_value). Returns (np.nan, np.nan) if fewer than 3 valid pairs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    r, p = stats.pearsonr(x, y)
    return r, p


def partial_correlation(x, y, covariates):
    """
    Compute partial Pearson correlation controlling for covariates.

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    covariates : numpy.ndarray
        Matrix of shape (n_samples, n_covariates) or 1-d array for single covariate.

    Returns
    -------
    tuple
        (partial_r, p_value).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))

    if covariates.shape[0] == 1 and covariates.shape[1] != 1:
        covariates = covariates.T

    valid = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(covariates), axis=1))
    x = x[valid]
    y = y[valid]
    covariates = covariates[valid]

    if len(x) < covariates.shape[1] + 3:
        return np.nan, np.nan

    cov_with_intercept = np.column_stack([np.ones(len(x)), covariates])

    pinv = np.linalg.pinv(cov_with_intercept)
    residual_x = x - cov_with_intercept @ (pinv @ x)
    residual_y = y - cov_with_intercept @ (pinv @ y)

    r, p = stats.pearsonr(residual_x, residual_y)
    return r, p


def vectorized_permutation_test(group1, group2, n_permutations=10000, seed=42):
    """
    Vectorized two-sample permutation test on difference of means.

    Parameters
    ----------
    group1 : array-like
        Observations from first group.
    group2 : array-like
        Observations from second group.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'observed_diff', 'p_value'.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) == 0 or len(g2) == 0:
        return {"observed_diff": np.nan, "p_value": np.nan}

    observed_diff = np.mean(g1) - np.mean(g2)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    n_total = len(combined)

    rng = np.random.RandomState(seed)
    perm_indices = np.array([rng.permutation(n_total) for _ in range(n_permutations)])

    perm_means_1 = np.mean(combined[perm_indices[:, :n1]], axis=1)
    perm_means_2 = np.mean(combined[perm_indices[:, n1:]], axis=1)
    null_diffs = perm_means_1 - perm_means_2

    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))

    return {"observed_diff": observed_diff, "p_value": p_value}


def bootstrap_statistic(data_func, n_reps=100, seed=42):
    """
    Bootstrap a statistic by repeated calls to a sampling function.

    Parameters
    ----------
    data_func : callable
        Function that takes a numpy RandomState and returns a scalar.
    n_reps : int
        Number of bootstrap repetitions.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'mean', 'std', 'values'.
    """
    rng = np.random.RandomState(seed)
    values = np.array([data_func(rng) for _ in range(n_reps)])
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "values": values,
    }
