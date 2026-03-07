"""
Critical slowing down indicators for conversational time series.
Implements rolling-window lag-1 autocorrelation, variance, and
Kendall tau trend detection with permutation-based significance testing.
"""

import numpy as np
from scipy import stats


def rolling_lag1_autocorrelation(series, window_size=5):
    """
    Compute rolling-window lag-1 autocorrelation of a time series.

    Parameters
    ----------
    series : array-like
        Input time series.
    window_size : int
        Size of rolling window.

    Returns
    -------
    numpy.ndarray
        Lag-1 autocorrelation values. Length is len(series) - window_size + 1.
        NaN where computation is undefined.
    """
    series = np.asarray(series, dtype=float)
    n_output = len(series) - window_size + 1

    if n_output <= 0:
        return np.array([])

    result = np.full(n_output, np.nan)

    for i in range(n_output):
        window = series[i:i + window_size]
        if len(window) < 3:
            continue
        x = window[:-1]
        y = window[1:]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        denom = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
        if denom == 0:
            result[i] = 0.0
        else:
            result[i] = np.sum((x - mean_x) * (y - mean_y)) / denom

    return result


def rolling_variance(series, window_size=5):
    """
    Compute rolling-window variance of a time series.

    Parameters
    ----------
    series : array-like
        Input time series.
    window_size : int
        Size of rolling window.

    Returns
    -------
    numpy.ndarray
        Variance values. Length is len(series) - window_size + 1.
    """
    series = np.asarray(series, dtype=float)
    n_output = len(series) - window_size + 1

    if n_output <= 0:
        return np.array([])

    result = np.full(n_output, np.nan)

    for i in range(n_output):
        window = series[i:i + window_size]
        result[i] = np.var(window, ddof=1) if len(window) > 1 else 0.0

    return result


def kendall_tau_trend(series):
    """
    Compute Kendall tau correlation between a series and its time index.

    Parameters
    ----------
    series : array-like
        Input time series.

    Returns
    -------
    tuple
        (tau, p_value). Returns (np.nan, np.nan) if series has fewer than 3 elements.
    """
    series = np.asarray(series, dtype=float)
    valid = ~np.isnan(series)
    series = series[valid]

    if len(series) < 3:
        return np.nan, np.nan

    time_index = np.arange(len(series))
    tau, p_value = stats.kendalltau(time_index, series)
    return tau, p_value


def compute_csd_indicators(series, window_size=5, pre_breakdown_length=5):
    """
    Compute full set of critical slowing down indicators for a time series.

    Parameters
    ----------
    series : array-like
        Full time series for a conversation.
    window_size : int
        Rolling window size for AC1 and variance computation.
    pre_breakdown_length : int
        Number of final values to use for trend assessment.

    Returns
    -------
    dict
        Keys: 'ac1_series', 'var_series', 'ac1_tau', 'ac1_tau_p',
              'var_tau', 'var_tau_p'.
    """
    ac1 = rolling_lag1_autocorrelation(series, window_size)
    var = rolling_variance(series, window_size)

    ac1_tail = ac1[-pre_breakdown_length:] if len(ac1) >= pre_breakdown_length else ac1
    var_tail = var[-pre_breakdown_length:] if len(var) >= pre_breakdown_length else var

    ac1_tau, ac1_tau_p = kendall_tau_trend(ac1_tail)
    var_tau, var_tau_p = kendall_tau_trend(var_tail)

    return {
        "ac1_series": ac1,
        "var_series": var,
        "ac1_tau": ac1_tau,
        "ac1_tau_p": ac1_tau_p,
        "var_tau": var_tau,
        "var_tau_p": var_tau_p,
    }


def permutation_test_two_groups(group1, group2, n_permutations=10000, seed=42):
    """
    Vectorized permutation test comparing means of two groups.

    Parameters
    ----------
    group1 : array-like
        First group of observations.
    group2 : array-like
        Second group of observations.
    n_permutations : int
        Number of random permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: 'observed_diff', 'p_value', 'null_distribution'.
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    valid1 = group1[~np.isnan(group1)]
    valid2 = group2[~np.isnan(group2)]

    if len(valid1) == 0 or len(valid2) == 0:
        return {"observed_diff": np.nan, "p_value": np.nan, "null_distribution": np.array([])}

    observed_diff = np.mean(valid1) - np.mean(valid2)
    combined = np.concatenate([valid1, valid2])
    n1 = len(valid1)

    rng = np.random.RandomState(seed)
    indices = np.array([rng.permutation(len(combined)) for _ in range(n_permutations)])

    perm_group1_means = np.mean(combined[indices[:, :n1]], axis=1)
    perm_group2_means = np.mean(combined[indices[:, n1:]], axis=1)
    null_distribution = perm_group1_means - perm_group2_means

    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "null_distribution": null_distribution,
    }
