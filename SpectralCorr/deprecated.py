"""
Deprecated functions for backward compatibility.

This module contains functions that have been deprecated in favor of newer
implementations with improved APIs. These functions are maintained for
backward compatibility but will issue deprecation warnings when used.
"""

import warnings
import numpy as np
import xarray as xr
from .timeseries import TimeSeries
from .correlations import _correlation_pearson


def cross_correlation_maxima(ts1, ts2, maxlags=None):
    """
    DEPRECATED: Use maximum_cross_correlation() instead.

    Find the maximum cross-correlation and its lag between two time series.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series objects
    maxlags : int, optional
        Maximum lag to search. If None, uses N//4 where N is series length

    Returns
    -------
    lag_max : float
        Lag at which maximum correlation occurs
    ccf_max : float
        Maximum cross-correlation value
    """
    warnings.warn(
        "cross_correlation_maxima() is deprecated and will be removed in a future version. "
        "Use maximum_cross_correlation() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from .correlations import maximum_cross_correlation

    # Convert TimeSeries to xarray for the new function if needed
    if isinstance(ts1, TimeSeries):
        ts1_xr = ts1.to_xarray()
    else:
        ts1_xr = ts1

    if isinstance(ts2, TimeSeries):
        ts2_xr = ts2.to_xarray()
    else:
        ts2_xr = ts2

    return maximum_cross_correlation(ts1_xr, ts2_xr, maxlags=maxlags)


def shift_maximally_correlated(ts1, ts2, maxlags=None):
    """
    DEPRECATED: Use align_at_maximum_correlation() instead.

    Shift two time series to align them at maximum cross-correlation.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series to align
    maxlags : int, optional
        Maximum lag to search for alignment

    Returns
    -------
    ts1_aligned, ts2_aligned : TimeSeries
        Aligned time series objects
    """
    warnings.warn(
        "shift_maximally_correlated() is deprecated and will be removed in a future version. "
        "Use align_at_maximum_correlation() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from .correlations import align_at_maximum_correlation

    # Convert TimeSeries to xarray for the new function if needed
    if isinstance(ts1, TimeSeries):
        ts1_xr = ts1.to_xarray()
    else:
        ts1_xr = ts1

    if isinstance(ts2, TimeSeries):
        ts2_xr = ts2.to_xarray()
    else:
        ts2_xr = ts2

    # Get aligned xarray results
    ts1_aligned_xr, ts2_aligned_xr = align_at_maximum_correlation(ts1_xr, ts2_xr, maxlags=maxlags)

    # Convert back to TimeSeries for backward compatibility
    ts1_aligned = TimeSeries.from_xarray(ts1_aligned_xr)
    ts2_aligned = TimeSeries.from_xarray(ts2_aligned_xr)

    return ts1_aligned, ts2_aligned


def bootstrapped_cross_correlation(ts1, ts2, maxlags=None, n_iter=1000,
                                 return_distributions=False, detrend=False, seed=None):
    """
    DEPRECATED: Use cross_correlation(method='ebisuzaki') instead.

    Compute bootstrap cross-correlation using phase scrambling.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series for cross-correlation analysis
    maxlags : int, optional
        Maximum lag for cross-correlation. Default is N//4
    n_iter : int, optional
        Number of bootstrap iterations. Default is 1000
    return_distributions : bool, optional
        Whether to return full bootstrap distributions. Default is False
    detrend : bool, optional
        Whether to detrend the time series before analysis. Default is False
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    xarray.Dataset
        Dataset containing cross-correlation results and confidence intervals
    """
    warnings.warn(
        "bootstrapped_cross_correlation() is deprecated and will be removed in a future version. "
        "Use cross_correlation(method='ebisuzaki') instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from .correlations import cross_correlation

    # Convert TimeSeries to xarray for the new function if needed
    if isinstance(ts1, TimeSeries):
        ts1_xr = ts1.to_xarray()
    else:
        ts1_xr = ts1

    if isinstance(ts2, TimeSeries):
        ts2_xr = ts2.to_xarray()
    else:
        ts2_xr = ts2

    return cross_correlation(
        ts1_xr, ts2_xr,
        maxlags=maxlags,
        method='ebisuzaki',
        n_iter=n_iter,
        return_distributions=return_distributions,
        detrend=detrend,
        seed=seed
    )