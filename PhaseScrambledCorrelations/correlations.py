"""
Correlation and phase scrambling utilities for time series analysis.
"""

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from statsmodels.distributions.empirical_distribution import ECDF
from .fft_bootstrap import *
from .timeseries import TimeSeries
from tqdm import trange

def cross_correlation(ts1: TimeSeries, ts2: TimeSeries, maxlags=None):
    """
    Compute cross-correlation and p-values between two time series.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series.
    maxlags : int, optional
        Maximum lag to compute.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            - lag: lag values
            - cross_correlation: cross-correlation coefficients
            - cross_correlation_pvalue: p-values for each lag
    """
    x = np.asarray(ts1.data)
    y = np.asarray(ts2.data)
    if ts1.dt != ts2.dt:
        raise ValueError(f"TimeSeries dt mismatch: {ts1.dt} vs {ts2.dt}")
    dt = ts1.dt
    n = x.size
    if y.size != n:
        raise ValueError("Input vectors must have the same length")
    if maxlags is None:
        maxlags = n - 1
    maxlags = min(maxlags, n - 1)
    lags = np.arange(-maxlags, maxlags+1) * dt
    ccf = np.zeros(len(lags))
    pvals = np.zeros(len(lags))
    for i, k in enumerate(range(-maxlags, maxlags+1)):
        if k < 0:
            xi = x[:n+k]
            yi = y[-k:]
        else:
            xi = x[k:]
            yi = y[:n-k]
        ccf[i] = pearsonr(xi, yi)[0]
        pvals[i] = pearsonr(xi, yi)[1]
    ccf_da = xr.DataArray(
        ccf,
        coords={"lag": lags},
        dims=["lag"],
        name="cross_correlation",
        attrs={"description": "Cross-correlation coefficient at each lag"}
    )
    pvals_da = xr.DataArray(
        pvals,
        coords={"lag": lags},
        dims=["lag"],
        name="cross_correlation_pvalue",
        attrs={"description": "P-value for cross-correlation at each lag"}
    )
    lag_da = xr.DataArray(
        lags,
        dims=["lag"],
        name="lag",
        attrs={"description": "Lag values"}
    )
    ds = xr.Dataset(
        {
            "lag": lag_da,
            "cross_correlation": ccf_da,
            "cross_correlation_pvalue": pvals_da
        },
        attrs={"description": "Cross-correlation and p-values for all lags"}
    )
    return ds

def cross_correlation_maxima(ts1: TimeSeries, ts2: TimeSeries, maxlags=None):
    """
    Find the lag and value of maximum cross-correlation.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series.
    maxlags : int, optional
        Maximum lag to compute.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            - lag_max: lag at which maximum correlation occurs
            - ccf_max: maximum correlation value
    """
    ds = cross_correlation(ts1, ts2, maxlags)
    idx = np.argmax(ds["cross_correlation"].values)
    lag_max = ds["lag"].values[idx]
    ccf_max = ds["cross_correlation"].values[idx]

    return lag_max, ccf_max

def shift_maximally_correlated(ts1: TimeSeries, ts2: TimeSeries, maxlags=None):
    """
    Align two time series at the lag of maximal cross-correlation.
    
    This function finds the lag at which the cross-correlation between
    two time series is maximized and returns both series aligned at
    that optimal lag, with overlapping time windows.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series to align. Must have compatible time steps.
    maxlags : int, optional
        Maximum lag to consider when searching for optimal alignment.
        If None, uses full range (N-1 where N is series length).

    Returns
    -------
    ts1_shifted, ts2_shifted : tuple of TimeSeries
        Time series aligned at the lag of maximum correlation, with
        matching time windows. Both series will have the same length
        and time coordinates after alignment.
        
    Raises
    ------
    ValueError
        If the time series have incompatible time steps or lengths.
        
    Notes
    -----
    The alignment process:
    1. Computes cross-correlation for all lags within maxlags
    2. Identifies the lag with maximum correlation
    3. Trims both series to their overlapping time window at that lag
    
    This is useful for analyzing relationships between time series
    that may be offset in time due to physical delays or measurement
    timing differences.
    
    Examples
    --------
    >>> ts1_aligned, ts2_aligned = shift_maximally_correlated(ts1, ts2, maxlags=50)
    >>> # Now ts1_aligned and ts2_aligned have maximum correlation at zero lag
    """
    lag_max, _ = cross_correlation_maxima(ts1, ts2, maxlags)
    dt = ts1.time[1] - ts1.time[0] if len(ts1.time) > 1 else 1.0
    k = int(np.round(lag_max / dt))
    n = ts1.N
    x = ts1.data
    y = ts2.data
    time = ts1.time
    if k < 0:
        xa = x[:n+k]
        ya = y[-k:]
        ta = time[:n+k]
    else:
        xa = x[k:]
        ya = y[:n-k]
        ta = time[k:]
    return TimeSeries(ta, xa, ts1.dt), TimeSeries(ta, ya, ts2.dt)

def bootstrap_correlation(ts1: TimeSeries, ts2: TimeSeries, n_iter=1000, detrend=True):
    """
    Compute bootstrapped correlation and p-value using phase scrambling.
    p-values are calculated using the empirical cumulative distribution function (ECDF) 
    and a one-sided test of the observed correlation against the bootstrapped distribution.
    being greater than the observed correlation.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series.
    n_iter : int, default 1000
        Number of bootstrap iterations.
    detrend : bool, default True
        If True, detrend the signal before scrambling.

    Returns
    -------
    tuple
        (ref_corr, boot_p_value, bootstrapped_correlation):
            - ref_corr: observed correlation (float)
            - boot_p_value: bootstrapped p-value (float)
            - bootstrapped_correlation: bootstrapped correlation values (np.ndarray)
    """
    if ts1.dt != ts2.dt:
        raise ValueError(f"TimeSeries dt mismatch: {ts1.dt} vs {ts2.dt}")

    ref_corr = pearsonr(ts1.data, ts2.data)[0]
    xs_surrogates = phase_scrambled(ts1, detrend=detrend, n_scrambled=n_iter)
    ys_surrogates = phase_scrambled(ts2, detrend=detrend, n_scrambled=n_iter)
    # Use scipy.stats.pearsonr with axis=1 for vectorized calculation
    corrs = pearsonr(xs_surrogates, ys_surrogates, axis=1)[0]
    ecdf = ECDF(corrs)
    boot_p_value = 2 * (1 - ecdf(ref_corr))

    return ref_corr, boot_p_value, corrs

def bootstrapped_cross_correlation(ts1: TimeSeries, ts2: TimeSeries, maxlags=None, 
                                   n_iter=1000, return_distributions=False, detrend=True):
    """
    Compute bootstrapped cross-correlation and p-values for all lags.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series.
    maxlags : int, optional
        Maximum lag to compute.
    n_iter : int, default 1000
        Number of bootstrap iterations.
    return_distributions : bool, default False
        If True, return full bootstrap distributions.
    detrend : bool, default True
        If True, detrend the signal before scrambling.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            - lag: lag values
            - bootstrapped_cross_correlation: cross-correlation coefficients
            - bootstrapped_cross_correlation_pvalue: bootstrapped p-values for each lag
            - bootstrapped_cross_correlation_distribution: bootstrap distributions (if requested)
    """
    x = np.asarray(ts1.data)
    y = np.asarray(ts2.data)
    n = x.size
    if y.size != n:
        raise ValueError("Input vectors must have the same length")
    
    if ts1.dt != ts2.dt:
        raise ValueError(f"TimeSeries dt mismatch: {ts1.dt} vs {ts2.dt}")
    
    if ts1.n != ts2.n:
        raise ValueError(f"TimeSeries n mismatch: {ts1.n} vs {ts2.n}")
    
    dt = ts1.dt

    if maxlags is None:
        maxlags = n - 1
    maxlags = min(maxlags, n - 1)
    lags = np.arange(-maxlags, maxlags+1) * dt
    ccf = np.zeros(len(lags))
    ccf_pval = np.zeros(len(lags))
    if return_distributions:
        ccf_dist = np.zeros((n_iter, len(lags)))
    for i, k in enumerate(trange(-maxlags, maxlags+1, desc="Bootstrapping lags")):
        if k < 0:
            xi = x[:n+k]
            yi = y[-k:]
            ti = ts1.time[:n+k]
        else:
            xi = x[k:]
            yi = y[:n-k]
            ti = ts1.time[k:]
        ts_xi = TimeSeries(ti, xi, ts1.dt)
        ts_yi = TimeSeries(ti, yi, ts2.dt)
        ref_corr, boot_p_value, corrs = bootstrap_correlation(ts_xi, ts_yi, n_iter=n_iter, 
                                                              detrend=detrend)
        ccf[i] = ref_corr
        ccf_pval[i] = boot_p_value
        if return_distributions:
            ccf_dist[:, i] = corrs


    ccf_da = xr.DataArray(
        ccf,
        coords={"lag": lags},
        dims=["lag"],
        name="bootstrapped_cross_correlation",
        attrs={"description": "Bootstrapped cross-correlation coefficients at each lag"}
    )
    ccf_pval_da = xr.DataArray(
        ccf_pval,
        coords={"lag": lags},
        dims=["lag"],
        name="bootstrapped_cross_correlation_pvalue",
        attrs={"description": "Bootstrapped p-values for cross-correlation at each lag"}
    )
    lag_da = xr.DataArray(
        lags,
        dims=["lag"],
        name="lag",
        attrs={"description": "Lag values"}
    )
    data_vars = {
        "lag": lag_da,
        "bootstrapped_cross_correlation": ccf_da,
        "bootstrapped_cross_correlation_pvalue": ccf_pval_da
    }

    if return_distributions:

        ccf_dist_da = xr.DataArray(
            ccf_dist,
            coords={"bootstrap_iter": np.arange(n_iter), "lag": lags},
            dims=["bootstrap_iter", "lag"],
            name="bootstrapped_cross_correlation_distribution",
            attrs={"description": "Bootstrap distributions for cross-correlation at each lag"}
        )
        data_vars["bootstrapped_cross_correlation_distribution"] = ccf_dist_da
    ds = xr.Dataset(
        data_vars,
        attrs={"description": "Bootstrapped cross-correlation and p-values for all lags"}
    )
    return ds