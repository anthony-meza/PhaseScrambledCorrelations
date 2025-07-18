"""
phase_scrambled_corr.py: Phase-scrambled Monte Carlo correlation analysis

This module implements Monte Carlo phase scrambling for correlation and cross-correlation
analysis following Ebisuzaki (1997).

# References
- Ebisuzaki, W. (1997). A Method to Estimate the Statistical Significance of a
  Correlation When the Data Are Serially Correlated. *Journal of Climate*, 10(9), 2147–2153.
"""

import numpy as np
from numpy.fft import rfft, rfftfreq, irfft
from scipy.stats import pearsonr
from statsmodels.distributions.empirical_distribution import ECDF

def real_fft(signal, fs, remove_mean=True, periodogram=False):
    """
    Compute the real-valued FFT of a time series.

    Parameters
    ----------
    signal : array_like
        Input time series data.
    fs : float
        Sampling frequency (Hz).
    remove_mean : bool, default True
        If True, subtract the mean before FFT.
    periodogram : bool, default False
        If True, return 1/freq and FFT magnitudes (excluding zero-frequency term).

    Returns
    -------
    freqs : ndarray
        Array of frequency bins.
    spectrum : ndarray
        FFT of the input signal (complex values) or magnitudes if periodogram.
    """
    signal = np.asarray(signal)
    nt = signal.size
    if remove_mean or periodogram:
        signal = signal - np.mean(signal)
    F = rfft(signal)
    # zero out the DC component
    F[0] = 0 + 0j
    freqs = rfftfreq(nt, 1/fs)

    if periodogram:
        # exclude zero-frequency
        nonzero = freqs[1:]
        return 1.0 / nonzero, F[1:]
    return freqs, F


def phase_scrambled(signal, fs, remove_mean=True):
    """
    Generate a phase-randomized surrogate of a real-valued time series.

    Parameters
    ----------
    signal : array_like
        Input time series.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    surrogate : ndarray
        Time series with same power spectrum but random phases.
    """
    signal = np.asarray(signal)
    nt = signal.size
    freqs, F = real_fft(signal, fs, remove_mean=remove_mean)
    nf = freqs.size

    # generate random phases
    phases = np.random.uniform(0, 2*np.pi, size=nf)
    # preserve amplitude, apply random phases (exclude DC)
    F[1:nf-1] = np.abs(F[1:nf-1]) * np.exp(1j * phases[1:nf-1])
    # Nyquist frequency if even-length
    if nt % 2 == 0 and nf > 1:
        F[-1] = np.abs(F[-1]) * np.cos(phases[-1]) * np.sqrt(2)

    surrogate = irfft(F, nt)
    return surrogate


def cross_correlation(x, y, dt=1.0, maxlags=None):
    """
    Compute cross-correlation function between two real sequences.

    Parameters
    ----------
    x, y : array_like
        Input time series (same length).
    dt : float, default 1.0
        Time step between samples.
    maxlags : int, optional
        Maximum lag (in number of samples). Defaults to length-1.

    Returns
    -------
    lags : ndarray
        Array of lag times.
    ccf : ndarray
        Cross-correlation coefficients at each lag.
    """
    x = np.asarray(x)
    y = np.asarray(y)
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

    return lags, ccf, pvals

def cross_correlation_maxima(x, y, dt=1.0, maxlags=None):
    """
    Find the lag with maximum cross-correlation.

    Returns
    -------
    lag_max : float
        Lag time at maximum correlation.
    ccf_max : float
        Maximum correlation coefficient.
    """
    lags, ccf, _ = cross_correlation(x, y, dt, maxlags)
    idx = np.argmax(ccf)
    return lags[idx], ccf[idx]


def shift_maximally_correlated(x, y, dt=1.0, maxlags=None):
    """
    Shift two series to align at their maximum correlation lag.

    Returns
    -------
    x_aligned, y_aligned : ndarray
        Truncated arrays of equal length aligned at optimal lag.
    """
    lag_max, _ = cross_correlation_maxima(x, y, dt, maxlags)
    k = int(np.round(lag_max / dt))
    n = x.size

    if k < 0:
        xa = x[:n+k]
        ya = y[-k:]
    else:
        xa = x[k:]
        ya = y[:n-k]
    return xa, ya


def bootstrap_corr(x, y, n_iter=1000, fs=1.0, remove_mean = True):
    """
    Estimate null distribution of correlation via phase-randomized surrogates.

    Parameters
    ----------
    x, y : array_like
        Input time series.
    n_iter : int
        Number of surrogate pairs.
    fs : float
        Sampling frequency for phase randomization.

    Returns
    -------
    corrs : ndarray
        Array of correlation coefficients from surrogate pairs.
    """
    corrs = np.empty(n_iter, dtype=float)
    ref_corr = pearsonr(x, y)[0]
    for i in range(n_iter):
        xs = phase_scrambled(x, fs, remove_mean = remove_mean)
        ys = phase_scrambled(y, fs, remove_mean = remove_mean)
        corrs[i] = pearsonr(xs, ys)[0]

    ecdf = ECDF(corrs)
    boot_p_value = 2 * (1 - ecdf(ref_corr))

    return ref_corr, boot_p_value, corrs


def bootstrapped_cross_correlation(x, y, dt=1.0, maxlags=None, 
                                   n_iter = 1000, return_distributions = False, remove_mean = True):
    """
    Compute cross-correlation function between two real sequences.

    Parameters
    ----------
    x, y : array_like
        Input time series (same length).
    dt : float, default 1.0
        Time step between samples.
    maxlags : int, optional
        Maximum lag (in number of samples). Defaults to length-1.

    Returns
    -------
    lags : ndarray
        Array of lag times.
    ccf : ndarray
        Cross-correlation coefficients at each lag.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    if y.size != n:
        raise ValueError("Input vectors must have the same length")

    if maxlags is None:
        maxlags = n - 1
    maxlags = min(maxlags, n - 1)


    fs = 1 / dt
    lags = np.arange(-maxlags, maxlags+1) * dt
    ccf = np.zeros(len(lags))
    ccf_pval = np.zeros(len(lags))
    if return_distributions:
        ccf_dist = np.zeros(n_iter, len(lags))

    for i, k in enumerate(range(-maxlags, maxlags+1)):
        if k < 0:
            xi = x[:n+k]
            yi = y[-k:]
        else:
            xi = x[k:]
            yi = y[:n-k]

        ref_corr, p_value, corrs = bootstrap_corr(xi, yi, n_iter=n_iter, fs=fs, remove_mean = remove_mean)
        ccf[i] = ref_corr
        ccf_pval[i] = p_value
        if return_distributions:
            ccf_dist[:, i] = corrs
    if return_distributions:
        return lags, ccf, ccf_pval, ccf_dist
    else:
        return lags, ccf, ccf_pval