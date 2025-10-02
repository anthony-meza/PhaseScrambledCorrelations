"""
Phase scrambling utilities for generating surrogate time series.

This module implements the phase randomization method for creating
surrogate time series that preserve the power spectrum while destroying
temporal correlations. Used for significance testing in time series analysis.
"""

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from statsmodels.distributions.empirical_distribution import ECDF
from .fft_utils import real_fft
from .timeseries import TimeSeries

def phase_scrambled(ts: TimeSeries, detrend=True, n_scrambled=1, return_xarray=False):
    """
    Generate phase-scrambled surrogates of a time series.

    Creates surrogate time series by randomizing the phases of Fourier components
    while preserving the power spectrum. This destroys temporal correlations
    while maintaining spectral properties, enabling robust significance testing.

    Parameters
    ----------
    ts : TimeSeries
        Input time series to scramble.
    detrend : bool, default True
        If True, detrend the signal before phase scrambling.
    n_scrambled : int, default 1
        Number of surrogate series to generate.
    return_xarray : bool, default False
        If True, return as xarray.DataArray with proper coordinates.

    Returns
    -------
    np.ndarray or xarray.DataArray
        Phase-scrambled surrogate(s). Shape is (n_scrambled, len(time)) when
        n_scrambled > 1, otherwise (len(time),).

    Notes
    -----
    The phase scrambling algorithm:
    1. Computes FFT of the input signal
    2. Randomizes phases while preserving magnitudes
    3. Applies inverse FFT to get surrogate time series
    4. Handles even/odd length signals appropriately

    This method follows Ebisuzaki (1997) and is widely used in climate
    and geophysical time series analysis for creating null distributions.

    References
    ----------
    Ebisuzaki, W. (1997). A method to estimate the statistical significance of a
    correlation when the data are serially correlated. Journal of Climate, 10(9),
    2147-2153.
    """
    signal = np.asarray(ts.data)
    nt = signal.size
    F_da = real_fft(ts, detrend=detrend)
    freqs = F_da.coords["freq"].values
    nf = freqs.size

    F = F_da.values.copy()
    surrogates = np.empty((n_scrambled, nt), dtype=float)
    for i in range(n_scrambled):
        F_copy = F.copy()
        phases = np.random.uniform(0, 2*np.pi, size=nf)

        F_copy[0] = 0 * F_copy[0]  # Zero out the mean (DC component)
        F_copy[1:nf-1] = np.abs(F_copy[1:nf-1]) * np.exp(1j * phases[1:nf-1])
        if nt % 2 == 0 and nf > 1:
            F_copy[-1] = np.abs(F_copy[-1]) * np.cos(phases[-1]) * np.sqrt(2)
        surrogates[i, :] = np.real(np.fft.irfft(F_copy, nt))

    if n_scrambled == 1:
        result = surrogates[0]
    else:
        result = surrogates

    if return_xarray:
        da = xr.DataArray(
            result,
            coords={"iter": np.arange(n_scrambled), "time": ts.time},
            dims=["iter", "time"],
            name="phase_scrambled",
            attrs={
                "description": f"{n_scrambled} phase-scrambled surrogate time series",
                "dt": ts.dt,
                "original_detrended": detrend
            }
        )
        if n_scrambled == 1:
            da = da.drop_vars("iter")
        return da
    else:
        return result