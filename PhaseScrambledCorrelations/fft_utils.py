"""
FFT utilities for time series analysis.
"""

import numpy as np
import xarray as xr
from numpy.fft import rfft, rfftfreq, irfft
from .timeseries import TimeSeries

def simple_detrend(time, signal):
    """
    Remove the mean from a signal for simple detrending.
    
    This is a minimal detrending approach that subtracts only the mean,
    preserving all variability without fitting linear trends.
    
    Parameters
    ----------
    time : array-like
        Time points (not used in calculation, kept for interface consistency).
    signal : array-like
        Input signal to detrend.
        
    Returns
    -------
    np.ndarray
        Signal with mean removed.
        
    Notes
    -----
    This function only removes the DC component (mean) rather than 
    fitting and removing linear trends. This preserves all temporal
    variability in the signal which is often desired for spectral
    analysis and phase scrambling applications.
    """
    return signal - np.mean(signal)

def real_fft(ts: TimeSeries, detrend=True, periodogram=False):
    """
    Compute the real-valued FFT or periodogram of a time series.

    Parameters
    ----------
    ts : TimeSeries
        Input time series data.
    detrend : bool, default True
        If True, detrend the signal before FFT using scipy.signal.detrend.
    periodogram : bool, default False
        If True, return periodogram (magnitude spectrum, excluding zero-frequency term).

    Returns
    -------
    spectrum : xarray.DataArray
        FFT coefficients (complex) or periodogram (magnitude, 1/freq) as DataArray.
        Coordinates: 'freq' (FFT) or 'period' (periodogram).
        Name and description reflect output type.
    """
    signal = np.asarray(ts.data)
    time = np.asarray(ts.time)
    nt = signal.size
    if detrend or periodogram:
        signal = simple_detrend(time,  signal)

    F = rfft(signal)

    dt = ts.dt

    freqs = rfftfreq(nt, dt)

    if periodogram:
        # exclude zero-frequency
        nonzero_freqs = freqs[1:]
        periods = 1.0 / nonzero_freqs
        magnitude = np.abs(F[1:])
        spectrum_da = xr.DataArray(
            magnitude,
            coords={"period": periods},
            dims=["period"],
            name="periodogram",
            attrs={
                "description": "Periodogram (magnitude spectrum, 1/freq)",
                "dt": dt,
                "N": nt,
                "detrended": detrend
            }
        )
        return spectrum_da
    else:
        spectrum_da = xr.DataArray(
            F,
            coords={"freq": freqs},
            dims=["freq"],
            name="fft_coefficients",
            attrs={
                "description": "FFT coefficients (complex)",
                "dt": dt,
                "N": nt,
                "detrended": detrend
            }
        )
        return spectrum_da

