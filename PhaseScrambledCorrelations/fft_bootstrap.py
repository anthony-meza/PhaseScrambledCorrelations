import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from statsmodels.distributions.empirical_distribution import ECDF
from .fft_utils import real_fft
from .timeseries import TimeSeries

def phase_scrambled(ts: TimeSeries, detrend=True, n_scrambled=1, return_xarray=False):
    """
    Generate one or more phase-scrambled surrogates of a time series.

    Parameters
    ----------
    ts : TimeSeries
        Input time series.
    detrend : bool, default True
        If True, detrend the signal before scrambling.
    n_scrambled : int, default 1
        Number of surrogates to generate.
    return_xarray : bool, default False
        If True, return xarray.DataArray (with 'time' and 'iter' dims if n_scrambled > 1).

    Returns
    -------
    np.ndarray or xarray.DataArray
        Surrogate(s) of the time series. Shape (n_scrambled, len(time)) if n_scrambled > 1.
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

        F_copy[0] = 0 * F_copy[0] #zero out the mean
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