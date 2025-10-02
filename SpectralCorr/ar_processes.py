"""
Autoregressive time series generation utilities.

This module provides functions for generating synthetic time series data
following autoregressive models, particularly AR(1) processes commonly
used in climate and geophysical applications.
"""

from typing import Optional
import numpy as np
from .timeseries import TimeSeries

def AR1_process(
    rho: float,
    sigma: float,
    y0: float,
    N: int,
    seed: Optional[int] = None,
    dt: float = 1.0,
    return_xarray: bool = True
):
    """
    Simulate a first-order autoregressive AR(1) process.

    Generates a time series following the equation:
    y[t] = rho * y[t-1] + eps[t]

    where eps[t] ~ N(0, sigma²) is white Gaussian noise.

    Parameters
    ----------
    rho : float
        AR(1) coefficient (-1 < rho < 1 for stationarity).
        Values closer to 1 indicate stronger persistence.
    sigma : float
        Standard deviation of Gaussian noise innovations.
        Must be positive.
    y0 : float
        Initial value y[0] of the time series.
    N : int
        Number of time points to generate. Must be positive.
    seed : int, optional
        Random seed for reproducible results. If None, uses
        current random state.
    dt : float, optional, default=1.0
        Time step between consecutive samples.
    return_xarray : bool, optional, default=True
        If True, return xarray.DataArray. If False, return TimeSeries object.

    Returns
    -------
    xarray.DataArray or TimeSeries
        Simulated AR(1) time series. Default is xarray.DataArray with time coordinate.
        
    Raises
    ------
    ValueError
        If rho is not in (-1, 1), sigma <= 0, or N <= 0.
        
    Notes
    -----
    The AR(1) process is stationary when |rho| < 1, with:
    - Mean: y0 / (1 - rho) (asymptotic)
    - Variance: sigma² / (1 - rho²) (asymptotic)
    - Autocorrelation at lag k: rho^k
    
    References
    ----------
    Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). 
    Time series analysis: forecasting and control. John Wiley & Sons.
    
    Examples
    --------
    Generate a stationary AR(1) process with strong persistence:
    
    >>> ts = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=100, seed=42)
    >>> print(f"Generated {ts.n} points with dt={ts.dt}")
    Generated 100 points with dt=1.0
    """
    # Input validation
    if not (-1 < rho < 1):
        raise ValueError(f"AR(1) coefficient rho must be in (-1, 1) for stationarity, got {rho}")
    if sigma <= 0:
        raise ValueError(f"Noise standard deviation sigma must be positive, got {sigma}")
    if N <= 0:
        raise ValueError(f"Number of points N must be positive, got {N}")
    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize time series
    data = np.zeros(N)
    data[0] = y0
    
    # Generate AR(1) process iteratively
    for t in range(1, N):
        noise = np.random.normal(0, sigma)
        data[t] = rho * data[t-1] + noise
    
    # Create time array
    time = np.arange(N) * dt

    # Create TimeSeries object for internal representation
    ts = TimeSeries(time, data, dt)

    # Return format based on user preference
    if return_xarray:
        return ts.to_xarray()
    else:
        return ts