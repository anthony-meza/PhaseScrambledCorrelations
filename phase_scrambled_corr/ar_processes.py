"""
Example time series generation utilities.
"""

from .timeseries import TimeSeries
import numpy as np

def AR1_process(rho, sigma, y0, N, seed=None, dt=1.0):
    """
    Simulate an AR(1) process: y[t] = rho*y[t-1] + eps[t]

    Parameters
    ----------
    rho : float
        AR(1) coefficient.
    sigma : float
        Standard deviation of Gaussian noise.
    y0 : float
        Initial value.
    N : int
        Length of time series.
    seed : int, optional
        Random seed for reproducibility.
    dt : float, optional
        Time step between samples.

    Returns
    -------
    TimeSeries
        Simulated AR(1) time series object.
    """
    if seed is not None:
        np.random.seed(seed)
    data = np.zeros(N)
    data[0] = y0
    for t in range(1, N):
        data[t] = rho * data[t-1] + np.random.normal(0, sigma)
    time = np.arange(N) * dt
    return TimeSeries(time, data, dt)