import numpy as np 

def AR1_simulate(rho, sigma, y0, N, seed=None):
    """
    Simulate ana AR(1) process: y[t] = rho*y[t-1] + eps[t]

    Parameters
    ----------
    rho : float
        AR(1) coefficient.
    sigma : float
        Standard deviation of Gaussian noise.
    y0 : float
        Initial value.
    T : int
        Length of time series.

    Returns
    -------
    y : ndarray
        Simulated AR(1) series of length T.
    """
    if seed is not None:
        np.random.seed(seed)
        
    y = np.empty(N)
    eps = np.random.normal(0, sigma, N)
    y[0] = y0
    for t in range(N):
        y[t] = rho * y[t-1] + eps[t]
    return range(N), y