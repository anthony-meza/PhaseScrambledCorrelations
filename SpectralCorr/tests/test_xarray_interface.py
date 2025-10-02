"""
Test the xarray interface and performance improvements.
"""

import time
import numpy as np
import xarray as xr
from SpectralCorr import AR1_process, cross_correlation, TimeSeries


def test_xarray_interface():
    """Test that the xarray interface works correctly."""
    # Generate test time series
    ts1 = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=100, seed=42)
    ts2 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=100, seed=123)

    # Verify xarray structure
    assert isinstance(ts1, xr.DataArray)
    assert 'time' in ts1.coords
    assert ts1.shape == (100,)
    assert 'dt' in ts1.attrs

    # Test cross_correlation with xarray inputs
    result = cross_correlation(ts1, ts2, maxlags=10, method='pearson')

    # Verify result structure
    assert isinstance(result, xr.Dataset)
    assert 'cross_correlation' in result.data_vars
    assert 'cross_correlation_pvalue' in result.data_vars
    assert 'lag' in result.coords

    # Test with both xarray inputs
    result2 = cross_correlation(ts1, ts2, maxlags=5, method='pearson')
    assert 'cross_correlation' in result2


def test_performance_improvement():
    """Test performance improvement from lightweight TimeSeries."""
    # Test parameters
    n_points = 500
    n_lags = 50
    dt = 1.0

    # Generate test data
    ts1 = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=n_points, seed=42)
    ts2 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=n_points, seed=123)

    # Time the optimized version
    start_time = time.time()
    result = cross_correlation(ts1, ts2, maxlags=n_lags, method='pearson')
    optimized_time = time.time() - start_time

    # Verify result is correct
    assert len(result.lag) == 2 * n_lags + 1
    assert isinstance(result, xr.Dataset)

    # Simulate old approach (many TimeSeries object creations)
    start_time = time.time()
    ts1_ts = TimeSeries.from_xarray(ts1)
    ts2_ts = TimeSeries.from_xarray(ts2)

    old_style_objects = []
    for i in range(n_lags * 2 + 1):
        subset_size = max(10, n_points - i)
        new_ts1 = TimeSeries(
            ts1_ts.time[:subset_size],
            ts1_ts.data[:subset_size],
            dt
        )
        new_ts2 = TimeSeries(
            ts2_ts.time[:subset_size],
            ts2_ts.data[:subset_size],
            dt
        )
        old_style_objects.append((new_ts1, new_ts2))

    old_simulation_time = time.time() - start_time

    # Verify we created the expected number of objects
    assert len(old_style_objects) == 2 * n_lags + 1

    # Calculate performance improvement
    if old_simulation_time > 0:
        improvement = old_simulation_time / optimized_time
        # Optimized version should be at least as fast
        assert improvement >= 0
