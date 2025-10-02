"""
Test basic AR1 process simulation functionality.
"""

import numpy as np
import pytest
from SpectralCorr.ar_processes import AR1_process


def test_ar1_simulate_length_and_reproducibility():
    """Test that AR1_process produces correct length and is reproducible."""
    # Simulation parameters
    N = 250
    dt = 1.0
    rho = 0.9
    noise_std = 1.0
    y0 = 1

    # Generate two time series with the same seed
    ts1 = AR1_process(rho, noise_std, y0, N, seed=42, dt=dt, return_xarray=False)
    ts2 = AR1_process(rho, noise_std, y0, N, seed=42, dt=dt, return_xarray=False)

    # Test 1: Verify correct length
    assert ts1.n == N
    assert ts2.n == N

    # Test 2: Verify time array starts at 0 and increments by dt
    expected_time = np.arange(N) * dt
    assert np.allclose(ts1.time, expected_time)

    # Test 3: Verify reproducibility with same seed
    assert np.array_equal(ts1.data, ts2.data)
