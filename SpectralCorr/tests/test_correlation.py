"""
Tests for the unified correlation function.
"""

import numpy as np
import pytest
from SpectralCorr import AR1_process, correlation, TimeSeries


class TestCorrelation:
    """Test suite for the unified correlation function."""

    def test_correlation_pearson_method(self):
        """Test correlation function with Pearson method."""
        # Create two correlated time series (using xarray DataArrays)
        ts1 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=100, seed=42)
        ts2 = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=100, seed=42)  # Same seed for correlation

        corr, p_value = correlation(ts1, ts2, method='pearson')

        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert not np.isnan(corr)
        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1

    def test_correlation_ebisuzaki_method(self):
        """Test correlation function with Ebisuzaki method."""
        ts1 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42)
        ts2 = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=42)

        corr, p_value = correlation(ts1, ts2, method='ebisuzaki', n_iter=50)  # Small n_iter for speed

        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert not np.isnan(corr)
        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1

    def test_correlation_xarray_inputs(self):
        """Test correlation function with xarray DataArray inputs."""
        ts1_xr = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42)  # returns xarray by default
        ts2_xr = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=123)

        corr, p_value = correlation(ts1_xr, ts2_xr, method='pearson')

        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert not np.isnan(corr)
        assert not np.isnan(p_value)

    def test_correlation_different_seeds(self):
        """Test correlation function with different time series."""
        ts1 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42)
        ts2 = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=123)

        # Test with both methods
        corr1, p_value1 = correlation(ts1, ts2, method='pearson')
        assert isinstance(corr1, float) and isinstance(p_value1, float)

        corr2, p_value2 = correlation(ts1, ts2, method='ebisuzaki', n_iter=50)
        assert isinstance(corr2, float) and isinstance(p_value2, float)

        # Correlation values should be similar between methods
        assert abs(corr1 - corr2) < 0.1  # Allow some difference due to bootstrap

    def test_correlation_identical_series(self):
        """Test correlation with identical time series (should be 1.0)."""
        ts = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=50, seed=42)

        corr, p_value = correlation(ts, ts, method='pearson')

        assert np.isclose(corr, 1.0, atol=1e-10)
        assert p_value < 0.01  # Should be highly significant

    def test_correlation_uncorrelated_series(self):
        """Test correlation with uncorrelated time series."""
        ts1 = AR1_process(rho=0.1, sigma=1.0, y0=0.0, N=200, seed=42)
        ts2 = AR1_process(rho=0.1, sigma=1.0, y0=0.0, N=200, seed=999)  # Different seed

        corr, p_value = correlation(ts1, ts2, method='pearson')

        # Should be close to zero for uncorrelated series
        assert abs(corr) < 0.2  # Allow some variation due to randomness
        # P-value might or might not be significant depending on chance

    def test_correlation_invalid_method(self):
        """Test error handling for invalid method."""
        ts1 = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=50, seed=42)
        ts2 = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=50, seed=123)

        with pytest.raises(ValueError, match="Method must be 'pearson' or 'ebisuzaki'"):
            correlation(ts1, ts2, method='invalid_method')

    def test_correlation_mismatched_dt(self):
        """Test error handling for mismatched time steps."""
        import xarray as xr
        ts1 = xr.DataArray(np.random.randn(50), coords={'time': np.arange(50) * 1.0}, dims=['time'])
        ts2 = xr.DataArray(np.random.randn(50), coords={'time': np.arange(50) * 2.0}, dims=['time'])

        with pytest.raises(ValueError, match="Time coordinates must be aligned"):
            correlation(ts1, ts2, method='pearson')

    def test_correlation_mismatched_length(self):
        """Test error handling for mismatched time series lengths."""
        import xarray as xr
        ts1 = xr.DataArray(np.random.randn(50), coords={'time': np.arange(50)}, dims=['time'])
        ts2 = xr.DataArray(np.random.randn(40), coords={'time': np.arange(40)}, dims=['time'])

        with pytest.raises(ValueError, match="Time series must have same length"):
            correlation(ts1, ts2, method='pearson')

    def test_correlation_insufficient_data(self):
        """Test handling of insufficient data points."""
        import xarray as xr
        ts1 = xr.DataArray([1.0], coords={'time': [0]}, dims=['time'])
        ts2 = xr.DataArray([2.0], coords={'time': [0]}, dims=['time'])

        corr, p_value = correlation(ts1, ts2, method='pearson')

        assert np.isnan(corr)
        assert np.isnan(p_value)

    def test_correlation_ebisuzaki_parameters(self):
        """Test Ebisuzaki method with different parameters."""
        ts1 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42)
        ts2 = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=123)

        # Test with different n_iter
        corr1, p_val1 = correlation(ts1, ts2, method='ebisuzaki', n_iter=20)
        corr2, p_val2 = correlation(ts1, ts2, method='ebisuzaki', n_iter=50)

        # Correlation should be similar, p-values might vary due to different bootstrap samples
        assert np.isclose(corr1, corr2, atol=0.01)

        # Test with different detrend setting
        corr3, p_val3 = correlation(ts1, ts2, method='ebisuzaki', n_iter=20, detrend=False)
        assert isinstance(corr3, float) and isinstance(p_val3, float)

    def test_correlation_methods_give_same_correlation(self):
        """Test that both methods give the same correlation coefficient."""
        ts1 = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=100, seed=42, return_xarray=False)
        ts2 = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=100, seed=123, return_xarray=False)

        corr_pearson, _ = correlation(ts1, ts2, method='pearson')
        corr_ebisuzaki, _ = correlation(ts1, ts2, method='ebisuzaki', n_iter=50)

        # The correlation coefficients should be very similar
        assert np.isclose(corr_pearson, corr_ebisuzaki, atol=0.01)

    def test_correlation_returns_tuple(self):
        """Test that correlation always returns a tuple of two values."""
        ts1 = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=50, seed=42, return_xarray=False)
        ts2 = AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=50, seed=123, return_xarray=False)

        result = correlation(ts1, ts2, method='pearson')
        assert isinstance(result, tuple)
        assert len(result) == 2

        result2 = correlation(ts1, ts2, method='ebisuzaki', n_iter=20)
        assert isinstance(result2, tuple)
        assert len(result2) == 2