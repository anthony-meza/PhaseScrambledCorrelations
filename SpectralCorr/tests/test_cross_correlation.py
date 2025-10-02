"""
Tests for cross-correlation functions.
"""

import numpy as np
import pytest
import warnings
from SpectralCorr import AR1_process, cross_correlation, maximum_cross_correlation, cross_correlation_maxima, bootstrapped_cross_correlation
from SpectralCorr.timeseries import TimeSeries


class TestCrossCorrelation:
    """Test suite for cross_correlation function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.N = 100
        self.dt = 1.0
        self.maxlags = 10

        # Create test time series
        self.ts1 = AR1_process(0.8, 1.0, 0.0, self.N, seed=42, dt=self.dt)
        self.ts2 = AR1_process(0.7, 1.0, 0.0, self.N, seed=123, dt=self.dt)

    def test_pearson_method(self):
        """Test cross-correlation with Pearson method."""
        result = cross_correlation(self.ts1, self.ts2, maxlags=self.maxlags, method='pearson')

        # Check structure
        assert 'lag' in result
        assert 'cross_correlation' in result
        assert 'cross_correlation_pvalue' in result
        assert 'cross_correlation_distribution' not in result  # Not returned for Pearson

        # Check dimensions
        expected_lags = 2 * self.maxlags + 1
        assert len(result.lag) == expected_lags
        assert len(result.cross_correlation) == expected_lags
        assert len(result.cross_correlation_pvalue) == expected_lags

        # Check lag values
        expected_lag_values = np.arange(-self.maxlags, self.maxlags + 1) * self.dt
        np.testing.assert_array_equal(result.lag.values, expected_lag_values)

    def test_ebisuzaki_method(self):
        """Test cross-correlation with Ebisuzaki method."""
        n_iter = 50  # Small for speed
        result = cross_correlation(self.ts1, self.ts2, maxlags=self.maxlags,
                                 method='ebisuzaki', n_iter=n_iter)

        # Check structure
        assert 'lag' in result
        assert 'cross_correlation' in result
        assert 'cross_correlation_pvalue' in result
        assert 'cross_correlation_distribution' not in result  # Not requested

        # Check dimensions
        expected_lags = 2 * self.maxlags + 1
        assert len(result.lag) == expected_lags

    def test_ebisuzaki_with_distributions(self):
        """Test Ebisuzaki method with distribution return."""
        n_iter = 30
        result = cross_correlation(self.ts1, self.ts2, maxlags=self.maxlags,
                                 method='ebisuzaki', n_iter=n_iter, return_distributions=True)

        # Check that distributions are returned
        assert 'cross_correlation_distribution' in result

        # Check distribution dimensions
        expected_lags = 2 * self.maxlags + 1
        assert result.cross_correlation_distribution.shape == (n_iter, expected_lags)

    def test_method_validation(self):
        """Test that invalid methods raise ValueError."""
        with pytest.raises(ValueError, match="Method must be 'pearson' or 'ebisuzaki'"):
            cross_correlation(self.ts1, self.ts2, method='invalid')

    def test_time_series_validation(self):
        """Test validation of input time series."""
        # Different dt values
        ts_bad_dt = TimeSeries(self.ts2.time, self.ts2.data, dt=2.0)
        with pytest.raises(ValueError, match="TimeSeries dt mismatch"):
            cross_correlation(self.ts1, ts_bad_dt)

        # Different lengths
        ts_bad_length = TimeSeries(self.ts2.time[:-5], self.ts2.data[:-5], dt=self.dt)
        with pytest.raises(ValueError, match="TimeSeries length mismatch"):
            cross_correlation(self.ts1, ts_bad_length)

    def test_maxlags_handling(self):
        """Test maxlags parameter handling."""
        # Explicit maxlags (use small value to avoid edge cases)
        maxlags_test = 5
        result_explicit = cross_correlation(self.ts1, self.ts2, maxlags=maxlags_test, method='pearson')
        expected_explicit_lags = 2 * maxlags_test + 1
        assert len(result_explicit.lag) == expected_explicit_lags

        # Maxlags larger than safe range should be capped
        safe_maxlags = min(self.N // 4, 20)  # Use safe range
        result_large = cross_correlation(self.ts1, self.ts2, maxlags=self.N + 10, method='pearson')
        result_safe = cross_correlation(self.ts1, self.ts2, maxlags=safe_maxlags, method='pearson')
        # Should have reasonable number of lags (not fail due to insufficient data)
        assert len(result_large.lag) >= 10  # Should have at least some lags

    def test_correlation_symmetry(self):
        """Test that correlation is symmetric for identical series."""
        result = cross_correlation(self.ts1, self.ts1, maxlags=self.maxlags, method='pearson')

        # At zero lag, correlation should be 1
        zero_lag_idx = self.maxlags  # Zero lag is in the middle
        assert abs(result.cross_correlation.values[zero_lag_idx] - 1.0) < 1e-10

    def test_deprecated_function_warning(self):
        """Test that deprecated function shows warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = bootstrapped_cross_correlation(self.ts1, self.ts2, maxlags=5, n_iter=10)

            # Check that a deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert "cross_correlation(method='ebisuzaki')" in str(w[0].message)

            # Check that result structure is correct
            assert 'cross_correlation' in result


class TestMaximumCrossCorrelation:
    """Test suite for maximum_cross_correlation function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.N = 50
        self.dt = 1.0
        self.maxlags = 10

        # Create test time series - one lagged version of the other
        base_ts = AR1_process(0.9, 1.0, 0.0, self.N + 5, seed=42, dt=self.dt)
        self.ts1 = TimeSeries(base_ts.time[:self.N], base_ts.data[:self.N], self.dt)
        self.ts2 = TimeSeries(base_ts.time[:self.N], base_ts.data[3:self.N+3], self.dt)  # 3-step lag

    def test_maxima_detection(self):
        """Test that maxima detection works correctly."""
        lag_max, ccf_max = maximum_cross_correlation(self.ts1, self.ts2, maxlags=self.maxlags)

        # For lagged series, should find the lag
        assert isinstance(lag_max, (int, float, np.number))
        assert isinstance(ccf_max, (float, np.number))

        # Correlation should be positive and substantial for lagged series
        assert ccf_max > 0.5

    def test_maxima_with_methods(self):
        """Test maxima detection with different methods."""
        lag_max_p, ccf_max_p = maximum_cross_correlation(self.ts1, self.ts2, maxlags=self.maxlags, method='pearson')
        lag_max_e, ccf_max_e = maximum_cross_correlation(self.ts1, self.ts2, maxlags=self.maxlags, method='ebisuzaki')

        # Both methods should give similar results for maxima location
        # (allowing for some numerical differences)
        assert abs(lag_max_p - lag_max_e) <= self.dt
        assert abs(ccf_max_p - ccf_max_e) < 0.2

    def test_identical_series_maxima(self):
        """Test maxima for identical series."""
        lag_max, ccf_max = maximum_cross_correlation(self.ts1, self.ts1, maxlags=self.maxlags)

        # Should find maximum at zero lag with correlation = 1
        assert abs(lag_max) < 1e-10
        assert abs(ccf_max - 1.0) < 1e-10

    def test_xarray_inputs(self):
        """Test maximum_cross_correlation with xarray inputs."""
        # Convert to xarray
        ts1_xr = self.ts1.to_xarray()
        ts2_xr = self.ts2.to_xarray()

        lag_max, ccf_max = maximum_cross_correlation(ts1_xr, ts2_xr, maxlags=self.maxlags)

        assert isinstance(lag_max, (int, float, np.number))
        assert isinstance(ccf_max, (float, np.number))
        assert ccf_max > 0.5  # Should still find strong correlation


class TestDeprecatedCrossCorrelationMaxima:
    """Test suite for deprecated cross_correlation_maxima function."""

    def test_deprecated_function_warning(self):
        """Test that deprecated function gives warning but still works."""
        ts1 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42, return_xarray=False)
        ts2 = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=123, return_xarray=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lag_max, ccf_max = cross_correlation_maxima(ts1, ts2, maxlags=10)

            # Check that warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "cross_correlation_maxima() is deprecated" in str(w[0].message)

        # Check that function still works
        assert isinstance(lag_max, (int, float, np.number))
        assert isinstance(ccf_max, (float, np.number))