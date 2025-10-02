"""
Tests for the align_at_maximum_correlation function.
"""

import numpy as np
import pytest
import warnings
import xarray as xr
from SpectralCorr import AR1_process, align_at_maximum_correlation, shift_maximally_correlated, TimeSeries


class TestAlignAtMaximumCorrelation:
    """Test suite for align_at_maximum_correlation function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.N = 100
        self.dt = 1.0
        self.maxlags = 20

        # Create a base time series
        base_ts = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=self.N + 20, seed=42)

        # Create two series with known lag offset - ts2 is ts1 shifted by lag_offset
        self.lag_offset = 5
        self.ts1 = xr.DataArray(
            base_ts.values[:self.N],
            coords={'time': base_ts.coords['time'].values[:self.N]},
            dims=['time']
        )
        # ts2 is a lagged version of ts1, so they should be highly correlated when aligned
        self.ts2 = xr.DataArray(
            base_ts.values[self.lag_offset:self.N+self.lag_offset],
            coords={'time': base_ts.coords['time'].values[:self.N]},
            dims=['time']
        )

    def test_basic_alignment_xarray_output(self):
        """Test basic alignment functionality with xarray output."""
        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=self.maxlags
        )

        # Check return types
        assert isinstance(ts1_aligned, xr.DataArray)
        assert isinstance(ts2_aligned, xr.DataArray)

        # Check that aligned series have same length
        assert len(ts1_aligned) == len(ts2_aligned)

        # Check that time coordinates are identical
        assert np.allclose(ts1_aligned.coords["time"], ts2_aligned.coords["time"])

        # Since ts2 is a shifted version of ts1, alignment should give high correlation
        correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
        assert correlation > 0.988  # Should be very high for same data shifted

    def test_basic_alignment_functionality(self):
        """Test basic alignment functionality - always returns xarray."""
        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=self.maxlags
        )

        # Check return types (always xarray)
        assert isinstance(ts1_aligned, xr.DataArray)
        assert isinstance(ts2_aligned, xr.DataArray)

        # Check that aligned series have same length
        assert len(ts1_aligned) == len(ts2_aligned)

        # Check that time coordinates are identical
        assert np.allclose(ts1_aligned.coords["time"], ts2_aligned.coords["time"])

        # Since ts2 is a shifted version of ts1, alignment should give high correlation
        correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
        assert correlation > 0.988  # Should be very high for same data shifted

    def test_xarray_inputs(self):
        """Test alignment with xarray DataArray inputs."""
        # Already using xarray from setup_method
        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=self.maxlags
        )

        # Check return types (should be xarray by default)
        assert isinstance(ts1_aligned, xr.DataArray)
        assert isinstance(ts2_aligned, xr.DataArray)

        # Check alignment worked
        correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
        assert correlation > 0.988

    def test_mixed_inputs(self):
        """Test alignment with two xarray inputs."""
        # Both inputs are already xarray from setup_method
        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=self.maxlags
        )

        assert isinstance(ts1_aligned, xr.DataArray)
        assert isinstance(ts2_aligned, xr.DataArray)

        correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
        assert correlation > 0.98

    def test_identical_series_alignment(self):
        """Test alignment with identical time series."""
        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, self.ts1, maxlags=self.maxlags
        )

        # Should be perfectly correlated
        correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
        assert np.isclose(correlation, 1.0, atol=1e-10)

        # Should have same length as input (no trimming needed)
        assert len(ts1_aligned) == len(ts2_aligned)

    def test_negative_lag_alignment(self):
        """Test alignment when optimal lag is negative."""
        # Create series where ts2 leads ts1
        ts2_leads = xr.DataArray(
            np.concatenate([self.ts1.values[3:], np.zeros(3)]),  # ts2 leads by 3 steps
            coords={'time': self.ts1.coords['time'].values},
            dims=['time']
        )

        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, ts2_leads, maxlags=self.maxlags
        )

        # Should still achieve reasonable correlation (less than perfect due to modification)
        correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
        assert correlation > 0.3  # Lower expectation due to data modification

    def test_maxlags_parameter(self):
        """Test behavior with different maxlags values."""
        # Test with small maxlags (smaller than the actual lag offset of 5)
        ts1_aligned_small, ts2_aligned_small = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=3
        )

        # Test with large maxlags (larger than the actual lag offset of 5)
        ts1_aligned_large, ts2_aligned_large = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=30
        )

        # Both should work, but large maxlags should find better alignment
        corr_small = np.corrcoef(ts1_aligned_small.values, ts2_aligned_small.values)[0, 1]
        corr_large = np.corrcoef(ts1_aligned_large.values, ts2_aligned_large.values)[0, 1]

        # Small maxlags might not find the true lag (5), so just check it's valid
        assert not np.isnan(corr_small)
        # Large maxlags should find the true lag and get very high correlation
        assert corr_large > 0.9
        # Large maxlags should find better alignment than small maxlags
        assert corr_large >= corr_small

    def test_default_maxlags(self):
        """Test alignment with default maxlags (None)."""
        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            self.ts1, self.ts2, maxlags=None
        )

        # Should still work with default maxlags and produce valid results
        assert len(ts1_aligned) == len(ts2_aligned)
        assert len(ts1_aligned) > 0

        # Check correlation only if we have enough data points
        if len(ts1_aligned) >= 2:
            correlation = np.corrcoef(ts1_aligned.values, ts2_aligned.values)[0, 1]
            # With very large maxlags, correlation might be lower but should be valid
            assert not np.isnan(correlation) or len(ts1_aligned) < 2

    def test_short_series_alignment(self):
        """Test alignment with very short time series."""
        short_ts1 = xr.DataArray(
            np.random.randn(10),
            coords={'time': np.arange(10)},
            dims=['time']
        )
        short_ts2 = xr.DataArray(
            np.random.randn(10),
            coords={'time': np.arange(10)},
            dims=['time']
        )

        ts1_aligned, ts2_aligned = align_at_maximum_correlation(
            short_ts1, short_ts2, maxlags=3
        )

        # Should return valid results
        assert isinstance(ts1_aligned, xr.DataArray)
        assert isinstance(ts2_aligned, xr.DataArray)
        assert len(ts1_aligned) == len(ts2_aligned)
        assert len(ts1_aligned) > 0

    def test_alignment_preserves_attributes(self):
        """Test that alignment preserves xarray attributes."""
        # Add some attributes to test preservation
        ts1_with_attrs = self.ts1.copy()
        ts2_with_attrs = self.ts2.copy()
        ts1_with_attrs.attrs['dt'] = self.dt
        ts2_with_attrs.attrs['dt'] = self.dt

        ts1_aligned, ts2_aligned = align_at_maximum_correlation(ts1_with_attrs, ts2_with_attrs)

        # Check that basic attributes are preserved
        assert "dt" in ts1_aligned.attrs
        assert "dt" in ts2_aligned.attrs

    def test_time_coordinate_validation(self):
        """Test validation of time coordinates."""
        # Test missing time coordinate
        data_no_time = xr.DataArray(np.random.randn(50), dims=["x"])

        with pytest.raises(ValueError, match="ts1 must have 'time' coordinate"):
            align_at_maximum_correlation(data_no_time, self.ts2)

        # Test non-numeric time coordinates
        ts_bad_time = xr.DataArray(
            np.random.randn(50),
            coords={"time": ["a", "b", "c"] * 16 + ["d", "e"]},
            dims=["time"]
        )

        with pytest.raises(ValueError, match="Time coordinates must be numeric"):
            align_at_maximum_correlation(self.ts1, ts_bad_time)

        # Test mismatched lengths
        ts_short = xr.DataArray(
            np.random.randn(30),
            coords={"time": np.arange(30)},
            dims=["time"]
        )

        with pytest.raises(ValueError, match="Time series must have same length"):
            align_at_maximum_correlation(self.ts1, ts_short)

        # Test misaligned time coordinates
        ts_misaligned = xr.DataArray(
            np.random.randn(len(self.ts1)),
            coords={"time": np.arange(len(self.ts1)) + 10},  # Offset by 10
            dims=["time"]
        )

        with pytest.raises(ValueError, match="Time coordinates must be aligned"):
            align_at_maximum_correlation(self.ts1, ts_misaligned)

        # Test non-uniform time steps
        irregular_time = np.array([0, 1, 2, 5, 8, 12, 17, 23, 30, 38])  # Irregular spacing
        ts_irregular = xr.DataArray(
            np.random.randn(len(irregular_time)),
            coords={"time": irregular_time},
            dims=["time"]
        )
        ts_irregular2 = xr.DataArray(
            np.random.randn(len(irregular_time)),
            coords={"time": irregular_time},
            dims=["time"]
        )

        with pytest.raises(ValueError, match="Time step \\(dt\\) must be constant"):
            align_at_maximum_correlation(ts_irregular, ts_irregular2)

        # Test NaN values in time coordinates
        time_with_nan = np.arange(50, dtype=float)
        time_with_nan[10] = np.nan
        ts_time_nan = xr.DataArray(
            np.random.randn(50),
            coords={"time": time_with_nan},
            dims=["time"]
        )
        ts_normal = xr.DataArray(
            np.random.randn(50),
            coords={"time": np.arange(50, dtype=float)},
            dims=["time"]
        )

        with pytest.raises(ValueError, match="Time coordinates must not contain NaN values"):
            align_at_maximum_correlation(ts_time_nan, ts_normal)

        # Test NaN values in data
        data_with_nan = np.random.randn(50)
        data_with_nan[15] = np.nan
        ts_data_nan = xr.DataArray(
            data_with_nan,
            coords={"time": np.arange(50, dtype=float)},
            dims=["time"]
        )

        with pytest.raises(ValueError, match="ts1 data must not contain NaN values"):
            align_at_maximum_correlation(ts_data_nan, ts_normal)

        with pytest.raises(ValueError, match="ts2 data must not contain NaN values"):
            align_at_maximum_correlation(ts_normal, ts_data_nan)


class TestDeprecatedShiftMaximallyCorrelated:
    """Test suite for deprecated shift_maximally_correlated function."""

    def test_deprecated_function_warning(self):
        """Test that deprecated function gives warning but still works."""
        ts1_xr = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42)
        ts2_xr = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=123)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ts1_aligned, ts2_aligned = shift_maximally_correlated(ts1_xr, ts2_xr, maxlags=10)

            # Check that warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "shift_maximally_correlated() is deprecated" in str(w[0].message)

        # Check that function still works and returns TimeSeries
        assert isinstance(ts1_aligned, TimeSeries)
        assert isinstance(ts2_aligned, TimeSeries)
        assert ts1_aligned.N == ts2_aligned.N

    def test_deprecated_function_equivalent_results(self):
        """Test that deprecated function gives equivalent results to new function."""
        ts1_xr = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=50, seed=42)
        ts2_xr = AR1_process(rho=0.7, sigma=1.0, y0=0.0, N=50, seed=123)

        # Get results from deprecated function (suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ts1_old, ts2_old = shift_maximally_correlated(ts1_xr, ts2_xr, maxlags=10)

        # Get results from new function (always returns xarray, so convert for comparison)
        ts1_new_xr, ts2_new_xr = align_at_maximum_correlation(ts1_xr, ts2_xr, maxlags=10)
        ts1_new = TimeSeries.from_xarray(ts1_new_xr)
        ts2_new = TimeSeries.from_xarray(ts2_new_xr)

        # Results should be equivalent
        assert np.allclose(ts1_old.data, ts1_new.data)
        assert np.allclose(ts2_old.data, ts2_new.data)
        assert np.allclose(ts1_old.time, ts1_new.time)
        assert np.allclose(ts2_old.time, ts2_new.time)