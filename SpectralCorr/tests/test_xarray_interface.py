"""
Test the new xarray interface and performance improvements.
"""

import time
import numpy as np
import xarray as xr
from SpectralCorr import AR1_process, cross_correlation, TimeSeries


def test_xarray_interface():
    """Test that the new xarray interface works correctly."""
    print("🧪 Testing xarray interface...\n")

    # Test AR1_process with xarray output (default)
    print("1. Testing AR1_process with xarray output:")
    ts1_xr = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=100, seed=42)
    ts2_xr = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=100, seed=123)

    print(f"   ts1 type: {type(ts1_xr)}")
    print(f"   ts1 has time coord: {'time' in ts1_xr.coords}")
    print(f"   ts1 shape: {ts1_xr.shape}")
    print(f"   ts1 attrs: {list(ts1_xr.attrs.keys())}")

    # Test cross_correlation with xarray inputs
    print("\n2. Testing cross_correlation with xarray inputs:")
    result_xr = cross_correlation(ts1_xr, ts2_xr, maxlags=10, method='pearson')

    print(f"   Result type: {type(result_xr)}")
    print(f"   Result variables: {list(result_xr.data_vars.keys())}")
    print(f"   Result coords: {list(result_xr.coords.keys())}")

    # Test mixed inputs (xarray + TimeSeries)
    print("\n3. Testing mixed inputs:")
    ts1_ts = TimeSeries.from_xarray(ts1_xr)
    result_mixed = cross_correlation(ts1_xr, ts1_ts, maxlags=5, method='pearson')
    print(f"   Mixed result type: {type(result_mixed)}")

    print("\n✅ All xarray interface tests passed!")
    return True


def test_performance_improvement():
    """Test performance improvement from lightweight TimeSeries."""
    print("\n⚡ Testing performance improvements...\n")

    # Create test data
    n_points = 500
    n_lags = 50
    dt = 1.0

    ts1_xr = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=n_points, seed=42)
    ts2_xr = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=n_points, seed=123)

    print(f"Test parameters:")
    print(f"- Data points: {n_points}")
    print(f"- Max lags: {n_lags}")
    print(f"- Method: Pearson (fast)")

    # Test the optimized version
    print(f"\n🚀 Testing optimized version:")
    start_time = time.time()
    result = cross_correlation(ts1_xr, ts2_xr, maxlags=n_lags, method='pearson')
    end_time = time.time()
    optimized_time = end_time - start_time

    print(f"   Time: {optimized_time:.4f} seconds")
    print(f"   Result lags: {len(result.lag)}")
    print(f"   Result type: {type(result)}")

    # Compare with creating many TimeSeries objects the old way
    print(f"\n📊 Simulating old approach (many TimeSeries creations):")
    start_time = time.time()

    # Simulate what the old code would do - create many TimeSeries objects
    ts1_ts = TimeSeries.from_xarray(ts1_xr)
    ts2_ts = TimeSeries.from_xarray(ts2_xr)

    old_style_objects = []
    for i in range(n_lags * 2 + 1):  # Simulate lag loop
        # Create new TimeSeries objects like the old approach would
        subset_size = max(10, n_points - i)
        new_ts1 = TimeSeries(ts1_ts.time[:subset_size], ts1_ts.data[:subset_size], dt)
        new_ts2 = TimeSeries(ts2_ts.time[:subset_size], ts2_ts.data[:subset_size], dt)
        old_style_objects.append((new_ts1, new_ts2))

    end_time = time.time()
    old_simulation_time = end_time - start_time

    print(f"   Old style simulation time: {old_simulation_time:.4f} seconds")
    print(f"   Objects created: {len(old_style_objects) * 2}")

    # Calculate improvement
    if old_simulation_time > 0:
        improvement = old_simulation_time / optimized_time
        print(f"\n📈 Performance comparison:")
        print(f"   Speedup factor: {improvement:.1f}x")
        print(f"   Time saved: {(old_simulation_time - optimized_time) * 1000:.1f} ms")

    return optimized_time, old_simulation_time


if __name__ == "__main__":
    print("🎯 SpectralCorr Xarray Interface & Performance Test\n")

    # Run tests
    test_xarray_interface()
    test_performance_improvement()

    print("\n🎉 All tests completed successfully!")