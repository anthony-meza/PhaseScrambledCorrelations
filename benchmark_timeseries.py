"""
Benchmark script to test TimeSeries creation performance.
"""

import time
import numpy as np
from SpectralCorr import TimeSeries


def benchmark_timeseries_creation():
    """Benchmark TimeSeries object creation."""

    # Test parameters
    n_points = 1000
    n_iterations = 100
    dt = 1.0

    # Generate test data
    time_array = np.arange(n_points) * dt
    data_array = np.random.randn(n_points)

    print(f"Benchmarking TimeSeries creation:")
    print(f"- Data size: {n_points} points")
    print(f"- Iterations: {n_iterations}")
    print(f"- Total TimeSeries objects: {n_iterations}")

    # Benchmark current implementation
    start_time = time.time()

    time_series_objects = []
    for i in range(n_iterations):
        # Simulate what happens in correlation loop
        subset_size = n_points - i  # Decreasing size like in lag correlation
        if subset_size > 10:  # Ensure reasonable size
            ts = TimeSeries(time_array[:subset_size], data_array[:subset_size], dt)
            time_series_objects.append(ts)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n📊 Results:")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average per object: {(total_time / n_iterations) * 1000:.2f} ms")
    print(f"Objects created: {len(time_series_objects)}")

    # Test xarray conversion time (on-demand)
    start_time = time.time()
    for ts in time_series_objects[:10]:  # Test first 10
        _ = ts.to_xarray().values  # Convert to xarray and access data
    access_time = time.time() - start_time
    print(f"Xarray conversion time (10 objects): {access_time * 1000:.2f} ms")

    return total_time, len(time_series_objects)


def benchmark_numpy_only():
    """Benchmark using numpy arrays only (no TimeSeries)."""

    n_points = 1000
    n_iterations = 100
    dt = 1.0

    time_array = np.arange(n_points) * dt
    data_array = np.random.randn(n_points)

    print(f"\n🔥 Numpy-only baseline:")

    start_time = time.time()

    arrays = []
    for i in range(n_iterations):
        subset_size = n_points - i
        if subset_size > 10:
            # Just store the arrays, no TimeSeries wrapper
            t_sub = time_array[:subset_size]
            d_sub = data_array[:subset_size]
            arrays.append((t_sub, d_sub, dt))

    end_time = time.time()
    numpy_time = end_time - start_time

    print(f"Numpy time: {numpy_time:.4f} seconds")
    print(f"Average per array: {(numpy_time / n_iterations) * 1000:.2f} ms")

    return numpy_time


if __name__ == "__main__":
    print("⏱️  TimeSeries Performance Benchmark\n")

    # Run benchmarks
    ts_time, n_objects = benchmark_timeseries_creation()
    numpy_time = benchmark_numpy_only()

    # Compare
    overhead = ts_time / numpy_time if numpy_time > 0 else float('inf')
    print(f"\n📈 Performance Comparison:")
    print(f"TimeSeries overhead: {overhead:.1f}x slower than numpy")
    print(f"Overhead per object: {((ts_time - numpy_time) / n_objects) * 1000:.2f} ms")