"""
Test that the notebook examples work with the new xarray interface.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from SpectralCorr import AR1_process, cross_correlation, maximum_cross_correlation

if __name__ == "__main__":
    print("🔬 Testing notebook compatibility...")

    try:
        # Test the code from AR1_example.ipynb
        print("\n1. Testing AR1_example.ipynb code:")

        N = 500               # number of time points
        dt = 1.0              # time step = 1 month
        fs = 1 / dt           # sampling frequency = 1 / 1 month

        # AR(1) parameters
        rho_x = 0.9
        rho_y = 0.9
        noise_std = 1.0

        # Generate time series (should return xarray by default)
        ts1 = AR1_process(rho_x, noise_std, 1, N, seed=42, dt=dt)
        ts2 = AR1_process(rho_y, noise_std, 1, N, seed=31, dt=dt)

        print(f"   ts1 type: {type(ts1)}")
        print(f"   ts1 has time coord: {'time' in ts1.coords}")
        print(f"   ts1 shape: {ts1.shape}")

        ccf_maxlag = 36
        ccf_ds = cross_correlation(ts1, ts2, maxlags=ccf_maxlag, method='pearson')
        lag_max, ccf_max = maximum_cross_correlation(ts1, ts2, maxlags=ccf_maxlag)

        print(f"   Cross-correlation result type: {type(ccf_ds)}")
        print(f"   Cross-correlation variables: {list(ccf_ds.data_vars.keys())}")
        print(f"   Lag max: {lag_max}, CCF max: {ccf_max:.3f}")

        # Test Ebisuzaki method
        print("\n2. Testing Ebisuzaki method (small n_iter for speed):")
        n_iter = 100  # Small for testing
        bccf_ds = cross_correlation(ts1, ts2, maxlags=ccf_maxlag, method='ebisuzaki',
                                   n_iter=n_iter, return_distributions=True, detrend=True)

        print(f"   Ebisuzaki result type: {type(bccf_ds)}")
        print(f"   Has distributions: {'cross_correlation_distribution' in bccf_ds}")

        # Test plotting (basic check - don't actually display)
        print("\n3. Testing plotting compatibility:")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Should work with xarray plotting
        ts1.plot(ax=axes[0], label='Time Series 1')
        ts2.plot(ax=axes[0], label='Time Series 2')
        axes[0].legend()

        ccf_ds["cross_correlation"].plot(ax=axes[1])
        plt.close(fig)  # Close to save memory

        print("   Plotting works with xarray interface ✅")

        print("\n✅ All notebook compatibility tests passed!")

    except Exception as e:
        print(f"\n❌ Notebook compatibility test failed: {e}")
        import traceback
        traceback.print_exc()