"""
SpectralCorr: Power-spectrum based significance testing for autocorrelated time series.

This package implements phase randomization methods for assessing the statistical
significance of cross-correlations in time series data, following Ebisuzaki (1997).

The main functionality includes:
- AR(1) time series simulation
- Cross-correlation analysis with various significance tests
- Phase scrambling bootstrap methods
- FFT-based surrogate generation

References
----------
Ebisuzaki, W. (1997). A method to estimate the statistical significance of a 
correlation when the data are serially correlated. Journal of Climate, 10(9), 
2147-2153.
"""

from .version import __version__

# Core functionality imports
from .ar_processes import *
from .correlations import *
from .fft_utils import *
from .timeseries import TimeSeries
from .plotting import (
    plot_significant_correlations,
    plot_conf_intervals,
    plot_cross_correlation
)

# Deprecated functions for backward compatibility
from .deprecated import cross_correlation_maxima, shift_maximally_correlated, bootstrapped_cross_correlation

__all__ = [
    "__version__",
    # Time series classes
    "TimeSeries",
    # AR process simulation
    "AR1_process",
    # Correlation functions
    "correlation",
    "cross_correlation",
    "maximum_cross_correlation",
    "cross_correlation_maxima",  # Deprecated - use maximum_cross_correlation
    "align_at_maximum_correlation",
    "shift_maximally_correlated",  # Deprecated - use align_at_maximum_correlation
    "bootstrap_correlation",
    # Legacy (deprecated)
    "bootstrapped_cross_correlation",  # Use cross_correlation(method='ebisuzaki') instead
    # Phase scrambling
    "phase_scrambled",
    # FFT utilities
    "real_fft",
    "simple_detrend",
    # Plotting
    "plot_significant_correlations",
    "plot_conf_intervals",
    "plot_cross_correlation",
]