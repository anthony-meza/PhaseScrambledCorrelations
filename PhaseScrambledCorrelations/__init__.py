"""
PhaseScrambledCorrelations: Phase randomization surrogates for cross-correlation analysis.

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

# __all__ = [
#     "__version__",
#     # Time series classes
#     "TimeSeries",
#     # AR process simulation
#     "AR1_process", 
#     # Correlation functions
#     "cross_correlation",
#     "cross_correlation_maxima", 
#     "bootstrapped_cross_correlation",
#     # FFT utilities
#     "phase_scramble",
#     "power_spectrum", 
#     "coherence_spectrum",
# ]