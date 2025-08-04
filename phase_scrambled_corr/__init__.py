"""
phase_scrambled_corr package initialization.
Imports core modules and sets version.
"""
"""
Main module for phase-scrambled correlation analysis.

Imports core functions for correlation, time series simulation, and FFT utilities.
"""


# read version from installed package
from importlib.metadata import version
from .version import __version__

# import numpy as np
# from scipy.stats import pearsonr
# from statsmodels.distributions.empirical_distribution import ECDF

from .correlations import *
from .ar_processes import *
from .fft_utils import *