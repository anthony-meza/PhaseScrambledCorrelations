# read version from installed package
from importlib.metadata import version
__version__ = version("phase_scrambled_corr")


from .correlations       import *
from .example_time_series  import *
from .fft_utils         import *

# from phase_scrambled_corr import * 
# from example_time_series import * 