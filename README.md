# SpectralCorr

<img width="989" height="396" alt="output" src="https://github.com/user-attachments/assets/e27730ac-2e9d-4d30-b918-cc7c63e14d92" />

Power-spectrum based correlation significance testing for autocorrelated time series. This Python package implements a non-parametric correlation test that utilizes randomly generated time series with the appropriate power spectra (Ebisuzaki, 1997). The Ebisuzaki correlation test can be applied here to test both lagged and non-lagged correlations. 


## Installation

```bash
$ pip install git+https://github.com/anthonymeza/SpectralCorr.git@main
```

## Usage

Here's a quick example to get you started:

```python
import numpy as np
from SpectralCorr import AR1_process, cross_correlation

# Generate two AR(1) time series
ts1 = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=500, seed=42)
ts2 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=500, seed=123)

# Compute cross-correlation with Pearson method (no autocorrelation in timeseries)
result_pearson = cross_correlation(ts1, ts2, maxlags=50, method='pearson')

# Or use the Ebisuzaki method for robust significance testing for autocorrelated timeseries 
result_ebisuzaki = cross_correlation(ts1, ts2, maxlags=50, method='ebisuzaki', n_iter=1000)

# Results are returned as xarray Datasets
print(result_ebisuzaki.cross_correlation)
print(result_ebisuzaki.cross_correlation_pvalue)
```

For more examples, see the Jupyter notebooks in the `notebook_examples/` directory.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`SpectralCorr` was created by Anthony Meza. It is licensed under the terms of the MIT license.

## Credits

`SpectralCorr` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## References 

[Ebisuzaki, W. (1997). A method to estimate the statistical significance of a correlation when the data are serially correlated. Journal of Climate, 10(9), 2147–2153. https://doi.org/10.1175/1520-0442(1997)010&#60;2147:amtets&#62;2.0.co;2](https://doi.org/10.1175/1520-0442(1997)010%3C2147:AMTETS%3E2.0.CO;2)
