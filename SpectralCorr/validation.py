"""
Input validation utilities for time series analysis functions.
"""

import numpy as np
import xarray as xr


def validate_dataarray_inputs(ts1, ts2, function_name="function"):
    """
    Validate xarray DataArray inputs for consistency and data quality.

    Parameters
    ----------
    ts1, ts2 : xarray.DataArray
        Input time series DataArrays to validate
    function_name : str
        Name of the calling function for error messages

    Returns
    -------
    ts1, ts2 : xarray.DataArray
        Validated time series DataArrays (same as input)

    Raises
    ------
    ValueError
        If validation fails for any reason

    Notes
    -----
    This function validates:
    - Input types are xarray DataArrays
    - Both DataArrays have 'time' coordinate
    - Time coordinates are numeric (convertible to float)
    - Series have equal lengths
    - No NaN values in time coordinates or data
    - Time coordinates are aligned (identical values)
    - Time steps are equally spaced
    """
    # Ensure inputs are xarray DataArrays
    if not isinstance(ts1, xr.DataArray):
        raise ValueError(f"{function_name}: ts1 must be an xarray.DataArray, got {type(ts1)}")
    if not isinstance(ts2, xr.DataArray):
        raise ValueError(f"{function_name}: ts2 must be an xarray.DataArray, got {type(ts2)}")

    # Validate time coordinates exist
    if "time" not in ts1.coords:
        raise ValueError(f"{function_name}: ts1 must have 'time' coordinate")
    if "time" not in ts2.coords:
        raise ValueError(f"{function_name}: ts2 must have 'time' coordinate")

    time1 = ts1.coords["time"].values
    time2 = ts2.coords["time"].values

    # Check that time coordinates are numeric (can be converted to float)
    try:
        time1_float = time1.astype(float)
        time2_float = time2.astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{function_name}: Time coordinates must be numeric (convertible to float): {e}")

    # Check that both series have the same length
    if len(time1) != len(time2):
        raise ValueError(f"{function_name}: Time series must have same length: ts1 has {len(time1)}, ts2 has {len(time2)}")

    # Check for NaN values in time coordinates (do this before allclose check)
    if np.any(np.isnan(time1_float)) or np.any(np.isnan(time2_float)):
        raise ValueError(f"{function_name}: Time coordinates must not contain NaN values")

    # Check that time coordinates are aligned (same values)
    if not np.allclose(time1_float, time2_float, rtol=1e-10):
        raise ValueError(f"{function_name}: Time coordinates must be aligned (same time values for both series)")

    # Check for NaN values in data
    data1_values = ts1.values
    data2_values = ts2.values
    if np.any(np.isnan(data1_values)):
        raise ValueError(f"{function_name}: ts1 data must not contain NaN values")
    if np.any(np.isnan(data2_values)):
        raise ValueError(f"{function_name}: ts2 data must not contain NaN values")

    # Validate time steps are equally spaced
    if len(time1) > 1:
        dt_values = np.diff(time1_float)
        if not np.allclose(dt_values, dt_values[0], rtol=1e-10):
            raise ValueError(f"{function_name}: Time step (dt) must be constant throughout the series")

    return ts1, ts2