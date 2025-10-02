import numpy as np
import xarray as xr

class TimeSeries:
    """
    Lightweight container for time series data used in internal computations.

    This class is optimized for performance in internal algorithms and does not
    create xarray objects by default. For user-facing operations, consider using
    xarray DataArrays directly.

    Attributes
    ----------
    time : ndarray
        Array of time points.
    data : ndarray
        Array of data values.
    dt : float
        Time step between consecutive points.
    n : int
        Number of data points.
    """

    def __init__(self, time, data, dt):
        """
        Initialize a lightweight TimeSeries object.

        Parameters
        ----------
        time : array-like
            Time points.
        data : array-like
            Data values.
        dt : float
            Time interval between data points.
        """
        self.time = np.asarray(time)
        self.data = np.asarray(data)
        if len(self.time) != len(self.data):
            raise ValueError("time and data must have the same length")
        self.dt = dt
        self.N = len(self.data)

    @property
    def n(self):
        """
        Return the number of data points in the time series.

        Returns
        -------
        int
            Number of data points.
        """
        return len(self.data)

    def copy(self):
        """
        Return a copy of the TimeSeries object.

        Returns
        -------
        TimeSeries
            A new TimeSeries object with copied data.
        """
        return TimeSeries(self.time.copy(), self.data.copy(), self.dt)

    def to_xarray(self):
        """
        Convert to xarray.DataArray for user-facing output.

        Returns
        -------
        xarray.DataArray
            DataArray representation of the time series.
        """
        return xr.DataArray(
            self.data,
            coords={"time": self.time},
            dims=["time"],
            name="timeseries",
            attrs={
                "description": "Time series data",
                "dt": self.dt,
                "length": self.N
            }
        )

    @classmethod
    def from_xarray(cls, da):
        """
        Create TimeSeries from xarray.DataArray.

        Parameters
        ----------
        da : xarray.DataArray
            Input DataArray with time coordinate.

        Returns
        -------
        TimeSeries
            New TimeSeries object.
        """
        if "time" not in da.coords:
            raise ValueError("DataArray must have 'time' coordinate")

        time = da.coords["time"].values
        data = da.values

        # Infer dt from time coordinate
        if len(time) > 1:
            dt = float(time[1] - time[0])
        else:
            dt = 1.0  # Default for single point

        return cls(time, data, dt)