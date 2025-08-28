import numpy as np
import xarray as xr

class TimeSeries:
    """
    Container for time series data with time and value arrays.

    Attributes
    ----------
    time : ndarray
        Array of time points.
    data : ndarray
        Array of data values.
    da : xarray.DataArray
        DataArray representation of the time series.
    """

    def __init__(self, time, data, dt):
        """
        Initialize a TimeSeries object.

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
        self.da = xr.DataArray(
            self.data,
            coords={"time": self.time},
            dims=["time"],
            name="timeseries",
            attrs={
                "description": "Original time series data",
                "dt": self.dt,
                "length": self.N
            }
        )

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
        Return the time series as an xarray.DataArray.

        Returns
        -------
        xarray.DataArray
            DataArray representation of the time series.
        """
        return self.da