import numpy as np
from numpy.fft import rfft, rfftfreq, irfft

def real_fft(signal, fs, remove_mean=True, periodogram=False):
    """
    Compute the real-valued FFT of a time series.

    Parameters
    ----------
    signal : array_like
        Input time series data.
    fs : float
        Sampling frequency (Hz).
    remove_mean : bool, default True
        If True, subtract the mean before FFT.
    periodogram : bool, default False
        If True, return 1/freq and FFT magnitudes (excluding zero-frequency term).

    Returns
    -------
    freqs : ndarray
        Array of frequency bins.
    spectrum : ndarray
        FFT of the input signal (complex values) or magnitudes if periodogram.
    """
    signal = np.asarray(signal)
    nt = signal.size
    if remove_mean or periodogram:
        signal = signal - np.mean(signal)
    F = rfft(signal)
    # zero out the DC component
    F[0] = 0 + 0j
    freqs = rfftfreq(nt, 1/fs)

    if periodogram:
        # exclude zero-frequency
        nonzero = freqs[1:]
        return 1.0 / nonzero, F[1:]
    return freqs, F

