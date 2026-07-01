import numpy as np
from scipy.signal import butter, lfilter


def snr_scaling(glitch, snr, srate=4096, psd=None):
    """Scale a signal to a target optimal SNR.

    Parameters
    ----------
    glitch : array_like
        Time-domain signal(s) to scale (last axis is time).
    snr : float or None
        Target optimal SNR. ``None`` returns *glitch* unchanged.
    srate : float
        Sample rate in Hz.
    psd : array_like or None
        One-sided PSD Sn(f), evaluated at ``np.fft.rfftfreq(glitch.shape[-1], 1/srate)``.
        ``None`` (default) assumes flat Sn(f)=1 — i.e. *glitch* is already in the
        whitened frame, where the noise it will be injected into is also whitened.
        Pass the detector PSD to scale a signal for injection into un-whitened
        (coloured) strain instead.
    """
    glitch = np.asarray(glitch)
    if snr is None:
        return glitch
    df = srate / glitch.shape[-1]
    glitch_FD = np.fft.rfft(glitch, axis=-1) / srate
    power = np.multiply(np.conj(glitch_FD), glitch_FD).real
    if psd is not None:
        power = power / np.asarray(psd)
    true_sigma_sq = 4.0 * df * np.sum(power, axis=-1)
    # Guard against zero-energy inputs (e.g. constant waveform after mean subtraction).
    # Injecting a zero-amplitude glitch is a no-op, so return zeros rather than ±inf.
    scale = np.where(true_sigma_sq > 0, snr / np.sqrt(np.maximum(true_sigma_sq, 1e-300)), 0.0)
    return (glitch.T * scale).T


def whitened_snr_scaling(glitch, snr, srate=4096):
    """Scale a glitch signal to the target SNR in the whitened frame."""
    return snr_scaling(glitch, snr, srate=srate, psd=None)


def quality_factor_conversion(Q, f_0):
    """Convert quality factor Q and central frequency f_0 to decay time tau."""
    tau = Q / (np.sqrt(2) * np.pi * f_0)
    return tau


def rescale(x):
    """Rescale each row of x to the range [-1, 1]."""
    abs_max = np.max(np.abs(x), axis=1, keepdims=True)
    return x / abs_max


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    d, c = butter(order, normal_cutoff, btype="high", analog=False)
    return d, c


def butter_filter(data, fs, order=5):
    """Apply a bandpass (20–1024 Hz) Butterworth filter to data."""
    low_cutoff = 1024
    high_cutoff = 20
    b, a = butter_lowpass(low_cutoff, fs, order=order)
    low_filtered = lfilter(b, a, data)
    d, c = butter_highpass(high_cutoff, fs, order=order)
    return lfilter(d, c, low_filtered)


def custom_whiten(self, psd, low_frequency_cutoff=None, return_psd=False, **kwds):
    """
    Return a whitened PyCBC TimeSeries.

    This function is designed to be used with a PyCBC TimeSeries instance
    (as a monkey-patched method). Pass ``self`` as the TimeSeries object.

    Parameters
    ----------
    psd : FrequencySeries
        The power spectral density used for whitening.
    low_frequency_cutoff : float, optional
        Low frequency cutoff for the inverse spectrum truncation.
    return_psd : bool, optional
        If True, return the PSD alongside the whitened data.

    Returns
    -------
    white : TimeSeries
        The whitened time series.
    psd : FrequencySeries, optional
        The PSD used (only returned if ``return_psd=True``).
    """
    white = (self.to_frequencyseries() / psd**0.5).to_timeseries()

    if return_psd:
        return white, psd

    return white


def generate_gaussian_noise(mean, std_dev, num_samples, sample_shape):
    """Generate Gaussian noise samples as a numpy array."""
    return np.random.normal(loc=mean, scale=std_dev, size=(num_samples, *sample_shape))
