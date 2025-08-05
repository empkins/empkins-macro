# Module for rotary spectral analysis (RSA) of postural cop data.
from typing import Self

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tpcp import Algorithm


class RotarySpectralAnalysis(Algorithm):
    """
    Rotary Spectral Analysis (RSA) for postural cop data.

    This algorithm calculates directional power spectral density components
    based on x and y cop data using a rotary spectrum method.
    """
    _action_methods = "run"

    rsa_data_: pd.DataFrame

    def __init__(self, lowpass_cutoff_hz=0.5, bandpass_cutoff_hz=(0.5, 20), filter_order=4):
        """
        Initializes the RotarySpectralAnalysis algorithm.
        """
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.bandpass_cutoff_hz = bandpass_cutoff_hz
        self.filter_order = filter_order
        self.rsa_data_ = pd.DataFrame()

    def run(self, data: pd.DataFrame, sampling_rate_hz) -> Self:
        """
        Run RSA on input data.

        Parameters
        ----------
        sampling_rate_hz: float
            Sampling frequency of the input data.
        data : pd.DataFrame
            DataFrame containing 'x' and 'y' columns of postural sway.

        Returns
        -------
        Self
            Returns the fitted object with RSA results stored in `rsa_data_`.
        """
        # TODO: check that data format is correct
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not all(col in data.columns for col in [('cop', 'both', 'total', 'x'), ('cop', 'both', 'total', 'y')]):
            raise ValueError("Data must contain 'cop' columns for 'x' and 'y' sway data.")

        cop_x = data[('cop', 'both', 'total', 'x')].values
        cop_y = data[('cop', 'both', 'total', 'y')].values

        self.rsa_data = self._calc_power_spectral_density(cop_x, cop_y, sampling_rate_hz, self.lowpass_cutoff_hz,
                                                          self.bandpass_cutoff_hz, self.filter_order)

        return self

    @staticmethod
    def _calc_power_spectral_density(x: np.ndarray, y: np.ndarray, fsamp: float, lowpass_cutoff, bandpass_cutoff,
                                     filter_order) -> pd.DataFrame:
        """
        Calculate the power spectral density (PSD) of x and y data using rotary spectral analysis.
        Parameters
        ----------
        x : np.ndarray
            The x component of the postural sway data.
        y : np.ndarray
            The y component of the postural sway data.
        fsamp : float
            Sampling frequency of the input data.
        lowpass_cutoff : float
            Cutoff frequency for low-pass filter in Hz.
        bandpass_cutoff : tuple
            Cutoff frequencies for band-pass filter in Hz (low, high).
        filter_order : int
            Order of the Butterworth filter to be applied.
        Returns
        -------
        pd.DataFrame
            DataFrame containing the frequency (index), PSD for clockwise (cw) and counterclockwise (cc)

        """


        t = len(x) / fsamp

        sos_low = butter(filter_order, lowpass_cutoff, 'low', fs=fsamp, output='sos')
        sos_band = butter(filter_order, bandpass_cutoff, 'bandpass', fs=fsamp, output='sos')
        x = sosfiltfilt(sos_low, x)
        x = sosfiltfilt(sos_band, x)
        y = sosfiltfilt(sos_low, y)
        y = sosfiltfilt(sos_band, y)

        freqs = np.fft.fftfreq(len(x), d=1 / fsamp)
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]

        x = np.fft.fft(x)
        y = np.fft.fft(y)
        w_plus = 0.5 * (x - 1j * y)
        w_minus = 0.5 * (x + 1j * y)

        psd_cw = (np.abs(w_plus[pos_mask]) ** 2) / t
        psd_ccw = (np.abs(w_minus[pos_mask]) ** 2) / t


        # TODO: Consider applying a smoothing filter to the PSDs if needed.
        """
        kernel_size = min(60, len(psd_plus))
        kernel = np.ones(kernel_size) / kernel_size
        psd_plus = np.convolve(psd_plus, kernel, mode='same')
        psd_minus = np.convolve(psd_minus, kernel, mode='same')
        """
        data = pd.DataFrame({
            'frequency_hz': freqs_pos,
            'PSD_cw': psd_cw,
            'PSD_ccw': psd_ccw
        }).set_index('frequency_hz')
        return data
