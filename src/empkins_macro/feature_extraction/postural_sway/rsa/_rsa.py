# Module for rotary spectral analysis (RSA) of postural cop data.
from typing import Self

import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
from empkins_io.datasets.d03.macro_ap03 import MacroBaseDataset
from tpcp import Algorithm

LOWPASS_CUTOFF = 0.5  # Hz
BANDPASS_CUTOFF = (0.5, 20)  # Hz
FILTER_ORDER = 4


class RotarySpectralAnalysis(Algorithm):
    """
    Rotary Spectral Analysis (RSA) for postural cop data.

    This algorithm calculates directional power spectral density components
    based on x and y cop data using a rotary spectrum method.
    """
    _action_methods = "run"

    rsa_data_: pd.DataFrame

    def __init__(self, lowpass_cutoff=LOWPASS_CUTOFF, bandpass_cutoff=BANDPASS_CUTOFF, filter_order=FILTER_ORDER):
        """
        Initializes the RotarySpectralAnalysis algorithm.
        """
        self.lowpass_cutoff = lowpass_cutoff
        self.bandpass_cutoff = bandpass_cutoff
        self.filter_order = filter_order
        self.rsa_data_ = pd.DataFrame()

    def run(self, data: pd.DataFrame) -> Self:
        """
        Run RSA on input data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing 'x' and 'y' columns of postural sway.

        Returns
        -------
        Self
            Returns the fitted object with RSA results stored in `rsa_data_`.
        """

        cop_x = data[('cop', 'both', 'total', 'x')].values
        cop_y = data[('cop', 'both', 'total', 'y')].values
        fsamp = MacroBaseDataset.SAMPLING_RATE_ZEBRIS

        freqs_pos, psd_plus, psd_minus = self._calc_power_spectral_density(cop_x, cop_y, fsamp)

        self.rsa_data_ = pd.DataFrame({
            'frequency': freqs_pos,
            'PSD_plus': psd_plus,
            'PSD_minus': psd_minus
        })
        return self

    @staticmethod
    def _calc_power_spectral_density(x: np.ndarray, y: np.ndarray, fsamp: float) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate rotary PSD components from x and y sway data.
        """
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        x, y = x - np.nanmean(x), y - np.nanmean(y)
        t = len(x) / fsamp

        sos_low = butter(FILTER_ORDER, LOWPASS_CUTOFF, 'low', fs=fsamp, output='sos')
        sos_band = butter(FILTER_ORDER, BANDPASS_CUTOFF, 'bandpass', fs=fsamp, output='sos')
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

        psd_plus = (np.abs(w_plus[pos_mask]) ** 2) / t
        psd_minus = (np.abs(w_minus[pos_mask]) ** 2) / t

        """
        kernel_size = min(60, len(psd_plus))
        kernel = np.ones(kernel_size) / kernel_size
        psd_plus = np.convolve(psd_plus, kernel, mode='same')
        psd_minus = np.convolve(psd_minus, kernel, mode='same')
        """
        return freqs_pos, psd_plus, psd_minus
