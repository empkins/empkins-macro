# Module for rotary spectral analysis (RSA) of postural sway data.
from typing import Self

import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
from empkins_io.datasets.d03.macro_ap03 import MacroBaseDataset
from tpcp import Algorithm

def calc_power_spectral_density(x, y, fsamp):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    x, y = x - np.nanmean(x), y - np.nanmean(y)
    t = len(x) / fsamp

    sos_low = butter(4, 0.5, 'low', fs=fsamp, output='sos')
    sos_band = butter(4, [0.5, 20], 'bandpass', fs=fsamp, output='sos')
    x = sosfiltfilt(sos_low, x)
    x = sosfiltfilt(sos_band, x)
    y = sosfiltfilt(sos_low, y)
    y = sosfiltfilt(sos_band, y)

    freqs = np.fft.fftfreq(len(x), d=1/fsamp)
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



class RotarySpectralAnalysis(Algorithm):

    _action_methods = "run"

    rsa_data_: pd.DataFrame

    def __init__(self):
        """Initializes the RotarySpectralAnalysis algorithm.
        """
        pass



    def run(self, data: pd.DataFrame) -> Self:
        x = data['x'].values
        y = data['y'].values
        fsamp = MacroBaseDataset.SAMPLING_RATE_ZEBRIS

        freqs_pos, psd_plus, psd_minus = calc_power_spectral_density(x, y, fsamp)
        self.rsa_data_ = pd.DataFrame({
            'frequency': freqs_pos,
            'PSD_plus': psd_plus,
            'PSD_minus': psd_minus
        })
        return self