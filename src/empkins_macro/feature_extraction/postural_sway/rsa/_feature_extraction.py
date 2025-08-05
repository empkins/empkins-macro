# Module for RSA feature extraction of postural cop data.
from typing import Self

import pandas as pd
import numpy as np
from tpcp import Algorithm

class RSAFeatures(Algorithm):
    _action_methods = ("total_power", "mean_frequency",
                       "median_frequency", "mode_frequency",
                       "spectral_skewness", "bandwidth_95",
                       "f25", "f50", "f75")


    feature_data_: pd.DataFrame

    def __init__(self):
        """
        Initializes the RSA Feature Extraction algorithm.
        """
        self.feature_data_ = pd.DataFrame()

    def _extract_features(self, freqs: np.ndarray, psd: np.ndarray) -> dict:
        df = freqs[1] - freqs[0]

        def total_power():
            return np.sum(psd) * df

        def mean_frequency():
            m0 = np.sum(psd) * df
            m1 = np.sum(freqs * psd) * df
            return m1 / m0 if m0 > 0 else np.nan

        def median_frequency():
            cumulative_power = np.cumsum(psd) * df
            idx = np.searchsorted(cumulative_power, cumulative_power[-1] / 2)
            return freqs[idx] if idx < len(freqs) else np.nan

        def mode_frequency():
            return freqs[np.argmax(psd)] if len(psd) > 0 else np.nan

        def spectral_skewness():
            m2 = np.sum((freqs ** 2) * psd) * df
            m3 = np.sum((freqs ** 3) * psd) * df
            return m3 / (m2 ** 1.5) if m2 > 0 else np.nan

        def bandwidth_95():
            cumulative_power = np.cumsum(psd) * df
            total = cumulative_power[-1]
            low_idx = np.searchsorted(cumulative_power, total * 0.025)
            high_idx = np.searchsorted(cumulative_power, total * 0.975)
            if high_idx < len(freqs) and low_idx < high_idx:
                return freqs[high_idx] - freqs[low_idx]
            return 0

        def cumulative_psd_percentile(percent):
            cp = np.cumsum(psd)
            cp /= cp[-1]
            idx = np.searchsorted(cp, percent / 100.0)
            return freqs[idx] if idx < len(freqs) else np.nan

        return {
            "total_power": total_power(),
            "mean_frequency": mean_frequency(),
            "median_frequency": median_frequency(),
            "mode_frequency": mode_frequency(),
            "spectral_skewness": spectral_skewness(),
            "bandwidth_95": bandwidth_95(),
            "f25": cumulative_psd_percentile(25),
            "f50": cumulative_psd_percentile(50),
            "f75": cumulative_psd_percentile(75),
        }

    def run(self, freqs: np.ndarray, psd: np.ndarray, label: str = "") -> Self:
        """
        Compute features from PSD.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis of PSD.
        psd : np.ndarray
            Power spectral density values.
        label : str
            Optional label (e.g., "PSD_plus" or "PSD_minus") to differentiate outputs.

        Returns
        -------
        Self
        """
        features = self._extract_features(freqs, psd)
        features = {f"{label}_{k}" if label else k: v for k, v in features.items()}
        self.feature_data_ = pd.DataFrame([features])
        return self
