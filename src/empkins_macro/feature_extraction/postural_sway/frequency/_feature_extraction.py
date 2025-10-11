# Module for PSD feature extraction of postural cop data.
from typing import Self

import numpy as np
import pandas as pd
from tpcp import Algorithm


class PsdFeatureExtraction(Algorithm):
    _action_methods = "run"

    feature_data_: pd.DataFrame

    def __init__(self):
        """
        Initializes the RSA Feature Extraction algorithm.
        """
        self.feature_data_ = pd.DataFrame()

    def _extract_features(self, frequency_hz: np.ndarray, psd: np.ndarray) -> dict:
        df = frequency_hz[1] - frequency_hz[0]

        def total_power():
            return np.sum(psd) * df

        def mean_frequency():
            m0 = np.sum(psd) * df
            m1 = np.sum(frequency_hz * psd) * df
            return m1 / m0 if m0 > 0 else np.nan

        def median_frequency():
            cumulative_power = np.cumsum(psd) * df
            idx = np.searchsorted(cumulative_power, cumulative_power[-1] / 2)
            return frequency_hz[idx] if idx < len(frequency_hz) else np.nan

        def mode_frequency():
            return frequency_hz[np.argmax(psd)] if len(psd) > 0 else np.nan

        def spectral_skewness():
            m2 = np.sum((frequency_hz ** 2) * psd) * df
            m3 = np.sum((frequency_hz ** 3) * psd) * df
            return m3 / (m2 ** 1.5) if m2 > 0 else np.nan

        def bandwidth_95():
            cumulative_power = np.cumsum(psd) * df
            total = cumulative_power[-1]
            low_idx = np.searchsorted(cumulative_power, total * 0.025)
            high_idx = np.searchsorted(cumulative_power, total * 0.975)
            if len(frequency_hz) > high_idx > low_idx:
                return frequency_hz[high_idx] - frequency_hz[low_idx]
            return 0

        def cumulative_psd_percentile(percent):
            cp = np.cumsum(psd)
            cp /= cp[-1]
            idx = np.searchsorted(cp, percent / 100.0)
            return frequency_hz[idx] if idx < len(frequency_hz) else np.nan

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

    def run(self, data: pd.DataFrame, label: str = "") -> Self:

        # TODO: define standardised format for PSD data

        if not isinstance(data, pd.DataFrame):
            raise TypeError("psd_data must be a pandas DataFrame.")

        frequency_hz = data.index.to_numpy()
        data = data.squeeze().to_numpy()

        features = self._extract_features(frequency_hz, data)
        features = {f"{label}_{k}" if label else k: v for k, v in features.items()}
        features = pd.DataFrame([features])
        features = features.stack().droplevel(0).to_frame(name="value")
        features.index.name = "feature"
        self.feature_data_ = features

        return self
