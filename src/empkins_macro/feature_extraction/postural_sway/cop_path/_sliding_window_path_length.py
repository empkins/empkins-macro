# Module for sliding window path length calculation of postural cop data.

from typing import Any

import numpy as np
import pandas as pd
from biopsykit.utils.array_handling import sliding_window
from tpcp import Algorithm


class SlidingWindowDistanceCalculation(Algorithm):
    _action_methods = "run"

    feature_data_: pd.DataFrame

    def __init__(self):
        """
        Initializes the SlidingWindowDistanceCalculation algorithm.
        """
        self.feature_data_ = pd.DataFrame()

    def _sliding_window_path_length(self, cop_data: pd.DataFrame, sampling_rate_hz: float, window_seconds: int,
                                    overlap_percent: float) -> \
            list[Any]:
        """
        Calculate the sum of COP distances within sliding windows.
        """

        sw = sliding_window(
            data=cop_data,
            sampling_rate=sampling_rate_hz,
            window_sec=window_seconds,
            overlap_percent=overlap_percent
        )
        col_nr = cop_data.columns.get_loc("distance")

        sum_distances = []
        for i, window in enumerate(sw):
            window_distance = np.nansum(window[:, col_nr])
            sum_distances.append({
                "window_index": i,
                "distance_sum": window_distance
            })

        return sum_distances

    def run(self, cop_data: pd.DataFrame, sampling_rate_hz: float, window_seconds: int, overlap_percent: float) -> list[
        Any]:
        """
        Run the sliding window path length calculation algorithm.

        Parameters
        ----------
        cop_data : pd.DataFrame
            DataFrame containing COP distance data with 'x', 'y' and 'distance' column.
        sampling_rate_hz : float
            Sampling rate in Hz.
        window_seconds : int
            Length of each sliding window in seconds.
        overlap_percent : float
            Overlap percentage between consecutive windows.

        Returns
        -------
        list[Any]
            List of dictionaries containing window index and distance sum for each window.
        """

        if not isinstance(cop_data, pd.DataFrame):
            raise TypeError("cop_data must be a pandas DataFrame containing distance column.")

        distance_sums = self._sliding_window_path_length(
            cop_data=cop_data,
            sampling_rate_hz=sampling_rate_hz,
            window_seconds=window_seconds,
            overlap_percent=overlap_percent
        )
        self.feature_data_ = pd.DataFrame(distance_sums)
        return distance_sums
